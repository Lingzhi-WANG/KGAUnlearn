import copy
import os
import argparse
import logging
import json
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import random
import math
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers import AdamW, get_scheduler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import MarianModel, MarianMTModel, MarianConfig
from transformers import BartModel, BartConfig
from sacrebleu.metrics import BLEU

from arg import parse_args
from data import TRANS, get_dataLoader
from run_translation import seed_everything, train_loop, test_loop, train, test, InverseSquareRootSchedule


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")


def more_args():
    _, parser = parse_args()

    parser.add_argument("--new_model_dir", default=None, type=str)
    parser.add_argument("--forget_model_dir", default=None, type=str)
    parser.add_argument("--train_model_dir", default=None, type=str)

    parser.add_argument("--num_train_updates", default=2000, type=int, help="Total number of updates to perform.")
    parser.add_argument("--do_unlearn", action="store_true", help="Whether to run unlearning.")

    parser.add_argument("--forget_file", default=None, type=str, help="The input forget set.")
    parser.add_argument("--new_file", default=None, type=str, help="The input new set.")

    parser.add_argument("--retain_loss_ratio", default=0.1, type=float, help="Ratio for remaining loss.")
    parser.add_argument("--stop_value", default=None, type=float, help="")
    parser.add_argument("--stop_portion", default=None, type=float, help="")

    return parser.parse_args()


def unlearn(
        args, train_dataset, dev_dataset, forget_dataset, new_dataset,
        train_model, forget_model, new_model, model, tokenizer):
    train_dataloader = get_dataLoader(args, train_dataset, model, tokenizer, shuffle=True)
    forget_dataloader = get_dataLoader(args, forget_dataset, model, tokenizer, shuffle=True)
    new_dataloader = get_dataLoader(args, new_dataset, model, tokenizer, shuffle=True)
    dev_dataloader = get_dataLoader(args, dev_dataset, model, tokenizer, shuffle=False)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon
    )
    lr_scheduler = InverseSquareRootSchedule(
        warmup_init_lr=-1,
        warmup_updates=args.warmup_steps,
        lr=args.learning_rate,
        optimizer=optimizer
    )
    # Train!
    logger.info("***** Running unlearning *****")
    logger.info(f"Num examples - {len(train_dataset)} (forget {len(forget_dataset)}) (new {len(new_dataset)})")
    logger.info(f"Num Updates - {args.num_train_updates}")
    model.train()

    total_updates = 0
    total_steps = 0
    stop_value = args.stop_value
    stop = False
    forget_iterator = iter(forget_dataloader)
    new_iterator = iter(new_dataloader)
    while total_updates < args.num_train_updates and not stop:
        for step, batch_data in enumerate(train_dataloader, start=1):
            total_steps += 1
            batch_data = batch_data.to(args.device)
            outputs = model(**batch_data)
            tgt_outputs = train_model(**batch_data)
            pred_logits = F.log_softmax(outputs.logits, dim=-1)
            tgt_logits = F.log_softmax(tgt_outputs.logits, dim=-1).detach()
            loss_remain = F.kl_div(input=pred_logits, target=tgt_logits, log_target=True, reduction='batchmean')
            (loss_remain * args.retain_loss_ratio / args.update_freq).backward()

            if total_steps % args.update_freq == 0:
                total_updates += 1

                try:
                    batch_forget = next(forget_iterator)
                except StopIteration:
                    forget_iterator = iter(forget_dataloader)
                    batch_forget = next(forget_iterator)
                try:
                    batch_new = next(new_iterator)
                except StopIteration:
                    new_iterator = iter(new_dataloader)
                    batch_new = next(new_iterator)

                batch_forget = batch_forget.to(args.device)
                outputs = model(**batch_forget)
                tgt_outputs = forget_model(**batch_forget)
                pred_logits = F.log_softmax(outputs.logits, dim=-1)
                tgt_logits = F.log_softmax(tgt_outputs.logits, dim=-1).detach()
                loss_align = F.kl_div(input=pred_logits, target=tgt_logits, log_target=True, reduction='batchmean')

                batch_new = batch_new.to(args.device)
                outputs = train_model(**batch_new)
                tgt_outputs = new_model(**batch_new)
                pred_logits = F.log_softmax(outputs.logits, dim=-1).detach()
                tgt_logits = F.log_softmax(tgt_outputs.logits, dim=-1).detach()
                tgt_align = F.kl_div(input=pred_logits, target=tgt_logits, log_target=True, reduction='batchmean')
                loss_align = torch.abs(loss_align - tgt_align.item())
                loss_align.backward()

                if stop_value is None and args.stop_portion is not None:
                    stop_value = loss_align.item() * args.stop_portion
                    logger.info(f'Set stop value as {stop_value} (Portion {args.stop_portion})')

                optimizer.step()
                lr_scheduler.step(total_updates)
                optimizer.zero_grad()

                total_loss = loss_align.item() + args.retain_loss_ratio * loss_remain.item()

                if total_updates % 200 == 0:
                    logger.info(f'Train  Step {total_updates}/{args.num_train_updates}: Loss {total_loss:>7f}')
                    logger.info(f'Align Loss {loss_align.item():>7f} Retain Loss {loss_remain.item():>7f}')
                    dev_bleu = test_loop(args, dev_dataloader, model, tokenizer)
                    if isinstance(dev_bleu, tuple):
                        dev_bleu = dev_bleu[0]
                    logger.info(f'Dev: BLEU - {dev_bleu:0.4f}')
                    forget_bleu = test_loop(args, get_dataLoader(args, forget_dataset, model, tokenizer, shuffle=False), model, tokenizer)
                    if isinstance(forget_bleu, tuple):
                        forget_bleu = forget_bleu[0]
                    logger.info(f'Forget: BLEU - {forget_bleu:0.4f}')
                    if stop_value is not None and loss_align.item() <= stop_value:
                        stop = True
                        break
                    model.train()
            if total_updates >= args.num_train_updates or stop:
                break
    save_weight = f'update_{total_updates}_dev_bleu_{dev_bleu:0.4f}_forget_bleu_{forget_bleu:0.4f}_weights.bin'
    torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))


if __name__ == '__main__':
    args = more_args()

    if args.do_unlearn:
        assert args.new_model_dir is not None
        assert args.forget_model_dir is not None
        assert args.train_model_dir is not None
    assert args.output_dir is not None
    if os.path.exists(args.output_dir):
        files = os.listdir(args.output_dir)
        if any([('.bin' in f) for f in files]):
            raise ValueError(f'Output directory ({args.output_dir}) already exists saved models.')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    logger.warning(f'Using {args.device} device, n_gpu: {args.n_gpu}')
    # Set seed
    seed_everything(args.seed)
    # Load pretrained model and tokenizer
    logger.info(f'loading pretrained model of {args.model_checkpoint} and tokenizer of {args.tokenizer_checkpoint} ...')
    toks = args.tokenizer_checkpoint.split(",")
    if len(toks) == 1:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_checkpoint)
    else:
        assert len(toks) == 2 and getattr(args, 'use_bart_init', False)
        tokenizer = [AutoTokenizer.from_pretrained(toks[0]), AutoTokenizer.from_pretrained(toks[1])]

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint).to(args.device)
    if getattr(args, 'use_bart_init', False):
        if args.model_checkpoint == 'bart-base':
            model.model.encoder.embed_tokens = nn.Embedding(58101, 768, padding_idx=58100).to(args.device)
        else:
            raise NotImplementedError
        model.model.shared = None
        nn.init.xavier_uniform_(model.model.encoder.embed_tokens.weight)
        configuration = BartConfig.from_pretrained(args.model_checkpoint)
        tmp_model = BartModel(configuration).to(args.device)
        model.model.encoder.layers = tmp_model.encoder.layers[:4].extend(model.model.encoder.layers)
    elif not args.not_reset_model:
        configuration = MarianConfig.from_pretrained(args.model_checkpoint)
        configuration.decoder_attention_heads = 4
        configuration.decoder_ffn_dim = 1024
        configuration.encoder_attention_heads = 4
        configuration.encoder_ffn_dim = 1024
        configuration.dropout = args.dropout
        configuration.num_beams = args.beam
        configuration.activation_function = "relu"
        model.model = MarianModel(configuration).to(args.device)

    new_model = copy.deepcopy(model)
    if os.path.isdir(args.new_model_dir):
        save_weights = [file for file in os.listdir(args.new_model_dir) if file.endswith('.bin')]
        assert len(save_weights) == 1
        args.new_model_dir = os.path.join(args.new_model_dir, save_weights[0])
    logger.info(f'loading new model from {args.new_model_dir} ...')
    new_model.load_state_dict(torch.load(args.new_model_dir))
    new_model.eval()
    for p in new_model.parameters():
        p.requires_grad = False
        
    forget_model = copy.deepcopy(model)
    if os.path.isdir(args.forget_model_dir):
        save_weights = [file for file in os.listdir(args.forget_model_dir) if file.endswith('.bin')]
        assert len(save_weights) == 1
        args.forget_model_dir = os.path.join(args.forget_model_dir, save_weights[0])
    logger.info(f'loading forget model from {args.forget_model_dir} ...')
    forget_model.load_state_dict(torch.load(args.forget_model_dir))
    forget_model.eval()
    for p in forget_model.parameters():
        p.requires_grad = False

    logger.info(f'loading trained model from {args.train_model_dir} ...')
    model.load_state_dict(torch.load(args.train_model_dir))
    train_model = copy.deepcopy(model)
    train_model.eval()
    for p in train_model.parameters():
        p.requires_grad = False

    with open(os.path.join(args.output_dir, 'args.txt'), 'wt') as f:
        f.write(str(args))

    # Training
    if args.do_unlearn:
        # Set seed
        seed_everything(args.seed)
        train_dataset = TRANS(args.train_file, args.source, args.target)
        dev_dataset = TRANS(args.dev_file, args.source, args.target)
        forget_dataset = TRANS(args.forget_file, args.source, args.target)
        new_dataset = TRANS(args.new_file, args.source, args.target)
        begin_time = time.time()
        unlearn(args, train_dataset, dev_dataset, forget_dataset, new_dataset, train_model, forget_model, new_model, model, tokenizer)
        logger.info(f'Total used time: {(time.time() - begin_time)/60} minutes!')
    else:
        raise NotImplementedError




