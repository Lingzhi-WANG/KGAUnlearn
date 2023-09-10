import copy
import os
import logging
import numpy as np
import math
import random
import torch
import time
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers import AdamW, get_scheduler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BartTokenizer, BartModel, BartConfig
from tqdm import tqdm

from arg import parse_args
from data import TRANS, get_dataLoader
from run_generation import seed_everything, dev_loop, metric_evaluate, diff_evaluate, InverseSquareRootSchedule


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

    # Prepare optimizer and schedule (linear warmup and decay)
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
    logger.info(f"Num Updates/Epochs - {args.num_train_updates}/{args.num_train_epochs}")
    with open(os.path.join(args.output_dir, 'args.txt'), 'wt') as f:
        f.write(str(args))
    model.train()

    total_updates = 0
    total_steps = 0
    epoch_num = 0
    stop_value = args.stop_value
    stop = False
    forget_iterator = iter(forget_dataloader)
    new_iterator = iter(new_dataloader)
    while (total_updates < args.num_train_updates or epoch_num < args.num_train_epochs) and not stop:
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
                    epoch_num += 1
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
                    dev_loss = dev_loop(args, dev_dataloader, model)
                    logger.info(f'Dev: Loss - {dev_loss:0.4f}')
                    forget_loss = dev_loop(args, get_dataLoader(args, forget_dataset, model, tokenizer, shuffle=False), model)
                    logger.info(f'Forget: Loss - {forget_loss:0.4f}')
                    if stop_value is not None and loss_align.item() <= stop_value:
                        stop = True
                        break
                    model.train()
            if (epoch_num >= args.num_train_epochs and total_updates >= args.num_train_updates) or stop:
                break
    save_weight = f'update_{total_updates}_dev_loss_{dev_loss:0.4f}_forget_loss_{forget_loss:0.4f}_weights.bin'
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
    logger.info(f'Initializing model ...')
    tokenizer = BartTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint).to(args.device)
    if not args.use_bart_init:
        configuration = BartConfig.from_pretrained(args.model_checkpoint)
        configuration.d_model = 512
        configuration.decoder_attention_heads = 4
        configuration.decoder_ffn_dim = 1024
        configuration.encoder_attention_heads = 4
        configuration.encoder_ffn_dim = 1024
        configuration.dropout = args.dropout
        configuration.num_beams = args.beam
        model.model = BartModel(configuration).to(args.device)
        model.lm_head = nn.Linear(512, 50265, bias=False).to(args.device)
        torch.nn.init.xavier_uniform_(model.lm_head.weight)

    new_model = copy.deepcopy(model)
    if os.path.isdir(args.new_model_dir):
        save_weights = [file for file in os.listdir(args.new_model_dir) if file.endswith('.bin')]
        args.new_model_dir = os.path.join(args.new_model_dir, save_weights[-1])
    logger.info(f'loading new model from {args.new_model_dir} ...')
    new_model.load_state_dict(torch.load(args.new_model_dir))
    new_model.eval()
    for p in new_model.model.parameters():
        p.requires_grad = False

    forget_model = copy.deepcopy(model)
    if os.path.isdir(args.forget_model_dir):
        save_weights = [file for file in os.listdir(args.forget_model_dir) if file.endswith('.bin')]
        args.forget_model_dir = os.path.join(args.forget_model_dir, save_weights[-1])
    logger.info(f'loading forget model from {args.forget_model_dir} ...')
    forget_model.load_state_dict(torch.load(args.forget_model_dir))
    forget_model.eval()
    for p in forget_model.model.parameters():
        p.requires_grad = False

    logger.info(f'loading trained model from {args.train_model_dir} ...')
    model.load_state_dict(torch.load(args.train_model_dir))
    train_model = copy.deepcopy(model)
    train_model.eval()
    for p in train_model.model.parameters():
        p.requires_grad = False

    # Unlearning
    if args.do_unlearn:
        train_dataset = TRANS(args.train_file, args.source, args.target)
        dev_dataset = TRANS(args.dev_file, args.source, args.target)
        forget_dataset = TRANS(args.forget_file, args.source, args.target)
        new_dataset = TRANS(args.new_file, args.source, args.target)
        begin_time = time.time()
        unlearn(args, train_dataset, dev_dataset, forget_dataset, new_dataset, train_model, forget_model, new_model, model, tokenizer)
        logger.info(f'Total used time: {(time.time() - begin_time) / 60} minutes!')
    else:
        raise NotImplementedError
