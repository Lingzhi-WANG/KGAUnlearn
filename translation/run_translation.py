import copy
import os
import logging
import json
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformers import AdamW, get_scheduler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import MarianModel, MarianConfig
from transformers import BartModel, BartConfig
from sacrebleu.metrics import BLEU

from arg import parse_args
from data import TRANS, get_dataLoader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def weight_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)


class InverseSquareRootSchedule(object):
    """From Fairseq
    """
    def __init__(self, warmup_init_lr, warmup_updates, lr, optimizer):
        super().__init__()

        self.optimizer = optimizer
        self.best = None

        warmup_end_lr = lr
        if warmup_init_lr < 0:
            warmup_init_lr = 0 if warmup_updates > 0 else warmup_end_lr

        # linearly warmup for the first warmup_updates
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * warmup_updates ** 0.5

        # initial learning rate
        self.lr = warmup_init_lr
        self.set_lr(self.lr)

        self.warmup_init_lr = warmup_init_lr
        self.warmup_updates = warmup_updates

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.warmup_updates:
            self.lr = self.warmup_init_lr + num_updates * self.lr_step
        else:
            self.lr = self.decay_factor * num_updates ** -0.5
        self.set_lr(self.lr)
        return self.lr


def train_loop(args, dataloader, model, optimizer, lr_scheduler, epoch, total_loss, total_updates):
    finish_step_num = epoch * len(dataloader)

    model.train()
    for step, batch_data in enumerate(dataloader, start=1):
        if getattr(args, 'use_bart_init', False) and total_updates == getattr(args, "bart_freeze_update", 2000):
            for p in model.model.parameters():
                p.requires_grad = True
            for p in model.lm_head.parameters():
                p.requires_grad = True

        batch_data = batch_data.to(args.device)
        outputs = model(**batch_data)
        if 'bart' in args.model_checkpoint:
            loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.shape[-1]), batch_data['labels'].view(-1),
                                   ignore_index=-100, reduction='mean')
        else:
            loss = outputs.loss
        (loss / args.update_freq).backward()

        if step % args.update_freq == 0:
            total_updates += 1
            optimizer.step()
            if isinstance(lr_scheduler, InverseSquareRootSchedule):
                lr_scheduler.step(total_updates)
            else:
                lr_scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        if step % 1000 == 0:
            logger.info(f'Train Epoch {epoch+1} Step {step}/{len(dataloader)}: Loss {total_loss/(finish_step_num + step):>7f}')
    return total_loss, total_updates


def test_loop(args, dataloader, model, tokenizer):
    preds, labels = [], []
    bleu = BLEU()

    model.eval()
    loss = None
    with torch.no_grad():
        for batch_data in dataloader:
            batch_data = batch_data.to(args.device)
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=args.max_target_length,
                num_beams=args.beam,
            ).cpu().numpy()
            label_tokens = batch_data["labels"].cpu().numpy()

            if isinstance(tokenizer, list):
                decoded_preds = tokenizer[1].batch_decode(generated_tokens, skip_special_tokens=True)
                label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer[1].pad_token_id)
                decoded_labels = tokenizer[1].batch_decode(label_tokens, skip_special_tokens=True)
            else:
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
                decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
            preds += [pred.strip() for pred in decoded_preds]
            labels += [label.strip() for label in decoded_labels]
    if loss is not None:
        return bleu.corpus_score(preds, [labels]).score, loss
    return bleu.corpus_score(preds, [labels]).score


def train(args, train_dataset, dev_dataset, model, tokenizer):
    """ Train the model """
    train_dataloader = get_dataLoader(args, train_dataset, model, tokenizer, shuffle=True)
    dev_dataloader = get_dataLoader(args, dev_dataset, model, tokenizer, shuffle=False)
    t_total = int(len(train_dataloader) * args.num_train_epochs / args.update_freq)
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
    if args.lr_schedule == "inverse_sqrt":
        lr_scheduler = InverseSquareRootSchedule(
            warmup_init_lr=-1,
            warmup_updates=args.warmup_steps,
            lr=args.learning_rate,
            optimizer=optimizer
        )
    else:
        lr_scheduler = get_scheduler(
            'linear',
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total
        )
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"Num examples - {len(train_dataset)}")
    logger.info(f"Num Epochs - {args.num_train_epochs}")
    logger.info(f"Total optimization steps - {t_total}")
    with open(os.path.join(args.output_dir, 'args.txt'), 'wt') as f:
        f.write(str(args))

    total_loss = 0.
    total_updates = 0
    best_bleu = 0.
    no_improve = 0
    saved = None
    if getattr(args, 'use_bart_init', False):  # first freeze most of the bart parameters
        for p in model.model.encoder.layers[5:].parameters():
            p.requires_grad = False
        for p in model.model.decoder.parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = False

    for epoch in range(args.num_train_epochs):
        logger.info(f"Epoch {epoch+1}/{args.num_train_epochs}\n-------------------------------")
        total_loss, total_updates = train_loop(
            args, train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss, total_updates)
        logger.info(f'Train Epoch {epoch+1} End! Loss {total_loss / ((epoch+1) * len(train_dataloader)):>7f}')
        dev_bleu = test_loop(args, dev_dataloader, model, tokenizer)
        if isinstance(dev_bleu, tuple):
            dev_bleu, dev_loss = dev_bleu[0], dev_bleu[1]
            logger.info(f'Dev: BLEU - {dev_bleu:0.4f}, Loss - {dev_loss:0.4f}')
        else:
            logger.info(f'Dev: BLEU - {dev_bleu:0.4f}')
        if dev_bleu > best_bleu:
            no_improve = 0
            best_bleu = dev_bleu
            logger.info(f'Saving new weights to {args.output_dir}...')
            save_weight = f'epoch_{epoch+1}_dev_bleu_{dev_bleu:0.4f}_weights.bin'
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))
            if saved is not None and os.path.exists(os.path.join(args.output_dir, saved)):
                os.remove(os.path.join(args.output_dir, saved))
            saved = save_weight
        else:
            no_improve += 1
        if no_improve >= args.patience:
            logger.info(f'No improvement over {args.patience} epochs!')
            break
    logger.info("Done!")


def test(args, test_dataset, model, tokenizer, save_weights):
    test_dataloader = get_dataLoader(args, test_dataset, model, tokenizer, shuffle=False)
    logger.info('***** Running testing *****')
    if save_weights is None:
        logger.info(f'loading weights from {args.output_dir}...')
        model.load_state_dict(torch.load(args.output_dir))
        bleu = test_loop(args, test_dataloader, model, tokenizer)
        if isinstance(bleu, tuple):
            bleu = bleu[0]
        logger.info(f'Test: BLEU - {bleu:0.4f}')
    else:
        for save_weight in save_weights:
            logger.info(f'loading weights from {save_weight}...')
            model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
            bleu = test_loop(args, test_dataloader, model, tokenizer)
            if isinstance(bleu, tuple):
                bleu = bleu[0]
            logger.info(f'Test: BLEU - {bleu:0.4f}')


def evaluate(args, dataset, base_model, unlearn_model, tokenizer):
    dataloader = get_dataLoader(args, dataset, model, tokenizer, batch_size=1, shuffle=False)
    logger.info('***** Running evaluation *****')

    ppl_dif1, ppl_dif2, ppl_dif3 = [], [], []
    base_model.eval()
    unlearn_model.eval()
    with torch.no_grad():
        for batch_data in dataloader:
            batch_data = batch_data.to(args.device)
            if args.model_checkpoint == 'lstm':
                output1 = base_model(
                    src_tokens=batch_data["input_ids"],
                    src_lengths=batch_data["attention_mask"].sum(dim=1).long(),
                    prev_output_tokens=batch_data['decoder_input_ids'],
                )[0]
                loss1 = F.cross_entropy(output1.view(-1, output1.shape[-1]), batch_data['labels'].view(-1),
                                        ignore_index=-100, reduction='mean')
                ppl1 = math.exp(loss1.item())
                output2 = unlearn_model(
                    src_tokens=batch_data["input_ids"],
                    src_lengths=batch_data["attention_mask"].sum(dim=1).long(),
                    prev_output_tokens=batch_data['decoder_input_ids'],
                )[0]
                loss2 = F.cross_entropy(output2.view(-1, output2.shape[-1]), batch_data['labels'].view(-1),
                                        ignore_index=-100, reduction='mean')
                ppl2 = math.exp(loss2.item())
            else:
                output1 = base_model(**batch_data)
                ppl1 = math.exp(output1.loss.item())
                output2 = unlearn_model(**batch_data)
                ppl2 = math.exp(output2.loss.item())

            ppl_dif1.append(abs(ppl1 - ppl2))
            ppl_dif2.append(abs(ppl1 - ppl2)/ppl1)
            ppl_dif3.append(1.0 if ppl2 - ppl1 > 0 else 0.0)
    ppl_dif1 = sum(ppl_dif1) / len(ppl_dif1)
    ppl_dif2 = sum(ppl_dif2) / len(ppl_dif2)
    ppl_dif3 = sum(ppl_dif3) / len(ppl_dif3)

    logger.info(f'PPL Diff1: {ppl_dif1:0.4f}')
    logger.info(f'PPL Diff2: {ppl_dif2:0.4f}')
    logger.info(f'PPL Diff3: {ppl_dif3:0.4f}')


if __name__ == '__main__':
    args, _ = parse_args()
    if args.do_train and os.path.exists(args.output_dir):
        files = os.listdir(args.output_dir)
        if any(['.bin' in f for f in files]):
            raise ValueError(f'Output directory ({args.output_dir}) already exists saved models.')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    logger.warning(f'Using {args.device} device, n_gpu: {args.n_gpu}')
    # Set seed
    seed_everything(args.seed)
    # Load pretrained model and tokenizer
    logger.info(f'loading model of {args.model_checkpoint} and tokenizer of {args.tokenizer_checkpoint} ...')
    toks = args.tokenizer_checkpoint.split(",")
    if len(toks) == 1:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_checkpoint)
    else:
        assert len(toks) == 2 and getattr(args, 'use_bart_init', False)
        tokenizer = [AutoTokenizer.from_pretrained(toks[0]), AutoTokenizer.from_pretrained(toks[1])]

    if getattr(args, 'use_bart_init', False):
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint).to(args.device)
        if args.model_checkpoint == 'bart-base':
            model.model.encoder.embed_tokens = nn.Embedding(58101, 768, padding_idx=58100).to(args.device)
        else:
            raise NotImplementedError
        model.model.shared = None
        nn.init.xavier_uniform_(model.model.encoder.embed_tokens.weight)
        configuration = BartConfig.from_pretrained(args.model_checkpoint)
        tmp_model = BartModel(configuration).to(args.device)
        model.model.encoder.layers = tmp_model.encoder.layers[:4].extend(model.model.encoder.layers)   
    elif args.not_reset_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint).to(args.device)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint).to(args.device)
        configuration = MarianConfig.from_pretrained(args.model_checkpoint)
        configuration.decoder_attention_heads = 4
        configuration.decoder_ffn_dim = 1024
        configuration.encoder_attention_heads = 4
        configuration.encoder_ffn_dim = 1024
        configuration.dropout = args.dropout
        configuration.num_beams = args.beam
        configuration.activation_function = "relu"
        model.model = MarianModel(configuration).to(args.device)

    # Training
    if args.do_train:
        # Set seed
        seed_everything(args.seed)
        train_dataset = TRANS(args.train_file, args.source, args.target)
        dev_dataset = TRANS(args.dev_file, args.source, args.target)
        train(args, train_dataset, dev_dataset, model, tokenizer)

    # Testing
    elif args.do_test:
        if os.path.isdir(args.output_dir):
            save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
        elif args.output_dir.endswith('.bin'):
            save_weights = None
        else:
            raise NotImplementedError
        test_dataset = TRANS(args.test_file, args.source, args.target)
        test(args, test_dataset, model, tokenizer, save_weights)

    # Evaluating unlearning effects
    elif args.do_evaluate:
        assert args.output_dir is not None and args.unlearn_model_dir is not None
        if os.path.isdir(args.output_dir):
            save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
            args.output_dir = os.path.join(args.output_dir, save_weights[-1])
        if os.path.isdir(args.unlearn_model_dir):
            save_weights = [file for file in os.listdir(args.unlearn_model_dir) if file.endswith('.bin')]
            args.unlearn_model_dir = os.path.join(args.unlearn_model_dir, save_weights[-1])
        base_model = copy.deepcopy(model)
        base_model.load_state_dict(torch.load(args.output_dir))
        model.load_state_dict(torch.load(args.unlearn_model_dir))
        test_dataset = TRANS(args.test_file, args.source, args.target)
        evaluate(args, test_dataset, base_model, model, tokenizer)
