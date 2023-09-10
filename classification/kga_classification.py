import random
import argparse
import copy
import sys
import time
import numpy as np
import os
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    Subset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)

from pytorch_transformers import (
    DistilBertConfig,
    DistilBertTokenizer,
)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils import (
    DonData, convert_examples_to_features, evaluate_multilabels, 
    apply_threshs, tune_threshs, multihot_to_label_lists, subsample
)

from run_classification import (
    sigmoid, evaluate, DesignSubset, manual_seed,
    DistilBertForMultilabelSequenceClassification
)

from data import SubsetDataWrapper

from arg import parse_args


def main():
    args, parser = parse_args()
    print(args)
    manual_seed(seed=args.seed)

    max_seq_length = args.max_seq_len
    don_data = DonData(path=args.data)

    model_name = 'distilbert'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = DistilBertConfig.from_pretrained(model_name, num_labels=len(don_data.all_lbls))
    tokenizer = DistilBertTokenizer.from_pretrained(model_name, do_lower_case=True)
    model = DistilBertForMultilabelSequenceClassification.from_pretrained(model_name, config=config)
    model.to(device)

    if args.do_unlearn:
        print("************Unlearning**************")
        assert args.new_model_path is not None and args.forget_model_path is not None
        assert args.file_removals is not None and args.file_as_new is not None

        print('loading models...')
        print('original model: ', args.model_path)
        print('new model: ', args.new_model_path)
        print('forget model: ', args.forget_model_path)
        if torch.cuda.is_available():
            model = torch.load(args.model_path)
            new_model = torch.load(args.new_model_path)
            forget_model = torch.load(args.forget_model_path)
        else:
            model = torch.load(args.model_path, map_location='cpu')
            new_model = torch.load(args.new_model_path, map_location='cpu')
            forget_model = torch.load(args.forget_model_path, map_location='cpu')

        original_model = copy.deepcopy(model)
        original_model.eval()
        new_model.eval()
        forget_model.eval()
        for p1, p2, p3 in zip(new_model.parameters(), forget_model.parameters(), original_model.parameters()):
            p1.requires_grad = False
            p2.requires_grad = False
            p3.requires_grad = False

        print("Loading training data")
        train_data = don_data.train()
        print('construct training data tensor')
        train_data = convert_examples_to_features(examples=train_data, max_seq_length=max_seq_length, tokenizer=tokenizer)
        dev_data = convert_examples_to_features(examples=don_data.dev(), max_seq_length=max_seq_length, tokenizer=tokenizer)


        scrub_list = []
        with open(args.file_removals, 'r') as f:
            for line in f:
                scrub_list.append(int(line.strip()))
        scrub_dataset = Subset(train_data, scrub_list)
        scrub_loader = torch.utils.data.DataLoader(scrub_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

        new_list = []
        with open(args.file_as_new, 'r') as f:
            for line in f:
                new_list.append(int(line.strip()))
        new_dataset = Subset(train_data, new_list)
        new_loader = torch.utils.data.DataLoader(new_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

        residual_dataset = SubsetDataWrapper(train_data, include_indices=None, exclude_indices=scrub_list+new_list)
        if args.sample_ratio is not None:
            assert 0 <= args.sample_ratio <= 1
            select_list = [idx for idx in range(len(residual_dataset)) if random.random() <= args.sample_ratio]
            residual_dataset = Subset(residual_dataset, select_list)
        residual_loader = torch.utils.data.DataLoader(residual_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
        print('size of data: ', len(scrub_dataset), len(new_dataset), len(residual_dataset))

        no_decay = {'bias', 'LayerNorm.weight'}
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    ],
                'weight_decay': args.weight_decay,
            },
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                    ],
                'weight_decay': 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon,
        )
        scheduler = WarmupLinearSchedule(
            optimizer=optimizer,
            warmup_steps=args.warmup_steps,
            t_total=min(args.max_update, args.max_epoch*len(residual_loader)),
        )

        model.eval()
        dev_res = evaluate(dev_data, model, args.batch_size)
        dev_mat = apply_threshs(probas=dev_res['pred'], threshs=[0.5 for _ in range(dev_res['pred'].shape[1])])
        dev_res = evaluate_multilabels(
            y=multihot_to_label_lists(dev_res['truth'], don_data.label_map),
            y_preds=multihot_to_label_lists(dev_mat, don_data.label_map),
            do_print=False,
        )
        print("original dev eval: Macro-f1 %.4f, Micro-f1 %.4f" % (dev_res['Macro']['f1'], dev_res['Micro']['f1']))
        model.train()
        end = False
        step = 0
        inner_step = 0
        total_unlearn_time = 0
        cur_time = time.time()
        os.makedirs(args.save_path, exist_ok=True)
        for epoch in range(args.max_epoch):
            scrub_iterator = iter(scrub_loader)
            new_iterator = iter(new_loader)
            for cur_step, batch_remain in enumerate(residual_loader):
                inner_step += 1
                batch_remain = tuple(t.to(device) for t in batch_remain)
                inputs = {
                    'input_ids': batch_remain[0],
                    'attention_mask': batch_remain[1],
                    'labels': batch_remain[3],
                }
                pred_logits = F.log_softmax(model(**inputs), dim=-1)
                tgt_logits = F.log_softmax(original_model(**inputs), dim=-1).detach()
                loss_r = F.kl_div(input=pred_logits, target=tgt_logits, log_target=True, reduction='mean')
                (loss_r * (1 - args.retain_loss_ratio) / args.inner_step).backward()

                if inner_step % args.inner_step == 0:
                    try:
                        batch_forget = next(scrub_iterator)
                    except StopIteration:
                        scrub_iterator = iter(scrub_loader)
                        batch_forget = next(scrub_iterator)
                    try:
                        batch_new = next(new_iterator)
                    except StopIteration:
                        new_iterator = iter(new_loader)
                        batch_new = next(new_iterator)

                    batch_forget = tuple(t.to(device) for t in batch_forget)
                    inputs = {
                        'input_ids': batch_forget[0],
                        'attention_mask': batch_forget[1],
                        'labels': batch_forget[3],
                    }
                    pred_logits = F.log_softmax(model(**inputs), dim=-1)
                    tgt_logits = F.log_softmax(forget_model(**inputs), dim=-1).detach()
                    loss_align = F.kl_div(input=pred_logits, target=tgt_logits, log_target=True, reduction='mean')

                    batch_new = tuple(t.to(device) for t in batch_new)
                    inputs = {
                        'input_ids': batch_new[0],
                        'attention_mask': batch_new[1],
                        'labels': batch_new[3],
                    }
                    pred_logits = F.log_softmax(new_model(**inputs), dim=-1).detach()
                    tgt_logits = F.log_softmax(original_model(**inputs), dim=-1).detach()
                    tgt_align = F.kl_div(input=pred_logits, target=tgt_logits, log_target=True, reduction='mean')
                    loss_align = torch.abs(loss_align - tgt_align.item())
                    loss_align.backward()

                    total_loss = loss_align.item() + args.retain_loss_ratio * loss_r.item()

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                    if args.eval_update >= 0 and (step + 1) % args.eval_update == 0:
                        total_unlearn_time += time.time() - cur_time
                        model.eval()
                        print("train step %d: total_loss %.4f, loss in residual %.4f, loss in align %.4f" % ((step+1), total_loss, loss_r.item(), loss_align.item()))

                        dev_res = evaluate(dev_data, model, args.batch_size)
                        dev_mat = apply_threshs(probas=dev_res['pred'], threshs=[0.5 for _ in range(dev_res['pred'].shape[1])])
                        dev_res = evaluate_multilabels(
                            y=multihot_to_label_lists(dev_res['truth'], don_data.label_map),
                            y_preds=multihot_to_label_lists(dev_mat, don_data.label_map),
                            do_print=False,
                        )
                        print("dev eval: Macro-f1 %.4f, Micro-f1 %.4f" % (dev_res['Macro']['f1'], dev_res['Micro']['f1']))

                        scrub_res = evaluate(scrub_dataset, model, args.batch_size)
                        scrub_mat = apply_threshs(probas=scrub_res['pred'], threshs=[0.5 for _ in range(scrub_res['pred'].shape[1])])
                        scrub_res = evaluate_multilabels(
                            y=multihot_to_label_lists(scrub_res['truth'], don_data.label_map),
                            y_preds=multihot_to_label_lists(scrub_mat, don_data.label_map),
                            do_print=False,
                        )
                        print("forget set eval: Macro-f1 %.4f, Micro-f1 %.4f" % (scrub_res['Macro']['f1'], scrub_res['Micro']['f1']))

                        residual_res = evaluate(residual_dataset, model, args.batch_size)
                        residual_mat = apply_threshs(probas=residual_res['pred'], threshs=[0.5 for _ in range(residual_res['pred'].shape[1])])
                        residual_res = evaluate_multilabels(
                            y=multihot_to_label_lists(residual_res['truth'], don_data.label_map),
                            y_preds=multihot_to_label_lists(residual_mat, don_data.label_map),
                            do_print=False,
                        )
                        print("remain set eval: Macro-f1 %.4f, Micro-f1 %.4f" % (residual_res['Macro']['f1'], residual_res['Micro']['f1']))
                        model.train()

                        if args.save_update >= 0 and (step + 1) % args.save_update == 0:
                            torch.save(model, args.save_path+'/ckpt_%d.pt' % (step+1))
                            print('save internal checkpoint at step %d to %s' % ((step+1), args.save_path+'/ckpt_%d.pt'))
                        cur_time = time.time()
                    elif args.print_loss >= 0 and (step + 1) % args.print_loss == 0:
                        print("train step %d: total_loss %.4f, loss in residual %.4f, loss in align %.4f" % ((step + 1), total_loss, loss_r.item(), loss_align.item()))

                step += 1
                if step >= args.max_update:
                    end = True
                    break

            model.eval()
            print("epoch %d end!" % (epoch+1))
            total_unlearn_time += time.time() - cur_time
            print('current total used time:', total_unlearn_time)
            dev_res = evaluate(dev_data, model, args.batch_size)
            dev_mat = apply_threshs(probas=dev_res['pred'], threshs=[0.5 for _ in range(dev_res['pred'].shape[1])])
            dev_res = evaluate_multilabels(
                y=multihot_to_label_lists(dev_res['truth'], don_data.label_map),
                y_preds=multihot_to_label_lists(dev_mat, don_data.label_map),
                do_print=False,
            )
            print("dev eval: Macro-f1 %.4f, Micro-f1 %.4f" % (dev_res['Macro']['f1'], dev_res['Micro']['f1']))

            scrub_res = evaluate(scrub_dataset, model, args.batch_size)
            scrub_mat = apply_threshs(probas=scrub_res['pred'],
                                      threshs=[0.5 for _ in range(scrub_res['pred'].shape[1])])
            scrub_res = evaluate_multilabels(
                y=multihot_to_label_lists(scrub_res['truth'], don_data.label_map),
                y_preds=multihot_to_label_lists(scrub_mat, don_data.label_map),
                do_print=False,
            )
            print(
            "forget set eval: Macro-f1 %.4f, Micro-f1 %.4f" % (scrub_res['Macro']['f1'], scrub_res['Micro']['f1']))

            residual_res = evaluate(residual_dataset, model, args.batch_size)
            residual_mat = apply_threshs(probas=residual_res['pred'],
                                         threshs=[0.5 for _ in range(residual_res['pred'].shape[1])])
            residual_res = evaluate_multilabels(
                y=multihot_to_label_lists(residual_res['truth'], don_data.label_map),
                y_preds=multihot_to_label_lists(residual_mat, don_data.label_map),
                do_print=False,
            )
            print("remain set eval: Macro-f1 %.4f, Micro-f1 %.4f" % (
            residual_res['Macro']['f1'], residual_res['Micro']['f1']))
            model.train()

            if end:
                break

        print('finish unlearning!')
        print('total used time:', total_unlearn_time)
        torch.save(model, args.save_path+'/ckpt_final.pt')
        print('save model to %s' %args.save_path+'/ckpt_final.pt')
        model.eval()
        dev_res = evaluate(dev_data, model, args.batch_size)
        threshs = tune_threshs(probas=dev_res['pred'], truth=dev_res['truth'])
        test_data = convert_examples_to_features(
            examples=don_data.test(),
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
        )
        test_res = evaluate(test_data, model, args.batch_size)
        test_mat = apply_threshs(probas=test_res['pred'], threshs=threshs)
        test_res = evaluate_multilabels(
            y=multihot_to_label_lists(test_res['truth'], don_data.label_map),
            y_preds=multihot_to_label_lists(test_mat, don_data.label_map),
            do_print=False,
        )
        scrub_res = evaluate(scrub_dataset, model, args.batch_size)
        scrub_mat = apply_threshs(probas=scrub_res['pred'], threshs=threshs)
        scrub_res = evaluate_multilabels(
            y=multihot_to_label_lists(scrub_res['truth'], don_data.label_map),
            y_preds=multihot_to_label_lists(scrub_mat, don_data.label_map),
            do_print=False,
        )
        residual_res = evaluate(residual_dataset, model, args.batch_size)
        residual_mat = apply_threshs(probas=residual_res['pred'], threshs=threshs)
        residual_res = evaluate_multilabels(
            y=multihot_to_label_lists(residual_res['truth'], don_data.label_map),
            y_preds=multihot_to_label_lists(residual_mat, don_data.label_map),
            do_print=False,
        )
        print("final results: ")
        print("test set eval: Macro-f1 %.4f, Micro-f1 %.4f" % (test_res['Macro']['f1'], test_res['Micro']['f1']))
        print("forget set eval: Macro-f1 %.4f, Micro-f1 %.4f" % (scrub_res['Macro']['f1'], scrub_res['Micro']['f1']))
        print("remain set eval: Macro-f1 %.4f, Micro-f1 %.4f" % (residual_res['Macro']['f1'], residual_res['Micro']['f1']))
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
