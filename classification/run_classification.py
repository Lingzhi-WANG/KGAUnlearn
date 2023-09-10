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
from pytorch_transformers.modeling_distilbert import (
    DistilBertPreTrainedModel,
    DistilBertModel,
)
from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils import (
    DonData, convert_examples_to_features, evaluate_multilabels, 
    apply_threshs, tune_threshs, multihot_to_label_lists, subsample
)

from arg import parse_args


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def evaluate(eval_dataset, model, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_loader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=batch_size
    )

    preds = None
    out_label_ids = None
    model.eval()
    for batch in eval_loader:
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[3]
            }

            logits = model(**inputs)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids,
                    inputs['labels'].detach().cpu().numpy(),
                    axis=0,
                )

    return {
        'pred': sigmoid(preds),
        'truth': out_label_ids,
    }


class DesignSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, exclude_indices, include_indices):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        """
        super(DesignSubset, self).__init__()

        assert include_indices is not None
        assert exclude_indices is not None

        self.dataset = dataset

        self.include_indices = include_indices
        S = set(exclude_indices)
        self.include_indices = [idx for idx in self.include_indices if idx not in S]

        # record important attributes
        if hasattr(dataset, 'statistics'):
            self.statistics = dataset.statistics

    def __len__(self):
        return len(self.include_indices)

    def __getitem__(self, idx):
        real_idx = self.include_indices[idx]
        return self.dataset[real_idx]


class DistilBertForMultilabelSequenceClassification(DistilBertPreTrainedModel):

    def __init__(self, config):
        super(DistilBertForMultilabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        print("Num labels: ", self.num_labels)

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            head_mask=None,
            labels=None,
            class_weights=None,
    ):
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


def train(train_dataset, valid_dataset, model, train_params, label_map, class_weights=None):
    # TODO: magic numbers, defaults in run_glue.py
    batch_size = train_params['batch_size']
    # eval_update = train_params['eval_update']
    updates = train_params['updates']
    weight_decay = train_params['weight_decay']
    learning_rate = train_params['learning_rate']
    adam_epsilon = train_params['adam_epsilon']
    warmup_steps = train_params['warmup_steps']
    # seed = train_params['seed']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_grad_norm = train_params['max_grad_norm']

    print('Train Set Size: ', len(train_dataset))
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
    )

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(device)

    no_decay = {'bias', 'LayerNorm.weight'}
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
                ],
            'weight_decay': weight_decay,
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
        lr=learning_rate,
        eps=adam_epsilon,
    )
    scheduler = WarmupLinearSchedule(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        t_total=updates,
    )

    global_step = 0
    model.zero_grad()
    train_iter = trange(9999, desc='Epoch')
    epoch = 0
    best_model = None
    best_f1 = -999
    total_train_time = 0
    no_better_count = 0
    cur_time = time.time()
    for _ in train_iter:
        epoch_iter = tqdm(train_dataloader, desc="Iteration", ncols=40)
        for step, batch in enumerate(epoch_iter):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[3],
                'class_weights': class_weights,
            }

            logits = model(**inputs)
            loss_fct = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=class_weights)
            loss = loss_fct(logits, inputs['labels'])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

        total_train_time += time.time() - cur_time
        model.eval()
        dev_res = evaluate(valid_dataset, model, batch_size)
        dev_mat = apply_threshs(probas=dev_res['pred'], threshs=[0.5 for _ in range(dev_res['pred'].shape[1])])
        dev_res = evaluate_multilabels(
            y=multihot_to_label_lists(dev_res['truth'], label_map),
            y_preds=multihot_to_label_lists(dev_mat, label_map), do_print=False)
        print("\ndev eval at epoch %d: Macro-f1 %.4f, Micro-f1 %.4f \n" % (epoch, dev_res['Macro']['f1'], dev_res['Micro']['f1']))
        if dev_res['Micro']['f1'] > best_f1:
            print("new best!")
            best_model = model.state_dict()
            best_f1 = dev_res['Micro']['f1']
            no_better_count = 0
        else:
            print("no better! current best %.4f" % best_f1)
            no_better_count += 1
            if no_better_count == 3:
                cur_time = time.time()
                break
        cur_time = time.time()

        epoch += 1
    total_train_time += time.time() - cur_time
    print('finish training! total train time:', total_train_time)

    model.load_state_dict(best_model)
    return model


def manual_seed(seed):
    print("Setting seeds to: ", seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    if args.do_train:  # train-forget or train-new
        print(f"************{args.mode}**************")

        train_params = {
            'seed': args.seed or 0xDEADBEEF,
            'batch_size': args.batch_size or 8,
            'updates': args.max_update or 10000,
            'weight_decay': args.weight_decay or 0.01,
            'learning_rate': args.learning_rate or 5e-5,
            'adam_epsilon': args.adam_epsilon or 1e-8,
            'warmup_steps': args.warmup_steps or 0,
            'max_grad_norm': args.max_grad_norm or 1.0,
        }

        # training
        train_data = don_data.train()
        print('construct training data tensor')
        train_data = convert_examples_to_features(examples=train_data, max_seq_length=max_seq_length, tokenizer=tokenizer)

        forget_list = []
        assert args.file_removals is not None
        with open(args.file_removals, 'r') as f:
            for line in f:
                forget_list.append(int(line.strip()))
        new_list = []
        assert args.file_as_new is not None
        with open(args.file_as_new, 'r') as f:
            for line in f:
                new_list.append(int(line.strip()))

        if args.mode == 'train-forget':
            train_list = forget_list
            exclude_list = new_list
        elif args.mode == 'train-new':
            train_list = new_list
            exclude_list = forget_list
        else:
            raise NotImplementedError

        assert 0 <= args.sample_ratio <= 1
        select_list = [idx for idx in range(len(train_data)) if random.random() <= args.sample_ratio]

        train_dataset = DesignSubset(train_data, exclude_indices=exclude_list, include_indices=train_list+select_list)
        dev_data = convert_examples_to_features(examples=don_data.dev(), max_seq_length=max_seq_length, tokenizer=tokenizer)

        print('start training')
        train(
            train_dataset=train_dataset,
            valid_dataset=dev_data,
            model=model,
            train_params=train_params,
            label_map=don_data.label_map,
            class_weights=None,
        )
        prefix = os.path.dirname(args.save_path)
        os.makedirs(prefix, exist_ok=True)
        torch.save(model, args.save_path)

    elif args.do_test:
        print("************Testing**************")
        if args.test_metric == 'f1':
            if torch.cuda.is_available():
                model = torch.load(args.model_path)
            else:
                model = torch.load(args.model_path, map_location='cpu')
            model.eval()
            print('process dev data')
            dev_data = convert_examples_to_features(
                examples=don_data.dev(),
                max_seq_length=max_seq_length,
                tokenizer=tokenizer,
            )
            dev_res = evaluate(dev_data, model, args.batch_size)
            threshs = tune_threshs(probas=dev_res['pred'], truth=dev_res['truth'])
            if args.test_data == 'test':
                print('process test data')
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
                print("test set eval: Macro-f1 %.4f, Micro-f1 %.4f" % (test_res['Macro']['f1'], test_res['Micro']['f1']))
            elif args.test_data == 'forget':
                print('process forget data')
                train_data = convert_examples_to_features(
                    examples=don_data.train(),
                    max_seq_length=max_seq_length,
                    tokenizer=tokenizer,
                )
                assert args.file_removals is not None
                scrub_list = []
                with open(args.file_removals, 'r') as f:
                    for line in f:
                        scrub_list.append(int(line.strip()))
                scrub_dataset = Subset(train_data, scrub_list)
                scrub_res = evaluate(scrub_dataset, model, args.batch_size)
                scrub_mat = apply_threshs(probas=scrub_res['pred'], threshs=threshs)
                scrub_res = evaluate_multilabels(
                    y=multihot_to_label_lists(scrub_res['truth'], don_data.label_map),
                    y_preds=multihot_to_label_lists(scrub_mat, don_data.label_map),
                    do_print=False,
                )
                print("forget set eval: Macro-f1 %.4f, Micro-f1 %.4f" % (scrub_res['Macro']['f1'], scrub_res['Micro']['f1']))
            else:
                raise NotImplementedError
        elif args.test_metric == 'js':
            model_paths = args.model_path.split(',')
            assert len(model_paths) == 2
            if torch.cuda.is_available():
                model1 = torch.load(model_paths[0])
                model2 = torch.load(model_paths[1])
            else:
                model1 = torch.load(model_paths[0], map_location='cpu')
                model2 = torch.load(model_paths[1], map_location='cpu')
            model1.eval()
            model2.eval()
            if args.test_data == 'test':
                print('process test data')
                test_data = convert_examples_to_features(
                    examples=don_data.test(),
                    max_seq_length=max_seq_length,
                    tokenizer=tokenizer,
                )
                test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=1)
                js_sum = 0.0
                for batch in test_loader:
                    batch = tuple(t.to(device) for t in batch)
                    inputs = {
                        'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'labels': batch[3],
                    }
                    pred_logits1 = F.softmax(model1(**inputs), dim=-1).detach()
                    pred_logits2 = F.softmax(model2(**inputs), dim=-1).detach()
                    mean_logits = (pred_logits1 + pred_logits2) / 2
                    loss1 = (F.kl_div(input=pred_logits1.log(), target=mean_logits, reduction='batchmean') +
                             F.kl_div(input=pred_logits2.log(), target=mean_logits, reduction='batchmean')) / 2
                    js_sum += loss1.item()
                js = js_sum / len(test_loader)
                print("test set eval: JSD %.4f" % js)
            elif args.test_data == 'forget':
                print('process forget data')
                train_data = convert_examples_to_features(
                    examples=don_data.train(),
                    max_seq_length=max_seq_length,
                    tokenizer=tokenizer,
                )
                assert args.file_removals is not None
                scrub_list = []
                with open(args.file_removals, 'r') as f:
                    for line in f:
                        scrub_list.append(int(line.strip()))
                scrub_dataset = Subset(train_data, scrub_list)
                scrub_loader = torch.utils.data.DataLoader(scrub_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
                kl_sum = 0.0
                for batch in scrub_loader:
                    batch = tuple(t.to(device) for t in batch)
                    inputs = {
                        'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'labels': batch[3],
                    }
                    pred_logits1 = F.softmax(model1(**inputs), dim=-1).detach()
                    pred_logits2 = F.softmax(model2(**inputs), dim=-1).detach()
                    mean_logits = (pred_logits1 + pred_logits2) / 2
                    loss1 = (F.kl_div(input=pred_logits1.log(), target=mean_logits, reduction='batchmean') +
                             F.kl_div(input=pred_logits2.log(), target=mean_logits, reduction='batchmean')) / 2
                    kl_sum += loss1.item()
                kl = kl_sum / len(scrub_loader)
                print("forget set eval: JSD %.4f" % kl)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
