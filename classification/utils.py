# Code modified from https://github.com/dtuggener/LEDGAR_provision_classification and https://github.com/vsingh-group/LCODEC-deep-unlearning

import itertools
import json
import numpy as np
import re

import torch
from torch.utils.data import TensorDataset

from typing import List, Union, Dict, DefaultDict, Tuple
from collections import defaultdict


def multihot(labels, label_map):
    res = np.zeros(len(label_map))
    for lbl in labels:
        res[label_map[lbl]] = 1.0
    return res


class DonData(object):

    def __init__(self, path):
        self.don_data = split_corpus(path)
        self.all_lbls = list(sorted({
            label
            for lbls in itertools.chain(
                self.don_data.y_train,
                self.don_data.y_test,
                self.don_data.y_dev if self.don_data.y_dev is not None else []
            )
            for label in lbls
        }))
        self.label_map = {
            label: i
            for i, label in enumerate(self.all_lbls)
        }

        total = 0
        self.class_weights = np.zeros(len(self.label_map), dtype=np.float32)
        for sample in self.train():
            self.class_weights += sample['label']
            total += 1
        self.class_weights = total / (len(self.label_map) * self.class_weights)

    def train(self):
        return [{
            'txt': x,
            'label': multihot(lbls, self.label_map),
        } for x, lbls in zip(self.don_data.x_train, self.don_data.y_train)]

    def test(self):
        return [{
            'txt': x,
            'label': multihot(lbls, self.label_map),
        } for x, lbls in zip(self.don_data.x_test, self.don_data.y_test)]

    def dev(self):
        return [{
            'txt': x,
            'label': multihot(lbls, self.label_map),
        } for x, lbls in zip(self.don_data.x_dev, self.don_data.y_dev)]


def convert_examples_to_features(
        examples,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token_segment_id=0,
        sep_token_extra=False,
        pad_on_left=False,
        pad_token_segment_id=0,
        sequence_segment_id=0,
        mask_padding_with_zero=True,
):
    # copy-pasted from https://github.com/huggingface/pytorch-transformers/blob/master/examples/utils_glue.py
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

    all_input_ids = []
    all_input_masks = []
    all_segment_ids = []
    all_label_ids = []
    for ex_ix, example in enumerate(examples):

        tokens = tokenizer.tokenize(example['txt'])

        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        if sep_token_extra:
            tokens += [sep_token]
        segment_ids = [sequence_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        pad_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * pad_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * pad_length) + input_mask
            segment_ids = ([pad_token_segment_id] * pad_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * pad_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * pad_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * pad_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        all_input_ids.append(input_ids)
        all_input_masks.append(input_mask)
        all_segment_ids.append(segment_ids)
        all_label_ids.append(example['label'])

    input_id_tensor = torch.tensor(all_input_ids, dtype=torch.long)
    input_mask_tensor = torch.tensor(all_input_masks, dtype=torch.long)
    segment_id_tensor = torch.tensor(all_segment_ids, dtype=torch.long)
    label_id_tensor = torch.tensor(all_label_ids, dtype=torch.float)

    return TensorDataset(
        input_id_tensor,
        input_mask_tensor,
        segment_id_tensor,
        label_id_tensor,
    )


def evaluate_multilabels(y: List[List[str]], y_preds: List[List[str]],
                         do_print: bool = False) -> DefaultDict[str, Dict[str, float]]:
    """
    Print classification report with multilabels
    :param y: Gold labels
    :param y_preds: Predicted labels
    :param do_print: Whether to print results
    :return: Dict of scores per label and overall
    """
    # Label -> TP/FP/FN -> Count
    label_eval: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
    assert len(y) == len(y_preds), "List of predicted and gold labels are of unequal length"
    for y_true, y_pred in zip(y, y_preds):
        for label in y_true:
            if label in y_pred:
                label_eval[label]['tp'] += 1
            else:
                label_eval[label]['fn'] += 1
        for label in y_pred:
            if label not in y_true:
                label_eval[label]['fp'] += 1

    max_len = max([len(l) for l in label_eval.keys()])
    if do_print:
        print('\t'.join(['Label'.rjust(max_len, ' '), 
            'Prec'.ljust(4, ' '), 'Rec'.ljust(4, ' '), 'F1'.ljust(4, ' '), 'Support']))

    eval_results: DefaultDict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    all_f1, all_rec, all_prec = [], [], []
    for label in sorted(label_eval.keys()):
        cnts = label_eval[label]
        if not cnts['tp'] == 0:
            prec = cnts['tp'] / (cnts['tp'] + cnts['fp'])
            rec = cnts['tp'] / (cnts['tp'] + cnts['fn'])
            f1 = (2 * prec * rec) / (prec + rec)
        else:
            prec, rec, f1 = 0.00, 0.00, 0.00
        eval_results[label]['prec'] = prec
        eval_results[label]['rec'] = rec
        eval_results[label]['f1'] = f1
        eval_results[label]['support'] = cnts['tp'] + cnts['fn']
        all_f1.append(f1)
        all_rec.append(rec)
        all_prec.append(prec)
        if do_print:
            print('\t'.join([label.rjust(max_len, ' '),
                         ('%.4f' % round(prec, 4)).ljust(4, ' '),
                         ('%.4f' % round(rec, 4)).ljust(4, ' '),
                         ('%.4f' % round(f1, 4)).ljust(4, ' '),
                         str(cnts['tp'] + cnts['fn']).rjust(5, ' ')
                         ]))

    eval_results['Macro']['prec'] = sum(all_prec) / len(all_prec)
    eval_results['Macro']['rec'] = sum(all_rec) / len(all_rec)
    if eval_results['Macro']['prec'] + eval_results['Macro']['rec'] == 0:
        eval_results['Macro']['f1'] = 0.0
    else:
        eval_results['Macro']['f1'] = (2 * eval_results['Macro']['prec'] * eval_results['Macro']['rec']) / \
                                  (eval_results['Macro']['prec'] + eval_results['Macro']['rec'])
    eval_results['Macro']['support'] = len(y)

    # Micro
    all_tp = sum(label_eval[label]['tp'] for label in label_eval)
    all_fp = sum(label_eval[label]['fp'] for label in label_eval)
    all_fn = sum(label_eval[label]['fn'] for label in label_eval)
    if all_fp == 0:
        eval_results['Micro']['prec'] = 0
        eval_results['Micro']['rec'] = 0
        eval_results['Micro']['f1'] = 0
    else:
        eval_results['Micro']['prec'] = all_tp / (all_tp + all_fp)
        eval_results['Micro']['rec'] = all_tp / (all_tp + all_fn)
        micro_prec = eval_results['Micro']['prec']
        micro_rec = eval_results['Micro']['rec']
        if micro_prec + micro_rec == 0:
            eval_results['Micro']['f1'] = 0.0
        else:
            eval_results['Micro']['f1'] = (2 * micro_rec * micro_prec) / (micro_rec + micro_prec)
    eval_results['Micro']['support'] = len(y)

    if do_print:
        print('Macro Avg. Rec:', round(eval_results['Macro']['rec'], 4))
        print('Macro Avg. Prec:', round(eval_results['Macro']['prec'], 4))
        print('Macro F1:', round(eval_results['Macro']['f1'], 4))
        print()
        print('Micro Avg. Rec:', round(eval_results['Micro']['rec'], 4))
        print('Micro Avg. Prec:',  round(eval_results['Micro']['prec'], 4))
        print('Micro F1:', round(eval_results['Micro']['f1'], 4))

    return eval_results


def subsample(data, quantile, n_classes):
    class_counts = np.zeros(n_classes, dtype=np.int32)
    for sample in data:
        class_counts += (sample['label'] > 0)

    cutoff = int(np.quantile(class_counts, q=quantile))

    n_to_sample = np.minimum(class_counts, cutoff)

    index_map = {
        i: []
        for i in range(n_classes)
    }
    to_keep = set()
    for ix, sample in enumerate(data):
        if np.sum(sample['label']) > 1:
            to_keep.add(ix)
            n_to_sample -= (sample['label'] > 0)
        else:
            label = np.argmax(sample['label'])
            index_map[label].append(ix)

    for c in range(n_classes):
        to_keep.update(index_map[c][:max(0, n_to_sample[c])])

    return [
        d
        for ix, d in enumerate(data)
        if ix in to_keep
    ]


def apply_threshs(probas, threshs):
    res = np.zeros(probas.shape)

    for i in range(probas.shape[1]):
        res[:, i] = probas[:, i] > threshs[i]

    return res


def tune_threshs(probas, truth):
    res = np.zeros(probas.shape[1])

    assert np.alltrue(probas >= 0.0)
    assert np.alltrue(probas <= 1.0)

    for i in range(probas.shape[1]):
        if np.sum(truth[:, i]) > 4 :
            thresh = max(
                np.linspace(
                    0.0,
                    1.0,
                    num=100,
                ),
                key=lambda t: f1_score(y_true=truth[:, i], y_pred=(probas[:, i] > t), pos_label=1, average='binary')
            )
            res[i] = thresh
        else:
            res[i] = 0.5

    return res


def multihot_to_label_lists(label_array, label_map):
    label_id_to_label = {
        v: k
        for k, v in label_map.items()
    }
    res = []
    for i in range(label_array.shape[0]):
        lbl_set = []
        for j in range(label_array.shape[1]):
            if label_array[i, j] > 0:
                lbl_set.append(label_id_to_label[j])
        res.append(lbl_set)
    return res