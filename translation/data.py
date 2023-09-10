import json
from torch.utils.data import Dataset, DataLoader
import torch

# MAX_DATASET_SIZE = 220000
# TRAIN_SET_SIZE = 200000
# VALID_SET_SIZE = 20000


class TRANS(Dataset):
    def __init__(self, data_file, src, tgt):
        self.data = self.load_data(data_file, src, tgt)

    def load_data(self, data_file, src, tgt):
        Data = {}
        with open(data_file+f".{src}-{tgt}.{src}", 'rt', encoding='utf-8') as sf:
            with open(data_file+f".{src}-{tgt}.{tgt}", 'rt', encoding='utf-8') as tf:
                for idx, (sline, tline) in enumerate(zip(sf, tf)):
                    sample = {'src': sline.strip(), 'tgt': tline.strip()}
                    Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataLoader(args, dataset, model, tokenizer, batch_size=None, shuffle=False):
    def collote_fn(batch_samples):
        batch_inputs, batch_targets = [], []
        for sample in batch_samples:
            batch_inputs.append(sample['src'])
            batch_targets.append(sample['tgt'])
        if isinstance(tokenizer, list):
            assert len(tokenizer) == 2
            batch_data = tokenizer[0](
                batch_inputs,
                padding=True,
                max_length=args.max_input_length,
                truncation=True,
                return_tensors="pt"
            )
            with tokenizer[1].as_target_tokenizer():
                labels = tokenizer[1](
                    batch_targets,
                    padding=True,
                    max_length=args.max_target_length,
                    truncation=True,
                    return_tensors="pt"
                )["input_ids"]
                batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
                batch_data['labels'] = torch.where(labels == 1, torch.zeros_like(labels).fill_(-100), labels)
        else:
            batch_data = tokenizer(
                batch_inputs,
                padding=True,
                max_length=args.max_input_length,
                truncation=True,
                return_tensors="pt"
            )
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    batch_targets,
                    padding=True,
                    max_length=args.max_target_length,
                    truncation=True,
                    return_tensors="pt"
                )["input_ids"]
                try:
                    batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
                except:
                    batch_data['decoder_input_ids'] = torch.where(
                        labels.eq(tokenizer.eos_token_id), torch.zeros_like(labels).fill_(tokenizer.pad_token_id), labels)
                    batch_data['decoder_input_ids'] = torch.cat(
                        [torch.zeros_like(labels[:, :1]).fill_(tokenizer.eos_token_id),
                         batch_data['decoder_input_ids'][:, :-1]], dim=1
                    )
                end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
                for idx, end_idx in enumerate(end_token_index):
                    labels[idx][end_idx + 1:] = -100
                batch_data['labels'] = labels
        return batch_data

    return DataLoader(
        dataset, batch_size=(batch_size if batch_size else args.batch_size), shuffle=shuffle, collate_fn=collote_fn)
