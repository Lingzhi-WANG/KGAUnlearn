from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchEncoding
import torch

# MAX_DATASET_SIZE = 220000
# TRAIN_SET_SIZE = 200000
# VALID_SET_SIZE = 20000


class TRANS(Dataset):
    def __init__(self, data_file, src, tgt):
        self.data = self.load_data(data_file, src, tgt)

    def load_data(self, data_file, src, tgt):
        Data = {}
        with open(data_file+f".{src}", 'rt', encoding='utf-8') as sf:
            with open(data_file+f".{tgt}", 'rt', encoding='utf-8') as tf:
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
            splits = sample['src'].split('<|endoftext|>')
            tokens = []
            for split in splits:
                tokens.extend(tokenizer.tokenize(split.strip()))
                tokens.append('<|endoftext|>')
            batch_inputs.append(torch.LongTensor([0] + tokenizer.convert_tokens_to_ids(tokens[:-1]) + [2]))
            # batch_inputs.append(sample['src'])
            batch_targets.append(sample['tgt'])
        batch_inputs = pad_sequence(batch_inputs, padding_value=1, batch_first=True)
        attn_mask = (batch_inputs != 1).long()
        batch_data = {
            'input_ids': batch_inputs,
            'attention_mask': attn_mask,
        }
        batch_data = BatchEncoding(batch_data)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch_targets,
                padding=True,
                max_length=args.max_target_length,
                truncation=True,
                return_tensors="pt"
            )["input_ids"]
            batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
            end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
            for idx, end_idx in enumerate(end_token_index):
                labels[idx][end_idx + 1:] = -100
            batch_data['labels'] = labels
        return batch_data

    return DataLoader(
        dataset, batch_size=(batch_size if batch_size else args.batch_size), shuffle=shuffle, collate_fn=collote_fn)

