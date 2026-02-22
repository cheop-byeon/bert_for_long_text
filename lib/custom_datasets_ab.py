import torch
from torch.utils.data import Dataset


class TokenizedDatasetAplusB(Dataset):
    """Dataset for tokens with labels"""
    def __init__(self, tokens, labels):
        self.input_ids = tokens['input_ids']
        self.attention_mask = tokens['attention_mask']
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


def collate_fn_pooled_tokens_AplusB(data):
    """Optimized collate function using zip"""
    input_ids, attention_mask, labels = zip(*data)
    return [list(input_ids), list(attention_mask), list(labels)]