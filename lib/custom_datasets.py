import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Dataset for raw texts with labels"""

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class TokenizedDataset(Dataset):
    """Dataset for tokens with labels"""
    def __init__(self, tokens, labels):
        self.input_ids = tokens['input_ids']
        self.attention_mask = tokens['attention_mask']
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


def collate_fn_pooled_tokens(data):
    """Optimized collate function using zip"""
    input_ids, attention_mask, labels = zip(*data)
    return [list(input_ids), list(attention_mask), list(labels)]

class TokenizedDatasetAB(Dataset):
    """Dataset for tokens with labels"""

    def __init__(self, tokens_para, tokens_comm, labels):
        self.input_ids_para = tokens_para['input_ids']
        self.input_ids_comm = tokens_comm['input_ids']
        self.attention_mask_para = tokens_para['attention_mask']
        self.attention_mask_comm = tokens_comm['attention_mask']
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.input_ids_para[idx], self.attention_mask_para[idx],
                self.input_ids_comm[idx], self.attention_mask_comm[idx],
                self.labels[idx])


def collate_fn_pooled_tokens_AB(data):
    """Optimized collate function using zip"""
    input_ids_para, attention_mask_para, input_ids_comm, attention_mask_comm, labels = zip(*data)
    return [list(input_ids_para), list(attention_mask_para), list(input_ids_comm),
            list(attention_mask_comm), list(labels)]