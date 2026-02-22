import torch
from functools import lru_cache

MODEL_LOAD_FROM_FILE = True
MODEL_PATH = "../bert-base-uncased/" # "../distillbert-base-uncased"
VISIBLE_GPUS = "0,1,2,3"
BATCH_SIZE = 16  # with accumulate step, which approximate batch size 16
LEARNING_RATE = 2e-5  # 2e-5 5e-6

# https://wandb.ai/jack-morris/david-vs-goliath/reports/Does-Model-Size-Matter-A-Comparison-of-BERT-and-DistilBERT--VmlldzoxMDUxNzU
# learning_rate: values: [3e-4, 1e-4, 5e-5, 3e-5]


@lru_cache(maxsize=1)
def get_device():
    """Lazily initialize device to avoid redundant checks"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Create device once
device = get_device()

DEFAULT_PARAMS_BERT = {
    'device': device,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE
}

DEFAULT_PARAMS_BERT_WITH_POOLING = {
    'device': device,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'pooling_strategy': 'mean',  # options: ['mean','max'],
    'score': 'dot',  # options: ['euclidean', 'dot', 'bilinear']
    'size': 510,
    'step': 510,  # no overlapping
    'minimal_length': 100
}

DEFAULT_PARAMS_SBERT = {
    'device': device,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'pooling_strategy': 'mean',  # options: ['mean','max'],
    'size': 510,
    'step': 510,  # no overlapping
    'minimal_length': 100,
    'concatenation_sent_rep': True,
    'concatenation_sent_difference': True,
    'concatenation_sent_multiplication': False
}