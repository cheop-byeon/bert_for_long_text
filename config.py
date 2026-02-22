import torch
from functools import lru_cache

MODEL_LOAD_FROM_FILE = True
MODEL_PATH = "../bert-base-uncased/"
# MODEL_PATH = "../distillbert-base-uncased"
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

DEFAULT_PARAS_GLOVE = {
    'vocab_size': 18766,
    'glove_dim': 300,
    'glove_file': "./data/glove/glove_300d.npy",
    'word2id_file': "./data/glove/word2id.npy",

    'emb_method': 'glove',
    'enc_method': 'mean',
    'hidden_size': 200,
    'out_size': 64,
    'num_labels': 2,

    'use_gpu': True,
    'seed': 2020,
    'gpu_id': 0,

    'dropout': 0.5,
    'epochs': 10,

    'lr': 1e-3,
    'weight_decay': 1e-4,
    'batch_size': 64,
    'device': 'cuda:0'

}