"""Inference script for trained BERT models on paragraph-commentary classification."""
import os
import sys
import gc
import time
import logging
import random
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split
from transformers import PreTrainedTokenizerFast, AutoModel, AdamW, BertTokenizer, BertModel, AutoConfig

from lib.architecture import BERTSequenceClassificationArch, BERTSequenceClassificationArchForAB
from lib.base_model import Model
from lib.custom_datasets import TokenizedDataset, collate_fn_pooled_tokens, TokenizedDatasetAB, collate_fn_pooled_tokens_AB
from lib.custom_datasets_ab import TokenizedDatasetAplusB, collate_fn_pooled_tokens_AplusB
from lib.text_preprocessors import BERTTokenizer, BERTTokenizerPooled, BERTTokenizerPooledAplusB, BERTTokenizerPooledAplusB_21
from lib.linear_lr import LinearLR
from config import MODEL_LOAD_FROM_FILE, MODEL_PATH, DEFAULT_PARAMS_BERT, DEFAULT_PARAMS_BERT_WITH_POOLING

def seed_everything(seed_value=1234):
    """Set all random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class KFoldN(KFold):
    """Custom K-Fold that splits training data into train/validation/test."""
    def split(self, X, groups=None):
        s = super().split(X, groups)
        for train_indxs, test_indxs in s:
            train_indxs, cv_indxs = train_test_split(
                train_indxs, test_size=(1 / (self.n_splits - 1)), random_state=0)
            yield train_indxs, cv_indxs, test_indxs

class BERTClassificationModel(Model):
    """BERT model for basic classification without pooling."""
    def __init__(self, params=DEFAULT_PARAMS_BERT):
        super().__init__()
        self.params = params
        tokenizer, bert = load_pretrained_model()
        self.preprocessor = BERTTokenizer(tokenizer)
        self.dataset_class = TokenizedDataset
        self.nn = initialize_model(bert, self.params['device'])
        self.optimizer = AdamW(self.nn.parameters(), lr=self.params['learning_rate'])

    def evaluate_single_batch(self, batch, model, device):
        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        model_input = batch[:-1]

        labels = batch[-1]

        # model predictions
        preds = model(*model_input)
        preds = torch.flatten(preds).cpu()
        labels = labels.float().cpu()
        return preds, labels

class BERTClassificationModelWithPooling(Model):
    """BERT model with pooling for long-text classification."""
    def __init__(self, params=DEFAULT_PARAMS_BERT_WITH_POOLING):
        super().__init__()
        self.params = params
        tokenizer, bert = load_pretrained_model()
        self.preprocessor = BERTTokenizerPooled(
            tokenizer, params['size'], params['step'], params['minimal_length'])
        self.dataset_class = TokenizedDataset
        self.collate_fn = collate_fn_pooled_tokens
        self.nn = initialize_model(bert, self.params['device'])
        self.optimizer = AdamW(self.nn.parameters(), lr=self.params['learning_rate'])

    def evaluate_single_batch(self, batch, model, device):
        input_ids = batch[0]
        attention_mask = batch[1]
        number_of_chunks = [len(x) for x in input_ids]
        labels = batch[2]

        # concatenate all input_ids into one batch

        input_ids_combined = []
        for x in input_ids:
            input_ids_combined.extend(x.tolist())

        input_ids_combined_tensors = torch.stack(
            [torch.tensor(x).to(device) for x in input_ids_combined])

        # concatenate all attention maska into one batch

        attention_mask_combined = []
        for x in attention_mask:
            attention_mask_combined.extend(x.tolist())

        attention_mask_combined_tensors = torch.stack(
            [torch.tensor(x).to(device) for x in attention_mask_combined])

        # get model predictions for the combined batch
        preds = model(
            input_ids_combined_tensors,
            attention_mask_combined_tensors)

        preds = preds.flatten().cpu()

        # split result preds into chunks

        preds_split = preds.split(number_of_chunks)

        # pooling
        if self.params['pooling_strategy'] == 'mean':
            pooled_preds = torch.cat(
                [torch.mean(x).reshape(1) for x in preds_split])
        elif self.params['pooling_strategy'] == 'max':
            pooled_preds = torch.cat([torch.max(x).reshape(1)
                                     for x in preds_split])
        labels = [float(x) for x in labels]
        labels_detached = torch.tensor(labels).float()

        return pooled_preds, labels_detached

class BERTClassificationModeWithPoolingAB(Model):
    def __init__(self, config, params=DEFAULT_PARAMS_BERT_WITH_POOLING):
        super().__init__()
        self.params = params
        tokenizer, bert = load_pretrained_model()
        self.preprocessor = BERTTokenizerPooled(
                    tokenizer, params['size'], params['step'], params['minimal_length'])
        self.dataset_class = TokenizedDatasetAB
        self.collate_fn = collate_fn_pooled_tokens_AB
        self.nn = initialize_model_AB(bert, config, self.params['device'])
        self.optimizer = AdamW(self.nn.parameters(),
                               lr=self.params['learning_rate'])
        self.scheduler = LinearLR(self.optimizer)
        self.max_slice = 1
        self.grad_scaler = torch.cuda.amp.GradScaler()

    def load_state_dict(self, path):
        return self.nn.load_state_dict(path)

    def evaluate_single_batch(self, batch, model, device):
        input_ids_para = batch[0]
        attention_mask_para = batch[1]
        input_ids_comm = batch[2]
        attention_mask_comm = batch[3]
        number_of_chunks_para = [len(x) for x in input_ids_para]
        number_of_chunks_comm = [len(x) for x in input_ids_comm]

        # Chunk paragraph: keep only first chunk
        input_ids_para_chunked = [x[:1] if len(x) > 1 else x for x in input_ids_para]
        number_of_chunks_para_chunked = [len(x) for x in input_ids_para_chunked]
        attention_mask_para_chunked = [x[:1] if len(x) > 1 else x for x in attention_mask_para]

        # Chunk commentary: limit to max_slice
        input_ids_comm_chunked = [x[:self.max_slice] if len(x) > self.max_slice else x for x in input_ids_comm]
        number_of_chunks_comm_chunked = [len(x) for x in input_ids_comm_chunked]
        attention_mask_comm_chunked = [x[:self.max_slice] if len(x) > self.max_slice else x for x in attention_mask_comm]
        # Flatten and combine para
        input_ids_combined_para = [item for sublist in input_ids_para_chunked for item in sublist.tolist()]
        input_ids_combined_tensors_para = torch.stack(
            [torch.tensor(x).to(device) for x in input_ids_combined_para])
        attention_mask_combined_para = [item for sublist in attention_mask_para_chunked for item in sublist.tolist()]
        attention_mask_combined_tensors_para = torch.stack(
            [torch.tensor(x).to(device) for x in attention_mask_combined_para])

        # Flatten and combine comm
        input_ids_combined_comm = [item for sublist in input_ids_comm_chunked for item in sublist.tolist()]
        input_ids_combined_tensors_comm = torch.stack(
            [torch.tensor(x).to(device) for x in input_ids_combined_comm])
        attention_mask_combined_comm = [item for sublist in attention_mask_comm_chunked for item in sublist.tolist()]
        attention_mask_combined_tensors_comm = torch.stack(
            [torch.tensor(x).to(device) for x in attention_mask_combined_comm])
        labels = batch[4]
        preds = model(
            input_ids_combined_tensors_para,
            attention_mask_combined_tensors_para,
            input_ids_combined_tensors_comm,
            attention_mask_combined_tensors_comm,
            number_of_chunks_para_chunked,
            number_of_chunks_comm_chunked)

        labels_detached = torch.tensor([float(x) for x in labels]).float()
        preds = preds.cpu()
        return preds, labels_detached

class BERTClassificationModelWithPoolingAplusB_21(Model):
    """BERT model for A+B classification with pooling (21 variant)."""
    def __init__(self, config, params=DEFAULT_PARAMS_BERT_WITH_POOLING):
        super().__init__()
        self.params = params
        tokenizer, bert = load_pretrained_model()
        self.preprocessor = BERTTokenizerPooledAplusB_21(tokenizer)
        self.dataset_class = TokenizedDatasetAplusB
        self.collate_fn = collate_fn_pooled_tokens_AplusB
        self.nn = initialize_model(bert, config, self.params['device'])
        self.optimizer = AdamW(self.nn.parameters(), lr=self.params['learning_rate'])
        self.scheduler = LinearLR(self.optimizer)
        self.grad_scaler = torch.cuda.amp.GradScaler()

    def load_state_dict(self, path):
        return self.nn.load_state_dict(path)

    def evaluate_single_batch(self, batch, model, device):
        input_ids = batch[0]
        attention_mask = batch[1]
        number_of_chunks = [len(x) for x in input_ids]
        labels = batch[2]
        # concatenate all input_ids into one batch

        input_ids_combined = []
        for x in input_ids:
            input_ids_combined.extend(x.tolist())

        input_ids_combined_tensors = torch.stack(
            [torch.tensor(x).to(device) for x in input_ids_combined])

        # concatenate all attention maska into one batch

        attention_mask_combined = []
        for x in attention_mask:
            attention_mask_combined.extend(x.tolist())

        attention_mask_combined_tensors = torch.stack(
            [torch.tensor(x).to(device) for x in attention_mask_combined])

        # get model predictions for the combined batch
        preds = model(
            input_ids_combined_tensors,
            attention_mask_combined_tensors,
            number_of_chunks)

        preds = preds.flatten().cpu()
        # split result preds into chunks, although it's unnecessary here.
        preds_split = preds.split(number_of_chunks)   
        # pooling
        if self.params['pooling_strategy'] == 'mean':
            pooled_preds = torch.cat(
                [torch.mean(x).reshape(1) for x in preds_split])
        elif self.params['pooling_strategy'] == 'max':
            pooled_preds = torch.cat([torch.max(x).reshape(1)
                                     for x in preds_split])
        labels = [float(x) for x in labels]
        labels_detached = torch.tensor(labels).float()
       
        return pooled_preds, labels_detached
    
class BERTClassificationModelWithPoolingAplusB(Model):
    def __init__(self, config, params=DEFAULT_PARAMS_BERT_WITH_POOLING):
        super().__init__()
        self.params = params
        tokenizer, bert = load_pretrained_model()
        self.preprocessor = BERTTokenizerPooledAplusB(
            tokenizer, params['size'], params['step'], params['minimal_length'])
        self.dataset_class = TokenizedDatasetAplusB
        self.collate_fn = collate_fn_pooled_tokens
        self.nn = initialize_model(bert, config, self.params['device'])
        self.optimizer = AdamW(self.nn.parameters(),
                               lr=self.params['learning_rate'])
        self.scheduler = LinearLR(self.optimizer)
        self.grad_scaler = torch.cuda.amp.GradScaler()

    def load_state_dict(self, path):
        return self.nn.load_state_dict(path)

    def evaluate_single_batch(self, batch, model, device):
        input_ids = batch[0]
        attention_mask = batch[1]
        number_of_chunks = [len(x) for x in input_ids]
        labels = batch[2]
        # concatenate all input_ids into one batch
        #################################################       
        input_ids_chuncked = []
        for x in input_ids:
            if len(x) > 2:
                x = x[:2]
            input_ids_chuncked.append(x)
        number_of_chunks_chuncked = [len(x) for x in input_ids_chuncked]

        attention_mask_chuncked = []
        for x in attention_mask:
            if len(x) > 1:
                x = x[:2]
            attention_mask_chuncked.append(x)
        #################################################
        input_ids_combined = []
        for x in input_ids_chuncked:
            input_ids_combined.extend(x.tolist())

        input_ids_combined_tensors = torch.stack(
            [torch.tensor(x).to(device) for x in input_ids_combined])

        # concatenate all attention maska into one batch

        attention_mask_combined = []
        for x in attention_mask_chuncked:
            attention_mask_combined.extend(x.tolist())

        attention_mask_combined_tensors = torch.stack(
            [torch.tensor(x).to(device) for x in attention_mask_combined])

        # get model predictions for the combined batch
        preds = model(
            input_ids_combined_tensors,
            attention_mask_combined_tensors,
            number_of_chunks_chuncked)

        preds = preds.flatten().cpu()
        # split result preds into chunks
        preds_split = preds.split(number_of_chunks_chuncked)
  
        # pooling
        if self.params['pooling_strategy'] == 'mean':
            pooled_preds = torch.cat(
                [torch.mean(x).reshape(1) for x in preds_split])
        elif self.params['pooling_strategy'] == 'max':
            pooled_preds = torch.cat([torch.max(x).reshape(1)
                                     for x in preds_split])
        labels = [float(x) for x in labels]
        labels_detached = torch.tensor(labels).float()

        return pooled_preds, labels_detached

def load_pretrained_model():
    """Load tokenizer and BERT model."""
    tokenizer = load_tokenizer()
    model = load_bert()
    return tokenizer, model


def load_tokenizer():
    """Load BERT tokenizer from file or pretrained."""
    if MODEL_LOAD_FROM_FILE:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(MODEL_PATH, "tokenizer.json"))
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer


def load_bert(freeze=False):
    """Load BERT model from file or pretrained."""
    if MODEL_LOAD_FROM_FILE:
        model = AutoModel.from_pretrained(MODEL_PATH)
    else:
        model = BertModel.from_pretrained("bert-base-uncased")

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model


def load_config():
    """Load BERT config from file."""
    config_file = os.path.join(MODEL_PATH, "config.json")
    config = AutoConfig.from_pretrained(config_file)
    return config


def initialize_model(bert, config, device):
    """Initialize BERT classification model."""
    model = BERTSequenceClassificationArch(bert, config, params=DEFAULT_PARAMS_BERT_WITH_POOLING)
    model = model.to(device)
    return model


def initialize_model_AB(bert, config, device):
    """Initialize BERT AB classification model with optional distributed training."""
    model = BERTSequenceClassificationArchForAB(bert, config, params=DEFAULT_PARAMS_BERT_WITH_POOLING)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.parallel.DistributedDataParallel(model)

    model.to(device)
    return model

def parse_args():
    """Parse command line arguments for inference."""
    parser = ArgumentParser(description="Inference for trained BERT models")
    
    # Data and I/O
    parser.add_argument("--data_path", type=str, default="../dataset/", help="Path to input data files")
    parser.add_argument("--input_file", type=str, help="Path to input TSV file for inference")
    parser.add_argument("--model_path", type=str, default="./models/best-model-parameters.pt", help="Path to trained model checkpoint")
    parser.add_argument("--eval_path", type=str, default="./AB", help="Path to save evaluation outputs")
    
    # Model configuration
    parser.add_argument("--model_type", type=str, default="AB", choices=["AB", "AplusB", "AplusB21"], help="Model type to use")
    parser.add_argument("--model_load_from_file", action='store_true', help="Load model from file")
    
    # Training parameters (for compatibility)
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    # Text processing
    parser.add_argument("--max_tokens", type=int, default=512*2, help="Maximum number of tokens")
    parser.add_argument("--max_slice", type=int, default=2, help="Maximum number of slices")
    parser.add_argument("--pooling", type=int, default=0, help="Pooling strategy: 0=mean, 1=max")
    
    # Experiment settings
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility")
    parser.add_argument("--kfold", type=int, default=1, help="Number of folds for cross-validation")
    parser.add_argument("--enable_wandb", action='store_true', help="Enable W&B logging")
    parser.add_argument("--verbose", type=bool, default=True, help="Log intermediate results")
    
    # Hardware
    parser.add_argument("--parallel", type=bool, default=False, help="Run in parallel mode")
    parser.add_argument("--visible_gpus", type=str, default="1,2,3,4", help="Visible GPU IDs")
    parser.add_argument("--resume", action='store_true', help="Resume from checkpoint")

    return parser.parse_args()

def start_timer():
    """Start performance timer and clear GPU memory."""
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()


def end_timer_and_print():
    """Stop timer and print execution stats."""
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Total execution time = {end_time - start_time:.3f} sec")
    print(f"Max memory used by tensors = {torch.cuda.max_memory_allocated()} bytes")

def main():
    """Main inference function."""
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    args = parse_args()
    seed_everything(args.seed)
    logger.info(f"Setting random seed to {args.seed}")

    # Load model configuration and initialize
    bert_config = load_config()
    if args.model_type == 'AB':
        model = BERTClassificationModeWithPoolingAB(bert_config)
    elif args.model_type == 'AplusB21':
        model = BERTClassificationModelWithPoolingAplusB_21(bert_config)
    elif args.model_type == 'AplusB':
        model = BERTClassificationModelWithPoolingAplusB(bert_config)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load test data
    if args.input_file:
        input_path = args.input_file
    else:
        # Default paths for backward compatibility
        path = "path/to/default/dataset"
        file = "rfc8335" # rfc7657
        input_path = f"{path}/{file}.tsv"
        logger.warning(f"No --input_file specified, using default: {input_path}")

    test = pd.read_csv(input_path, sep="\t")
    paras_te = test['paragraph'].astype(str).values.tolist()
    comms_te = test['commentary'].astype(str).values.tolist()
    labels_te = test['label'].values.tolist()

    # Load checkpoint and run inference
    logger.info(f"Loading checkpoint from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info("Starting inference")
    if args.model_type == 'AB':
        res, _, scores = model.predictAB(paras_te, comms_te, labels_te, args=args)
    elif args.model_type in ['AplusB', 'AplusB21']:
        res, _, scores = model.predict(paras_te, comms_te, labels_te, args=args)

    # Save results
    test["preds"] = scores
    output_path = input_path.replace('.tsv', '.predictions.tsv')
    test.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Metrics: {res}")

if __name__ == "__main__":
    sys.exit(main())