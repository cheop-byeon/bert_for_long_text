import os
import sys
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast, AutoModel, AdamW
from transformers import BertTokenizer, BertModel, AutoModel, AutoConfig
from lib.architecture import BERTSequenceClassificationArch, BERTSequenceClassificationArchForAB, SBERTSequenceClassificationArchForAB
from lib.base_model_ga import Model
from lib.custom_datasets import TokenizedDataset, collate_fn_pooled_tokens, TokenizedDatasetAB, collate_fn_pooled_tokens_AB
from lib.custom_datasets_ab import TokenizedDatasetAplusB, collate_fn_pooled_tokens_AplusB
from lib.text_preprocessors import BERTTokenizer, BERTTokenizerPooled, BERTTokenizerPooledAplusB, BERTTokenizerPooledAplusB_21
from lib.linear_lr import LinearLR
from config import MODEL_LOAD_FROM_FILE, MODEL_PATH, DEFAULT_PARAMS_BERT, DEFAULT_PARAMS_BERT_WITH_POOLING, DEFAULT_PARAMS_SBERT

from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import numpy as np
import glob
from collections import defaultdict
from argparse import ArgumentParser
import random
import logging
import wandb
import time
import gc

def seed_everything(seed_value=1234):
    """Set all random seeds for reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class KFoldN(KFold):
    """Custom K-Fold that splits training data into train/validation/test"""
    def split(self, X, groups=None):
        s = super().split(X, groups)
        for train_indxs, test_indxs in s:
            train_indxs, cv_indxs = train_test_split(
                train_indxs, test_size=(1 / (self.n_splits - 1)), random_state=0)
            yield train_indxs, cv_indxs, test_indxs

class BERTClassificationModeWithPoolingAB(Model):
    """BERT model for AB classification with pooling"""
    def __init__(self, config, params=DEFAULT_PARAMS_BERT_WITH_POOLING):
        super().__init__()
        self.params = params
        tokenizer, bert = load_pretrained_model()
        self.preprocessor = BERTTokenizerPooled(
            tokenizer, params['size'], params['step'], params['minimal_length'])
        self.dataset_class = TokenizedDatasetAB
        self.collate_fn = collate_fn_pooled_tokens_AB
        self.nn = initialize_model_AB(bert, config, self.params['device'])
        self.optimizer = AdamW(self.nn.parameters(), lr=self.params['learning_rate'])
        self.scheduler = LinearLR(self.optimizer)
        self.max_slice = 1
        self.grad_scaler = torch.cuda.amp.GradScaler()

    def load_state_dict(self, path):
        return self.nn.load_state_dict(path)

    def evaluate_single_batch(self, batch, model, device):
        """Evaluate a single batch with chunking and pooling"""
        input_ids_para, attention_mask_para, input_ids_comm, attention_mask_comm = batch[:4]
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

        labels = torch.tensor([float(x) for x in labels]).float()
        preds = preds.cpu()
        return preds, labels


class SBERTClassificationModeAB(Model):
    def __init__(self, config, params=DEFAULT_PARAMS_SBERT):
        super().__init__()
        self.params = params
        tokenizer, bert = load_pretrained_model()
        self.preprocessor = BERTTokenizerPooled(
                    tokenizer, params['size'], params['step'], params['minimal_length'])
        self.dataset_class = TokenizedDatasetAB
        self.collate_fn = collate_fn_pooled_tokens_AB
        self.nn = initialize_model_SBERT(bert, config, self.params['device'])
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
        #################################################       
        input_ids_para_chuncked = []
        for x in input_ids_para:
            if len(x) > 1:
                x = x[:1]
            input_ids_para_chuncked.append(x)
        number_of_chunks_para_chuncked = [len(x) for x in input_ids_para_chuncked]

        attention_mask_para_chuncked = []
        for x in attention_mask_para:
            if len(x) > 1:
                x = x[:1]
            attention_mask_para_chuncked.append(x)
        #################################################
        input_ids_comm_chuncked = []
        for x in input_ids_comm:
            if len(x) > self.max_slice:
                x = x[:self.max_slice]
            input_ids_comm_chuncked.append(x)
        number_of_chunks_comm_chuncked = [len(x) for x in input_ids_comm_chuncked]

        attention_mask_comm_chuncked = []
        for x in attention_mask_comm:
            if len(x) > self.max_slice:
                x = x[:self.max_slice]
            attention_mask_comm_chuncked.append(x)
        #################################################
    
        #################################################
        # concatenate all input_ids_para into one batch
        
        input_ids_combined_para = []
        for x in input_ids_para_chuncked:
            input_ids_combined_para.extend(x.tolist())

        input_ids_combined_tensors_para = torch.stack(
            [torch.tensor(x).to(device) for x in input_ids_combined_para])
   
        # concatenate all attention maska into one batch

        attention_mask_combined_para = []
        for x in attention_mask_para_chuncked:
            attention_mask_combined_para.extend(x.tolist())

        attention_mask_combined_tensors_para = torch.stack(
            [torch.tensor(x).to(device) for x in attention_mask_combined_para])
        #################################################
        # concatenate all input_ids_comm into one batch

        input_ids_combined_comm = []
        for x in input_ids_comm_chuncked:
            input_ids_combined_comm.extend(x.tolist())
      
        input_ids_combined_tensors_comm = torch.stack(
            [torch.tensor(x).to(device) for x in input_ids_combined_comm])

        # concatenate all attention maska into one batch

        attention_mask_combined_comm = []
        for x in attention_mask_comm_chuncked:
            attention_mask_combined_comm.extend(x.tolist())

        attention_mask_combined_tensors_comm = torch.stack(
            [torch.tensor(x).to(device) for x in attention_mask_combined_comm])
        #################################################
        labels = batch[4]
        # get model predictions for the combined batch
        preds = model(
            input_ids_combined_tensors_para,
            attention_mask_combined_tensors_para,
            input_ids_combined_tensors_comm,
            attention_mask_combined_tensors_comm,
            number_of_chunks_para_chuncked,
            number_of_chunks_comm_chuncked)

        labels = [float(x) for x in labels]
        labels_detached = torch.tensor(labels).float()
        # labels = labels_detached.to(device)
        preds = preds.cpu()
        
        return preds, labels_detached

class BERTClassificationModelWithPoolingAplusB_21(Model):
    def __init__(self, config, params=DEFAULT_PARAMS_BERT_WITH_POOLING):
        super().__init__()
        self.params = params
        tokenizer, bert = load_pretrained_model()
        self.preprocessor = BERTTokenizerPooledAplusB_21(tokenizer)
        self.dataset_class = TokenizedDatasetAplusB
        self.collate_fn = collate_fn_pooled_tokens_AplusB
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
    """BERT model for A+B classification with pooling"""
    def __init__(self, config, params=DEFAULT_PARAMS_BERT_WITH_POOLING):
        super().__init__()
        self.params = params
        tokenizer, bert = load_pretrained_model()
        self.preprocessor = BERTTokenizerPooledAplusB(
            tokenizer, params['size'], params['step'], params['minimal_length'])
        self.dataset_class = TokenizedDatasetAplusB
        self.collate_fn = collate_fn_pooled_tokens
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

        # Chunk: limit to 2 chunks
        input_ids_chunked = [x[:2] if len(x) > 2 else x for x in input_ids]
        number_of_chunks_chunked = [len(x) for x in input_ids_chunked]
        attention_mask_chunked = [x[:2] if len(x) > 2 else x for x in attention_mask]

        # Flatten and combine
        input_ids_combined = [item for sublist in input_ids_chunked for item in sublist.tolist()]
        input_ids_combined_tensors = torch.stack(
            [torch.tensor(x).to(device) for x in input_ids_combined])
        attention_mask_combined = [item for sublist in attention_mask_chunked for item in sublist.tolist()]
        attention_mask_combined_tensors = torch.stack(
            [torch.tensor(x).to(device) for x in attention_mask_combined])

        # Get predictions
        preds = model(input_ids_combined_tensors, attention_mask_combined_tensors, number_of_chunks_chunked)
        preds = preds.flatten().cpu()

        # Pooling
        preds_split = preds.split(number_of_chunks_chunked)
        if self.params['pooling_strategy'] == 'mean':
            pooled_preds = torch.cat([torch.mean(x).reshape(1) for x in preds_split])
        else:
            pooled_preds = torch.cat([torch.max(x).reshape(1) for x in preds_split])

        labels_detached = torch.tensor([float(x) for x in labels]).float()
        return pooled_preds, labels_detached

def load_pretrained_model():
    """Load tokenizer and BERT model"""
    tokenizer = load_tokenizer()
    model = load_bert()
    return tokenizer, model


def load_tokenizer():
    """Load BERT tokenizer from file or pretrained"""
    if MODEL_LOAD_FROM_FILE:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(MODEL_PATH, "tokenizer.json"))
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer


def load_bert(freeze=False):
    """Load BERT model from file or pretrained"""
    if MODEL_LOAD_FROM_FILE:
        model = AutoModel.from_pretrained(MODEL_PATH)
    else:
        model = BertModel.from_pretrained("bert-base-uncased")

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model


def load_config():
    """Load BERT config from file"""
    config_file = os.path.join(MODEL_PATH, "config.json")
    config = AutoConfig.from_pretrained(config_file)
    return config

def initialize_model(bert, config, device):
    """Initialize BERT classification model"""
    model = BERTSequenceClassificationArch(bert, config, params=DEFAULT_PARAMS_BERT_WITH_POOLING)
    model = model.to(device)
    return model


def initialize_model_AB(bert, config, device):
    """Initialize BERT AB classification model with optional distributed training"""
    model = BERTSequenceClassificationArchForAB(bert, config, params=DEFAULT_PARAMS_BERT_WITH_POOLING)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.parallel.DistributedDataParallel(model)

    model.to(device)
    return model


def initialize_model_SBERT(bert, config, device):
    """Initialize SBERT classification model with optional distributed training"""
    model = SBERTSequenceClassificationArchForAB(bert, config, params=DEFAULT_PARAMS_SBERT)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.parallel.DistributedDataParallel(model)

    model.to(device)
    return model


def parse_args():
    """Parse command line arguments"""
    parser = ArgumentParser(description="Training and evaluation of a single model")
    parser.add_argument("--enable_wandb", action='store_true', help="Enable Weights & Biases logging")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for training")
    parser.add_argument("--data_path", type=str, default="../dataset/", help="Location for data files")
    parser.add_argument("--model_type", type=str, default="AB", help="Model type to use")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility")
    parser.add_argument("--model_path", type=str, default="./models/best-model-parameters.pt", help="Path to store trained models")
    parser.add_argument("--parallel", type=bool, default=False, help="Run model in parallel mode")
    parser.add_argument("--kfold", type=int, default=1, help="K for k-fold cross validation")
    parser.add_argument("--verbose", type=bool, default=True, help="Log intermediate results")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--max_tokens", type=int, default=512 * 2, help="Maximum tokens")
    parser.add_argument("--max_slice", type=int, default=2, help="Maximum slices")
    parser.add_argument("--model_load_from_file", action='store_true', help="Load model from file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--visible_gpus", type=str, default="1,2,3,4", help="Visible GPUs")
    parser.add_argument("--pooling", type=int, default=0, help="Pooling strategy (0=mean, 1=max)")
    parser.add_argument("--resume", action='store_true', help="Resume training from checkpoint")
    parser.add_argument("--eval_path", type=str, default="./AB", help="Path for evaluation outputs")

    args = parser.parse_args()
    return args

def start_timer():
    """Reset CUDA cache and start timing"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()


def end_timer_and_print():
    """Print execution time and max memory used"""
    torch.cuda.synchronize()
    print("Total execution time = {:.3f} sec".format(time.time() - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))


def main():
    """Main training and evaluation pipeline"""
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    args = parse_args()
    seed_everything(args.seed)
    logger.info(f"Setting random seed to {args.seed}")

    # Initialize wandb
    wandb_config = {
        "epochs": args.epochs,
        "learning rate": DEFAULT_PARAMS_BERT_WITH_POOLING["learning_rate"],
        "optimizer": "AdamW",
        "scheduler": "Linear with warmup",
        "max_slice": args.max_slice
    }

    bert_config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Determine wandb mode
    if args.enable_wandb is False:
        wandb_mode = "disabled"
    elif device == "cpu":
        wandb_mode = "online"
    else:
        wandb_mode = "offline"

    wandb.init(project="ids-bi-cls", entity="account", config=wandb_config, name="bi-cls", mode=wandb_mode)

    # Initialize model based on type
    if args.model_type == 'AB':
        model = BERTClassificationModeWithPoolingAB(bert_config)
    elif args.model_type == 'AplusB21':
        model = BERTClassificationModelWithPoolingAplusB_21(bert_config)
    elif args.model_type == 'AplusB':
        model = BERTClassificationModelWithPoolingAplusB(bert_config)
    elif args.model_type == 'SBERT':
        model = SBERTClassificationModeAB(bert_config)
    else:
        model = BERTClassificationModeWithPoolingAB(bert_config)

    # K-fold training and evaluation
    for fold in range(args.kfold):
        fold_idx = fold + 1
        logger.info(f"Fold: {fold_idx}/{args.kfold}")

        path = args.data_path
        train_file = f"{path}train.{fold_idx}.tsv"
        valid_file = f"{path}valid.{fold_idx}.tsv"

        logger.info(f"Reading train/val data from {train_file}, {valid_file}")
        train = pd.read_csv(train_file, sep='\t')
        valid = pd.read_csv(valid_file, sep='\t')

        # Limit data for CPU/MPS testing
        train_limit = 4 if device in ["cpu", "mps"] else None
        valid_limit = 10 if device in ["cpu", "mps"] else None

        paras_tr = train['paragraph'].values.tolist()[:train_limit]
        comms_tr = train['commentary'].values.tolist()[:train_limit]
        labels_tr = train['label'].values.tolist()[:train_limit]

        paras_dev = valid['paragraph'].values.tolist()[:valid_limit]
        comms_dev = valid['commentary'].values.tolist()[:valid_limit]
        labels_dev = valid['label'].values.tolist()[:valid_limit]
        drafts_dev = valid['draft'].values.tolist()[:valid_limit]
        wgs_dev = valid['wg'].values.tolist()[:valid_limit]

        logger.info("Starting training and validation")
        if args.model_type == 'AB' or args.model_type == 'SBERT':
            model.train_and_evaluateAB(paras_tr, comms_tr, paras_dev, comms_dev,
                                       labels_tr, labels_dev, drafts_dev, wgs_dev, args)
        elif args.model_type in ['AplusB21', 'AplusB']:
            model.train_and_evaluateAplusB(paras_tr, comms_tr, paras_dev, comms_dev,
                                           labels_tr, labels_dev, drafts_dev, wgs_dev, args)
        else:
            logger.error(f"Unknown model type: {args.model_type}")

    wandb.finish()

if __name__ == "__main__":
    sys.exit(main())
