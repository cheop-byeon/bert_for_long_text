import numpy as np
import torch

from lib.pooling import transform_text_to_model_input, transform_para_comm_to_model_input, transform_para_comm_to_model_input_into_one


class Preprocessor:
    """Abstract class for text preprocessors. Preprocessor takes an array of strings and transforms it to an array of data compatible with the model"""

    def __init__(self):
        pass

    def preprocess(self, array_of_texts):
        raise NotImplementedError("Preprocessing is implemented for subclasses only")


class BERTTokenizer(Preprocessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def preprocess(self, array_of_texts):
        tokens = tokenize(array_of_texts, self.tokenizer)
        return tokens


class BERTTokenizerPooled(Preprocessor):
    def __init__(self, tokenizer, size, step, minimal_length):
        self.tokenizer = tokenizer
        self.text_splits_params = (size, step, minimal_length)

    def preprocess(self, array_of_texts):
        array_of_preprocessed_data = tokenize_pooled(
            array_of_texts, self.tokenizer, *self.text_splits_params)
        return array_of_preprocessed_data


class BERTTokenizerPooledAplusB(Preprocessor):
    def __init__(self, tokenizer, size, step, minimal_length):
        self.tokenizer = tokenizer
        self.text_splits_params = (size, step, minimal_length)

    def preprocess(self, array_of_paras, array_of_comms):
        array_of_preprocessed_data = tokenize_pooled_AplusB(
            array_of_paras, array_of_comms, self.tokenizer, *self.text_splits_params)
        return array_of_preprocessed_data


class BERTTokenizerPooledAplusB_21(Preprocessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def preprocess(self, array_of_paras, array_of_comms):
        array_of_preprocessed_data = tokenize_pooled_AplusB_into_one(
            array_of_paras, array_of_comms, self.tokenizer)
        return array_of_preprocessed_data


def tokenize(texts, tokenizer):
    """Transforms list of texts to list of tokens (truncated to 512 tokens)

    Parameters:
    texts - list of strings
    tokenizer - object of class transformers.PreTrainedTokenizerFast

    Returns:
    array_of_preprocessed_data - array of the length len(texts)
    """
    texts = list(texts)
    tokenizer.pad_token = "<pad>"
    tokens = tokenizer.batch_encode_plus(
        texts,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors='pt')
    return tokens


def tokenize_pooled(texts, tokenizer, size, step, minimal_length):
    """Tokenizes texts and splits to chunks of 512 tokens

    Parameters:
    texts - list of strings
    tokenizer - object of class transformers.PreTrainedTokenizerFast
    size - size of text chunk to tokenize (must be <= 510)
    step - stride of pooling
    minimal_length - minimal length of a text chunk

    Returns:
    array_of_preprocessed_data - array of the length len(texts)
    """
    model_inputs = [
        transform_text_to_model_input(
            text, tokenizer, size, step, minimal_length) for text in texts]
    input_ids = [model_input[0] for model_input in model_inputs]
    attention_mask = [model_input[1] for model_input in model_inputs]
    tokens = {'input_ids': input_ids, 'attention_mask': attention_mask}
    return tokens


def tokenize_pooled_AplusB(paras, comms, tokenizer, size, step, minimal_length):
    """Tokenizes paragraph-commentary pairs with pooling"""
    model_inputs = [
        transform_para_comm_to_model_input(
            para, comm, tokenizer, size, step, minimal_length) 
        for para, comm in zip(paras, comms)]
    input_ids = [model_input[0] for model_input in model_inputs]
    attention_mask = [model_input[1] for model_input in model_inputs]
    tokens = {'input_ids': input_ids, 'attention_mask': attention_mask}
    return tokens


def tokenize_pooled_AplusB_into_one(paras, comms, tokenizer, para_len=256, comm_len=256):
    """Tokenizes paragraph-commentary pairs combined into one sequence"""
    model_inputs = [
        transform_para_comm_to_model_input_into_one(
            para, comm, tokenizer, para_len, comm_len) 
        for para, comm in zip(paras, comms)]
    input_ids = [model_input[0] for model_input in model_inputs]
    attention_mask = [model_input[1] for model_input in model_inputs]
    tokens = {'input_ids': input_ids, 'attention_mask': attention_mask}
    return tokens