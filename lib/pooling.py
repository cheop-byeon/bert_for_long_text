import torch

# Functions for preparing input for longer texts - based on
# https://www.kdnuggets.com/2021/04/apply-transformers-any-length-text.html


def tokenize_all_text(text, tokenizer):
    '''
    Tokenizes the entire text without truncation and without special tokens

    Parameters:
    text - single str with arbitrary length
    tokenizer - object of class transformers.PreTrainedTokenizerFast

    Returns:
    tokens - dictionary of the form
    {
    'input_ids' : [...]
    'token_type_ids' : [...]
    'attention_mask' : [...]
    }
    '''
    tokens = tokenizer.encode_plus(text, add_special_tokens=False,
                                   return_tensors='pt')
    return tokens

def split_overlapping(array, size, step, minimal_length):
    ''' Helper function for dividing arrays into overlapping chunks '''
    result = [array[i:i + size] for i in range(0, len(array), step)]
    if len(result) > 1:
        # ignore chunks with less then minimal_length number of tokens
        result = [x for x in result if len(x) >= minimal_length]
    return result


def split_tokens_into_smaller_chunks(tokens, size, step, minimal_length):
    ''' Splits tokens into overlapping chunks with given size and step'''
    assert size <= 510
    input_id_chunks = split_overlapping(tokens['input_ids'][0], size, step, minimal_length)
    mask_chunks = split_overlapping(tokens['attention_mask'][0], size, step, minimal_length)
    return input_id_chunks, mask_chunks


def add_special_tokens_at_beginning_and_end(input_id_chunks, mask_chunks):
    ''' Adds special CLS token (token id = 101) at the beginning and SEP token (token id = 102) at the end of each chunk'''
    for i in range(len(input_id_chunks)):
        # adding CLS (token id 101) and SEP (token id 102) tokens
        input_id_chunks[i] = torch.cat(
            [torch.Tensor([101]), input_id_chunks[i], torch.Tensor([102])])
        mask_chunks[i] = torch.cat(
            [torch.Tensor([1]), mask_chunks[i], torch.Tensor([1])])

def add_special_tokens_at_beginning_and_middle_and_end(para_input_id, comm_input_id_chunks, para_mask, comm_mask_chunks):
    ''' Adds special CLS token (token id = 101) at the beginning and SEP token (token id = 102) at the end of each chunk'''
    for i in range(len(comm_input_id_chunks)):
        # adding CLS (token id 101) and SEP (token id 102) tokens
        comm_input_id_chunks[i] = torch.cat(
            [torch.Tensor([101]), para_input_id, torch.Tensor([102]), comm_input_id_chunks[i], torch.Tensor([102])])
        comm_mask_chunks[i] = torch.cat(
            [torch.Tensor([1]), para_mask, torch.Tensor([1]), comm_mask_chunks[i], torch.Tensor([1])])

def add_padding_tokens(input_id_chunks, mask_chunks):
    ''' Adds padding tokens (token id = 0) at the end to make sure that all chunks have exactly 512 tokens '''
    for i in range(len(input_id_chunks)):
        # get required padding length
        pad_len = 512 - input_id_chunks[i].shape[0]
        # check if tensor length satisfies required chunk size
        if pad_len > 0:
            # if padding length is more than 0, we must add padding
            input_id_chunks[i] = torch.cat([
                input_id_chunks[i], torch.Tensor([0] * pad_len)
            ])
            mask_chunks[i] = torch.cat([
                mask_chunks[i], torch.Tensor([0] * pad_len)
            ])


def stack_tokens_from_all_chunks(input_id_chunks, mask_chunks):
    ''' Reshapes data to a form compatible with BERT model input'''
    input_ids = torch.stack(input_id_chunks)
    attention_mask = torch.stack(mask_chunks)

    return input_ids.long(), attention_mask.int()


def transform_text_to_model_input(
        text,
        tokenizer,
        size=510,
        step=510,
        minimal_length=10):
    ''' Transforms the entire text to model input of BERT model'''
    tokens = tokenize_all_text(text, tokenizer)
    input_id_chunks, mask_chunks = split_tokens_into_smaller_chunks(
        tokens, size, step, minimal_length)
    add_special_tokens_at_beginning_and_end(input_id_chunks, mask_chunks)
    add_padding_tokens(input_id_chunks, mask_chunks)
    input_ids, attention_mask = stack_tokens_from_all_chunks(
        input_id_chunks, mask_chunks)
    return [input_ids, attention_mask]

def transform_para_comm_to_model_input(
    para,
    comm,
    tokenizer,
    size = 510,
    step = 510,
    minimal_length = 100):

    para_encode_all = tokenize_all_text(para, tokenizer)
    para_len_ = len(para_encode_all.input_ids[0])

    # if para_len >= 510: # we only consider para
    #     size, step = 510, 510
    #     return transform_text_to_model_input(para, tokenizer, size, step, minimal_length)

    max_para_len = 256 if para_len_ > 256 else para_len_
    max_comm_len = 512 - max_para_len - 1 - 1 - 1 # [cls] [sep] [sep]

    para_encode = tokenizer.encode_plus(para, max_length = max_para_len, truncation=True, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False)
    para_input_id = para_encode.input_ids[0]
    para_mask = para_encode.attention_mask[0]

    comm_encode = tokenize_all_text(comm, tokenizer)
    comm_input_id_chunks, comm_mask_chunks = split_tokens_into_smaller_chunks(comm_encode, size = max_comm_len, step = size, minimal_length=minimal_length)
    add_special_tokens_at_beginning_and_middle_and_end(para_input_id, comm_input_id_chunks, para_mask, comm_mask_chunks)
    add_padding_tokens(comm_input_id_chunks, comm_mask_chunks)
    input_ids, attention_mask = stack_tokens_from_all_chunks(comm_input_id_chunks, comm_mask_chunks)
    return [input_ids, attention_mask]

def transform_para_comm_to_model_input_into_one(
    para,
    comm,
    tokenizer,
    para_len = 256,
    comm_len = 256):
    
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    para_encode_all = tokenize_all_text(para, tokenizer)
    para_len_ = len(para_encode_all.input_ids[0])

    max_para_len = para_len if para_len_ > para_len else para_len_
    max_comm_len = 512 - max_para_len - 1 - 1 - 1 # [cls] [sep] [sep]

    para_encode = tokenizer.encode_plus(para, max_length = max_para_len, truncation=True, return_tensors="pt", add_special_tokens=False, padding='max_length', return_token_type_ids=False)
    para_input_id = para_encode.input_ids[0]
    para_mask = para_encode.attention_mask[0]

    comm_encode = tokenizer.encode_plus(comm, max_length = max_comm_len, truncation=True, return_tensors="pt", add_special_tokens=False, padding='max_length', return_token_type_ids=False)
    comm_input_id = comm_encode.input_ids[0]
    comm_mask = comm_encode.attention_mask[0]
   
    comm_input_id_chunks, comm_mask_chunks = split_tokens_into_smaller_chunks(comm_encode, size = max_comm_len, step = max_comm_len, minimal_length=100)
    add_special_tokens_at_beginning_and_middle_and_end(para_input_id, comm_input_id_chunks, para_mask, comm_mask_chunks)
    # add_padding_tokens(comm_input_id_chunks, comm_mask_chunks)
    input_ids, attention_mask = stack_tokens_from_all_chunks(comm_input_id_chunks, comm_mask_chunks)
    return [input_ids, attention_mask]