from torch import nn
import torch


class BERTSequenceClassificationHead(nn.Module):
    def __init__(self, input_dim=768, output_dim=1):
        super().__init__()
        self.out_proj = nn.Linear(input_dim, output_dim)

    def forward(self, cls_token_hidden_state):
        return self.out_proj(cls_token_hidden_state)

class DistilBertClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.relu = nn.ReLU()
    
    def forward(self, pooled_output):
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = self.relu(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)
        return logits

class BERTSequenceClassificationArch(nn.Module):
    def __init__(self, bert, config, params):
        super().__init__()
        self.bert = bert
        self.config = config
        self.params = params
        self.classification_head = BERTSequenceClassificationHead()
        
    def forward(self, input_ids, attention_mask, chunks):
        x = bert_vectorize(self.bert, input_ids, attention_mask)
        x = self.classification_head(x)
        return x
class BERTSequenceClassificationArchForAB(nn.Module):
    def __init__(self, bert, config, params):
        super().__init__()
        self.bert = bert
        self.config = config
        self.classification_head = BERTSequenceClassificationHead()
        self.params = params
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids_para, attention_mask_para, input_ids_comm, attention_mask_comm, chunk_para, chunk_comm):
        para = bert_vectorize(self.bert, input_ids_para, attention_mask_para)
        comm = bert_vectorize(self.bert, input_ids_comm, attention_mask_comm)
        paras = para.split(chunk_para)
        comms = comm.split(chunk_comm)

        # mean pooling over the CLS token when we allow two chunks of input
        if self.params['pooling_strategy'] == 'mean':
            paras = torch.cat([torch.mean(p, dim=0).unsqueeze(0) for p in paras])
            comms = torch.cat([torch.mean(c, dim=0).unsqueeze(0) for c in comms])
        else:  # max pooling over the CLS tokens
            paras = torch.cat([torch.max(p.unsqueeze(1), dim=0).values for p in paras])
            comms = torch.cat([torch.max(c.unsqueeze(1), dim=0).values for c in comms])

        assert paras.shape == comms.shape

        # Hadamard product
        out = torch.mul(paras, comms)

        if self.params['score'] == 'dot':
            score = self.classification_head(out)
        else:
            score = nes_torch(paras, comms)
            score = score.unsqueeze(1)

        return score
    
class SBERTSequenceClassificationArchForAB(nn.Module):
    def __init__(self, bert, config, params):
        super().__init__()
        self.bert = bert   
        self.config = config
        self.params = params

        num_vectors_concatenated = 0
        if self.params['concatenation_sent_rep']:
            num_vectors_concatenated += 2
        if self.params['concatenation_sent_difference']:
            num_vectors_concatenated += 1
        if self.params['concatenation_sent_multiplication']:
            num_vectors_concatenated += 1

        self.classifier = nn.Linear(num_vectors_concatenated * 768, 1)

    def forward(self, input_ids_para, attention_mask_para, input_ids_comm, attention_mask_comm, chunk_para, chunk_comm):
        # print(len(input_ids_para), len(input_ids_para[0]))
        para = bert_vectorize(self.bert, input_ids_para, attention_mask_para)
        comm = bert_vectorize(self.bert, input_ids_comm, attention_mask_comm)
        paras = para.split(chunk_para)
        comms = comm.split(chunk_comm)

        # mean pooling over the CLS token when we allow two chuncks of input
        if self.params['pooling_strategy'] == 'mean':
            paras = torch.cat([torch.mean(p, dim = 0).unsqueeze(0) for p in paras]) # torch.mean(p.unsqueeze(1), dim = 0)
            comms = torch.cat([torch.mean(c, dim = 0).unsqueeze(0) for c in comms])
        else: # max pooling over the CLS tokens
            paras = torch.cat([torch.max(p.unsqueeze(1), dim = 0).values for p in paras]) #tuple(map(torch.stack, zip(*x)))
            comms = torch.cat([torch.max(c.unsqueeze(1), dim = 0).values for c in comms])
        
        vectors_concat = []
        if self.params['concatenation_sent_rep']:
            vectors_concat.append(paras)
            vectors_concat.append(comms)

        if self.params['concatenation_sent_difference']:
            vectors_concat.append(torch.abs(paras - comms))

        if self.params['concatenation_sent_multiplication']:
            vectors_concat.append(paras * comms)

        features = torch.cat(vectors_concat, 1)
        output = self.classifier(features)

        return output

def bert_vectorize(bert, input_ids, attention_mask):
    """Extract CLS token representation from BERT"""
    outputs = bert(input_ids, attention_mask)
    sequence_output = outputs[0]
    return sequence_output[:, 0, :]  # take <CLS> token


def ned_torch(x1: torch.Tensor, x2: torch.Tensor, dim=1, eps=1e-8) -> torch.Tensor:
    """Compute normalized euclidean distance"""
    if len(x1.size()) == 1:
        ned_2 = 0.5 * ((x1 - x2).var() / (x1.var() + x2.var() + eps))
    elif x1.size() == torch.Size([x1.size(0), 1]):
        ned_2 = 0.5 * ((x1 - x2) ** 2 / (x1 ** 2 + x2 ** 2 + eps)).squeeze()
    else:
        ned_2 = 0.5 * ((x1 - x2).var(dim=dim) / (x1.var(dim=dim) + x2.var(dim=dim) + eps))
    return ned_2 ** 0.5


def nes_torch(x1, x2, dim=1, eps=1e-8):
    """Normalized Euclidean similarity"""
    return 1 - ned_torch(x1, x2, dim, eps)