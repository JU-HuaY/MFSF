import torch.nn as nn
import torch.nn.functional as F
import torch
from network.utils import *
import math
import numpy as np

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        # print(value.shape)
        # print(scores.shape)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attns = dropout(p_attn)

        return torch.matmul(p_attns, value), scores


class MultiHeadAttention(nn.Module):
    """
    Take in model size and number of heads.
    """
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        # self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        querys, keys, values = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(querys, keys, values, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = (x + querys).transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return x, attn

class Covariance_Attention(nn.Module):
    """
    Compute 'Covariance
    """
    def forward(self, qk, v, mask=None, dropout=None):
    	mean = torch.mean(qk, 2, keepdim=True)
        x = qk - mean.expand(-1, -1, qk.size(2))
        scores = torch.bmm(x, x.transpose(1, 2)) / math.sqrt(qk.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attns = dropout(p_attn)
        return torch.matmul(p_attns, value), scores


class Attention_ADJ(nn.Module):
    """
    Take in model size and number of heads.
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.qk_layers = nn.Linear(d_model, d_model)
        # self.k_layers = nn.Linear(d_model, d_model)
        self.v_layers = nn.Linear(d_model, d_model)
        self.attention = Covariance_Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, qk, v, mask=None):
        batch_size = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        qks = self.qk_layers(qk)
        vs = self.v_layers(v)
        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(qks, vs, mask=mask, dropout=self.dropout)
        return x, attn
