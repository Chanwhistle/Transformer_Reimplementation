import torch
import torch.nn as nn
import math

'''
Token Embedding
'''
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_embed):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.d_embed = d_embed

    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_embed)
        return out
    
'''
Transformer Embedding
'''
class TransformerEmbedding(nn.Module):
    def __init__(self, token_embed, pos_embed, drop_prob = 0):
        super(TransformerEmbedding, self).__init__()
        self.embedding = nn.Sequential(token_embed, pos_embed)
        self.dropout = nn.Dropout(p = drop_prob)

    def forward(self, x):
        out = self.embedding(x)
        out = self.dropout(out)
        return out