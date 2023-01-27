'''
Transformer

hyperparameter = d_model, max_len, num_layers, num_heads, d_ff

Input = Tensor
        shape()
        

Output = Tensor
         shape()
'''

import torch
from torch import nn
import numpy as np
import math


'''
Positional Encoding
'''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        '''
        sin, cos encoding
        
        parameter
        - d_model : dimention of model
        - max_len : max length of input sequence
        - device : cuda or cpu
        '''
        super(PositionalEncoding, self).__init__() # reset nn.Module
        
        # create tensor that has same size with embedding vector(max_len, d_model)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False # Encoding doesn't need gradient     
        
        # pos is tensor that have size of max_len
        pos = torch.arange(0, max_len, device =device)
        # unsqueeze 1D (max_len, ) -> 2D(max_len, 1) to represent position of word
        pos = pos.float().unsqueeze(dim=1) # int64 -> float32
        
        # _2i : (d_model, ) size, i is index of d_model
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        
        # (max_len, 1) / (d_model/2 ) -> (max_len, d_model/2)
        self.encoding[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        
        
    def forward(self, x):
        # x.shape : (batch, seq_len) or (batch, seq_len, d_model)
        seq_len = x.size()[1] 
        
        # return : (seq_len, d_model)
        # return matrix will be added to x by broadcasting
        return self.encoding[:seq_len, :]
    


'''
Scale Dot Product Attention
'''

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        '''
        Compute scaled dot product attention
        calculate attention score
        
        Query = given sentence that we focused on(decoder)
        Key = every sentence to check relationship with Query(encoder)
        Value = every sentence same with Key(encoder)
        '''        
        super(ScaleDotProductAttention).__init__()  # reset nn.Module
        self.softmax = nn.Softmax()
    
    def forward(self, q, k ,v , mask = None, e = 1e-12):
        
        # input = 4 dimention tensor [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()
        
        # dot product Query with Key_T to compute similarity
        k_t = k.reshape(batch_size, head, d_tensor, length)
        score = np.matmul(q @ k_t) / math.sqrt(d_tensor)
        
        # applying masking (optional)
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)
        
        # pass softmax to make [0,1] range
        score = self.softmax(score)
        
        # Multiply with Value
        v= np.matmul(score, v)
        
        return v, score
    
    
    
'''
Multi Head Attention
'''

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        
    def split(self, tensor):
        '''
        split tensor by number of head
        
        input = [batch_size, length, d_model]
        output = [batch_size, head, length, d_model]
        
        d_model = head * d_tensor
        '''
        batch_size, length, d_model = tensor.size()
        
        d_tensor = d_model//self.n_head
        
        tensor = tensor.reshape(batch_size, self.n_head, length, d_tensor)
        
        return tensor
    
    def concat(self, tensor):
        '''
        Inverse function of self.split
        
        input = [batch_size, head, length, d_model]
        output = [batch_size, length, d_model]
        '''
        batch_size, head, length, d_tensor = tensor.size()
        
        d_model = d_tensor * self.n_head
        
        tensor = tensor.reshape(batch_size, length, d_model)
        
        return tensor
    
    def forward(self, q, k ,v , mask = None):
        q, k ,v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        q, k ,v = self.split(q), self.split(k), self.split(v)     
        
        out, attention = self.attention(q, k ,v, mask = mask)
        
        out = self.concat(out)
        out = self.w_concat(out)
        
        return out
    
    

'''
Layer Normalization
'''

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        
        out = (x-mean)/(std+self.eps)
        out = self.gamma * out + self.beta
        return out
    
    
    
'''
(Position Wise) Feed Forward Network
'''

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = drop_prob)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

