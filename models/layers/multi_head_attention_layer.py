import math
import torch
import torch.nn as nn
import numpy as np

'''
Scale Dot Product Attention
'''

class ScaleDotProductAttention(nn.Module):
    '''
    Compute scaled dot product attention
    calculate attention score
    
    Query = given sentence that we focused on(decoder)
    Key = every sentence to check relationship with Query(encoder)
    Value = every sentence same with Key(encoder)
    '''     
    def __init__(self):   
        super(ScaleDotProductAttention, self).__init__()  # reset nn.Module
        self.softmax = nn.Softmax()
    
    def forward(self, query, key, value, mask = None, e = 1e-12):
        
        # input = 4 dimention tensor [batch_size, n_head, length, d_tensor]
        # n_head * d_tensor = d_model
        batch_size, n_head, length, d_tensor = key.size()
        
        # dot product Query with Key_t to compute similarity
        key_t = key.reshape(batch_size, n_head, d_tensor, length)
        attn_score = torch.matmul(query @ key_t) / math.sqrt(d_tensor)
        
        # applying masking (optional)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, e)
        
        # pass softmax to make [0,1] range
        attn_prob = self.softmax(attn_score)       # (batch_size, length, length)
        
        # Multiply with Value
        value = np.matmul(attn_prob, value)        # (batch_size, n_head, length, d_tensor)
        
        return value, attn_prob
    
    
    
'''
Multi Head Attention
'''

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.w_query = nn.Linear(d_model, d_model)
        self.w_key = nn.Linear(d_model, d_model)
        self.w_value = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        self.attention = ScaleDotProductAttention()
        
    def split(self, tensor):
        '''
        split tensor by number of head
        
        input = [batch_size, length, d_model]
        output = [batch_size, head, length, d_tensor]
        d_model = n_head * d_tensor
        '''
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model//self.n_head
        tensor = tensor.reshape(batch_size, self.n_head, length, d_tensor)
        return tensor
    
    def concat(self, tensor):
        '''
        Inverse function of self.split
        
        input = [batch_size, head, length, d_tensor]
        output = [batch_size, length, d_model]
        '''
        batch_size, head, length, d_tensor = tensor.size()
        d_model = d_tensor * self.n_head
        tensor = tensor.reshape(batch_size, length, d_model)
        return tensor
    
    def forward(self, query, key, value, mask = None):
        '''
        query, key, value: (n_batch, length, d_model)
        mask: (n_batch, length, length)
        return value: (n_batch, n_head, length, d_tensor)
        '''
        query, key ,value = self.w_query(query), self.w_key(key), self.w_value(value)
        query, key ,value = self.split(query), self.split(key), self.split(value)     
        out, attn_prob = self.attention(query, key ,value, mask = mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out, attn_prob
    
