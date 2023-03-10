import math
import torch
import torch.nn as nn

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
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, query, key, value, mask = None, e = 1e-12):
        # input = 4 dimention tensor [batch_size, n_head, length, d_tensor]
        # n_head * d_tensor = d_embed
        batch_size, n_head, length, d_tensor = key.size()
        
        # dot product Query with Key_t to compute similarity
        key_t = key.reshape(batch_size, n_head, d_tensor, length)
        attn_score = torch.matmul(query, key_t) / math.sqrt(d_tensor)
        
        # applying masking (optional)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, e)
        
        # pass softmax to make [0,1] range
        attn_prob = self.softmax(attn_score)       # (batch_size, n_head, length, length)
        
        # Multiply with Value
        value = torch.matmul(attn_prob, value)     # (batch_size, n_head, length, d_tensor)
        
        return value, attn_prob

    
'''
Multi Head Attention
'''

class MultiHeadAttention(nn.Module):
    def __init__(self, d_embed, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.d_embed = d_embed
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_query = nn.Linear(d_embed, d_model)
        self.w_key = nn.Linear(d_embed, d_model)
        self.w_value = nn.Linear(d_embed, d_model)
        self.w_concat = nn.Linear(d_model, d_embed)
        
    def forward(self, query, key, value, mask = None):
        '''
        query, key, value: (n_batch, length, d_model)
        mask: (n_batch, length, length)
        return value: (n_batch, length, d_model)
        '''
        # 1. dot product with weight matrix
        query, key ,value = self.w_query(query), self.w_key(key), self.w_value(value)
        
        # 2. split tensor by number of heads
        query, key ,value = self.split(query), self.split(key), self.split(value)
        
        # 3. calculate scale dot product to compute similarity
        out, attn_prob = self.attention(query, key ,value, mask = mask)
        
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)
        
        # 5. visualize attention map
        # Todo : visualization
        return out, attn_prob
        
    def split(self, tensor):
        '''
        split tensor by number of head
                
        input = [batch_size, length, d_embed]
        output = [batch_size, n_head, length, d_tensor]
        d_embed = n_head * d_tensor
        '''
        batch_size, length, d_embed = tensor.size()
        d_tensor = d_embed//self.n_head
        tensor = tensor.reshape(batch_size, self.n_head, length, d_tensor)
        return tensor
    
    def concat(self, tensor):
        '''
        Inverse function of self.split
        
        input = [batch_size, head, length, d_model]
        output = [batch_size, length, d_embed]
        '''
        batch_size, self.n_head, length, d_tensor = tensor.size()
        tensor = tensor.reshape(batch_size, length, self.n_head*d_tensor)
        return tensor
    

    