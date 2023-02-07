'''
Transformer

hyperparameter = d_model, max_len, n_layers, n_head, d_ff

Transformer input shape = (n_batch, seq_len)

Encoder input shape = (n_batch, length, d_model)
Decoder input shape = (n_batch, length, d_model)
        

Output = Tensor
         shape()
'''

import torch
import torch.nn as nn
import numpy as np
import math
import copy
import torch.nn.functional as F


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
    
    
class TokenEmbedding(nn.Module):

    def __init__(self, d_model, vocab_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model


    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_model)
        return out
    

class TransformerEmbedding(nn.Module):

    def __init__(self, token_embed, pos_embed):
        super(TransformerEmbedding, self).__init__()
        self.embedding = nn.Sequential(token_embed, pos_embed)


    def forward(self, x):
        out = self.embedding(x)
        return out


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
        super(MultiHeadAttention).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_query = nn.Linear(d_model, d_model)
        self.w_key = nn.Linear(d_model, d_model)
        self.w_value = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        
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
        return out
    
    

'''
Layer Normalization
'''

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        out = self.gamma * (x-mean)/(std+self.eps) + self.beta
        return out
    
    
    
'''
(Position Wise) Feed Forward Network
'''

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


'''
Sublayer Connection
'''

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, n_layers):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_layers)])


'''
Encoder
'''

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, n_layers):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, src, src_mask):
        "Pass the input (and mask) through each layer in turn."
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)
        out = self.norm(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, drop_prob):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, drop_prob), 6)
        self.size = size
        
    def forward(self, src, src_mask):
        out = src
        out = self.self_attn(query=out, key=out, value=out, mask=src_mask)
        out = self.feed_forward(out)
        return out        


'''
Decoder
'''

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, n_layers):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, trg, encoder_out, trg_mask, src_tgt_mask):
        out = trg
        for layer in self.layers:
            trg = layer(out, encoder_out, trg_mask, src_tgt_mask)
        out = self.norm(out)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, cross_attn, feed_forward, drop_prob):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, drop_prob), 6)
        
    def forward(self, trg, encoder_out, trg_mask, src_tgt_mask):
        out = trg
        out = self.self_attn(query=out, key=out, value=out, mask=trg_mask)
        out = self.cross_attn(query=out, key=encoder_out, value=encoder_out, mask=src_tgt_mask)
        out = self.feed_forward(out)
        return out        

        
        
'''
Transformer
'''
        
class Transformer(nn.Module):
    def __init__(self, src_embed, trg_embed, encoder, decoder, generator):
        super(Transformer, self).__init__()
        self.src_embed = src_embed   # src언어의 embeding
        self.trg_embed = trg_embed   # trg언어의 embeding
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        
        
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)


    def decode(self, trg, encoder_out, trg_mask, src_trg_mask):
        return self.decoder(self.trg_embed(trg), encoder_out, trg_mask, src_trg_mask)


    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        src_trg_mask = self.make_src_trg_mask(src, trg)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(trg, encoder_out, trg_mask, src_trg_mask)
        out = F.log_softmax(self.generator(decoder_out), dim=-1)
        return out, decoder_out


    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask


    def make_trg_mask(self, trg):
        pad_mask = self.make_pad_mask(trg, trg)
        seq_mask = self.make_subsequent_mask(trg, trg)
        mask = pad_mask & seq_mask
        return pad_mask & seq_mask
    
    
    def make_src_trg_mask(self, src, trg):
        pad_mask = self.make_pad_mask(trg, src)
        return pad_mask
   
    
    def make_pad_mask(self, query, key, pad_idx=3):
        # query: (n_batch, len_query)
        # key: (n_batch, len_key)
        len_query, len_key = query.size(1), key.size(1)

        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)
        key_mask = key_mask.repeat(1, 1, len_query, 1)    # (n_batch, 1, query_seq_len, key_seq_len)

        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, 1, len_key)  # (n_batch, 1, query_seq_len, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False
        return mask


    def make_subsequent_mask(self, query, key):
        len_query, len_key = query.size(1), key.size(1)

        mask = torch.tril(torch.ones(len_query, len_key)).type(torch.BoolTensor).to(self.device)
        return mask