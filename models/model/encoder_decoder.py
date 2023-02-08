import copy
import torch
import torch.nn as nn

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
Layer cloning
'''

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
        self.n_layers = n_layers
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
    def __init__(self, size, self_attn, feed_forward, drop_prob, n_layers):
        super(EncoderLayer, self).__init__()
        self.n_layers = n_layers
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, drop_prob), self.n_layers)
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
        self.n_layers = n_layers
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, trg, encoder_out, trg_mask, src_tgt_mask):
        out = trg
        for layer in self.layers:
            trg = layer(out, encoder_out, trg_mask, src_tgt_mask)
        out = self.norm(out)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, cross_attn, feed_forward, drop_prob, n_layers):
        super(DecoderLayer, self).__init__()
        self.n_layers = n_layers
        self.size = size
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, drop_prob), self.n_layers)
        
    def forward(self, trg, encoder_out, trg_mask, src_tgt_mask):
        out = trg
        out = self.self_attn(query=out, key=out, value=out, mask=trg_mask)
        out = self.cross_attn(query=out, key=encoder_out, value=encoder_out, mask=src_tgt_mask)
        out = self.feed_forward(out)
        return out   