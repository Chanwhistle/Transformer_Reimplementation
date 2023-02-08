import torch
import torch.nn as nn


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
        pos = torch.arange(0, max_len, device=device)
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