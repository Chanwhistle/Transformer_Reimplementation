import torch
import torch.nn as nn

    
'''
(Position Wise) Feed Forward Network
'''

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_embed, hidden, drop_prob):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_embed, hidden)
        self.linear2 = nn.Linear(hidden, d_embed)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        
    def forward(self, x):
        out = x
        out = self.linear1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out
