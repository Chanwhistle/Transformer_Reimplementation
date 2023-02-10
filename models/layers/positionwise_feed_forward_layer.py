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
        x = torch.Tensor(x[0])
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
