# Custom dataset

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self):
        f1 = open ("../dataset/train.en", "r", encoding="utf-8")
        en_data = f1.readlines()
        self.en_data = en_data
        
        f2 = open ("../dataset/train.de", "r", encoding="utf-8")
        de_data = f2.readlines()
        self.de_data = de_data        

    def __len__(self):
        return len(self.en_data)

    def __getitem__(self,idx):
        en = torch.FloatTensor(self.en_data[idx])
        de = torch.FloatTensor(self.de_data[idx])
        return en, de