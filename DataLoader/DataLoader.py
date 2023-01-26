#DataLoader

# Load StpTokenizer models
import argparse
import os
import torch
from torch.utils.data import Dataset
from DataLoader.CustomDataset import CustomDataset
from torch.utils.data import DataLoader

# set variables
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Please input variables')

    # Required parameter
    parser.add_argument(
        "--batch_size",
        default = 128,
        type = int,
        required = False,
    )    
    
    args = parser.parse_args()


dataset = CustomDataset()


def my_collate_fn(dataset):
    en_text, de_text, = [], []
  
    for (_en,_de) in dataset:
        en_text.append(_en)
        de_text.append(_de)
  
    return torch.Tensor(en_text), torch.Tensor(de_text)


dataloader = DataLoader(
    dataset, 
    batch_size = 128,
    shuffle = False, 
    collate_fn = my_collate_fn
)