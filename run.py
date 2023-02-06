# Preprocessing

import sys
import argparse
import os
from os.path import exists
import math
import copy
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from . import model

# set variables
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Please input variables')

    # Required parameter
    parser.add_argument(
        "--lang",
        default = "en", #de
        type = str,
        required = False,
    )    
    
    args = parser.parse_args()

