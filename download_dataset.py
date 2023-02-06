# download dataset

import glob
import pandas as pd
import os
import urllib.request
from urllib.request import urlretrieve
import tarfile
    
    
# dataset = WMT14


# train
if os.path.exists("./dataset/train/commoncrawl.de-en.de"):
    with urllib.request.urlopen("https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz") as res:
        tarfile.open(fileobj=res, mode="r|gz").extractall("./dataset/train")
    
if os.path.exists("./dataset/train/europarl-v7.de-en.de"):
    with urllib.request.urlopen("https://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz") as res:
        tarfile.open(fileobj=res, mode="r|gz").extractall("./dataset/train")
