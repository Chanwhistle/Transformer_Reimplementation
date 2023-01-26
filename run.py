# Preprocessing

import sys
import torch
from DataLoader.CustomDataset import CustomDataset
from DataLoader.DataLoader import *
from Tokenizer.Tokenizer import *
from torchtext.data import Field # torchtext == 0.6.0
import argparse

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


StpTokenizer = Stp()

def StpTokenize_en(en_text_list):
    Stp_en = StpTokenizer.Load(model_file = "Tokenizer/models/tokenized_train.en_32000/tokenized_train.en_32000.model")
    Stp_en = StpTokenizer.Encode(en_text_list)
    return [token for token in Stp_en]
    
def StpTokenize_de(de_text_list):
    Stp_de = StpTokenizer.Load(model_file = "Tokenizer/models/tokenized_train.de_32000/tokenized_train.de_32000.model")
    Stp_de = StpTokenizer.Encode(de_text_list)
    return [token for token in Stp_de]

StpTokenizer = Stp()
if args.lang == "en":
    Stp_en = StpTokenizer.Load(model_file = "Tokenizer/models/tokenized_train.en_32000/tokenized_train.en_32000.model")

elif args.lang == "de":
    Stp_de = StpTokenizer.Load(model_file = "Tokenizer/models/tokenized_train.de_32000/tokenized_train.de_32000.model")


SRC = Field(tokenize = StpTokenize_en, 
            init_token = "<sos>", 
            eos_token = "<eos>", 
            lower = True, 
            batch_first = True
            )

TRG = Field(tokenize = StpTokenize_de, 
            init_token = "<sos>", 
            eos_token = "<eos>", 
            lower = True, 
            batch_first = True
            )


# print(StpTokenize_en(dataset.en_data[:1]))
# print(StpTokenize_de(dataset.en_data[:1]))

# print(StpTokenizer.Decode(StpTokenize_en(dataset.en_data[:1])))
# print(StpTokenizer.Decode(StpTokenize_de(dataset.de_data[:1])))

# for i, token in enumerate (StpTokenize_en(dataset.en_data[:5])):
#     print(f"인덱스 {i} : {token}")
       
       
