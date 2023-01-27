#DataLoader

# Load StpTokenizer models
import argparse
import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as sp
from torchtext.vocab import *


# set variables
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description = 'Please input variables')

#     # Required parameter
#     parser.add_argument(
#         "--corpus_name",
#         type=str,
#         required=True,
#     )    

#     parser.add_argument(
#         "--save_dir",
#         default= './models/',
#         type=str,
#         required=False,
#     )
    
#     parser.add_argument(
#         "--dataset_dir",
#         default= './../dataset/',
#         type=str,
#         required=False,
#     )
    
#     parser.add_argument(
#         "--vocab_size",
#         default='32000',
#         type=int,
#         required=False,
#     )

#     parser.add_argument(
#         "--character_coverage",
#         default= 0.9999,
#         type=int,
#         required=False,
#     )
        
#     parser.add_argument(
#         "--model_type",
#         default='bpe',
#         type=str,
#         required=False,
#     )

#     parser.add_argument(
#         "--batch_size",
#         default = 128,
#         type = int,
#         required = False,
#     )    
    
#     args = parser.parse_args()

# Custom dataset

class CustomDataset(Dataset):
    def __init__(self):
        SPP_en = sp.SentencePieceProcessor()
        with open ("../Transformer_Reimplementation/dataset/IWSLT_2016.en", "r", encoding="utf-8") as f1:
            en_list = f1.read().splitlines()
            
        tokenized_lines_en = []
        
        SPP_en.Load(model_file = 
            "Tokenizer/models/tokenized_IWSLT_2016.en_32000/tokenized_IWSLT_2016.en_32000.model")
        for lines in en_list:
            tokenized_lines_en.append(SPP_en.EncodeAsIds(lines))
        
        max_len = max(len(item) for item in tokenized_lines_en)
                   
        for line in tokenized_lines_en:
            while len(line) < max_len:
                line.append(3)
        self.en_list = tokenized_lines_en
        
        
        SPP_de = sp.SentencePieceProcessor()
        with open ("../Transformer_Reimplementation/dataset/IWSLT_2016.de", "r", encoding="utf-8") as f2:
            de_list = f2.read().splitlines() 
            
        tokenized_lines_de = []
        
        SPP_de.Load(model_file = 
            "Tokenizer/models/tokenized_IWSLT_2016.de_32000/tokenized_IWSLT_2016.de_32000.model")
        for lines in de_list:
            tokenized_lines_de.append(SPP_de.EncodeAsIds(lines))     
        
        max_len = max(len(item) for item in tokenized_lines_de)
        
        for line in tokenized_lines_de:
            while len(line) < max_len:
                line.append(3)
        self.de_list = tokenized_lines_de

    def __len__(self):
        return len(self.en_list)

    def __getitem__(self,idx):
        en = torch.FloatTensor(self.en_list[idx])
        de = torch.FloatTensor(self.de_list[idx])
        return en, de


    
    # def build_vocab(Stp_de, Stp_en):
        
        
    #     def tokenize_de(text):
    #         return tokenize(text, Stp_de)

    #     def tokenize_en(text):
    #         return tokenize(text, Stp_en)
    
    #     print("Building German Vocabulary ...")
    #     train, val, test = dataset_de.de_data(language_pair=("de", "en"))
    #     vocab_src = build_vocab_from_iterator(
    #         yield_tokens(train + val + test, tokenize_de, index=0),
    #         min_freq=2,
    #         specials=["<s>", "</s>", "<blank>", "<unk>"],
    #     )

    #     print("Building English Vocabulary ...")
    #     train, val, test = dataset_en.en_data(language_pair=("de", "en"))
    #     vocab_tgt = build_vocab_from_iterator(
    #         yield_tokens(train + val + test, tokenize_en, index=1),
    #         min_freq=2,
    #         specials=["<s>", "</s>", "<blank>", "<unk>"],
    #     )

    #     vocab_src.set_default_index(vocab_src["<unk>"])
    #     vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    #     return vocab_src, vocab_tgt
    
    
    # def load_vocab(Stp_de, Stp_en):
    #     if not os.path.exists("vocab.pt"):
    #         vocab_src, vocab_tgt = build_vocabulary(Stp_de, Stp_en)
    #         torch.save((vocab_src, vocab_tgt), "vocab.pt")
    #     else:
    #         vocab_src, vocab_tgt = torch.load("vocab.pt")
    #     print("Finished.\nVocabulary sizes:")
    #     print(len(vocab_src))
    #     print(len(vocab_tgt))
    #     return vocab_src, vocab_tgt

dataset = CustomDataset()

dataloader = DataLoader(
    dataset, 
    batch_size = 128, # argparse 사용 예정
    shuffle = False, 
)

for i, batch in enumerate(dataloader): # batch[0]에는 en_list가 들어가고 batch[1]에는 de_list가 들어감
    print(batch[0].int())
    print(batch[1].int())








# a = dataset.en_list[:5]
# b = []
# for x in a:
#     for k in x:
#         b.append(int(k))
# SPP_en = sp.SentencePieceProcessor()
# SPP_en.Load(model_file = "Tokenizer/models/tokenized_IWSLT_2016.en_32000/tokenized_IWSLT_2016.en_32000.model")
# print(SPP_en.DecodeIds(b))


# c = dataset.de_list[:5]
# d = []
# for x in c:
#     for k in x:
#         d.append(int(k))
# SPP_de = sp.SentencePieceProcessor()
# SPP_de.Load(model_file = "Tokenizer/models/tokenized_IWSLT_2016.de_32000/tokenized_IWSLT_2016.de_32000.model")
# print(SPP_de.DecodeIds(d))