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





class IWSLTDataset(Dataset):
    def __init__(self, datapath = "./dataset/", src_lang = "de", trg_lang = "en", type = "train"):
        super().__init__()
        
        SP = sp.SentencePieceProcessor()
        
        if type == "train":
            datasets = ["IWSLT16"]
            trg = []
            src = []
            for dataset in datasets:
                trg_path = os.path.join(datapath, type, f"{type}.{trg_lang}")
                src_path = os.path.join(datapath, type, f"{type}.{src_lang}")
                with open(trg_path, encoding = "utf-8") as f:
                    temp_trg = f.read().splitlines()
                with open(src_path, encoding = "utf-8") as f:
                    temp_src = f.read().splitlines()
                trg += temp_trg
                src += temp_src
                   
            self.trg_tok_list = self.tokenize("en_32000", SP, trg)
            self.src_tok_list = self.tokenize("de_32000", SP, src)
                       
        elif type == "tst" or type == "dev":
            type = "tst"
            datasets = ["IWSLT16.TED.tst2014", "IWSLT16.TED.tst2013", "IWSLT16.TED.tst2012", "IWSLT16.TED.tst2011", "IWSLT16.TED.tst2010"]
            trg = []
            src = []
            for dataset in datasets:
                trg_path = os.path.join(datapath, type, f"{dataset}.{src_lang}-{trg_lang}.{trg_lang}.xml")
                src_path = os.path.join(datapath, type, f"{dataset}.{src_lang}-{trg_lang}.{src_lang}.xml")
                with open(trg_path, encoding = "utf-8") as f:
                    temp_trg = f.read().splitlines()
                with open(src_path, encoding = "utf-8") as f:
                    temp_src = f.read().splitlines()
                assert(len(temp_src) == len(temp_trg))
                trg += temp_trg
                src += temp_src
                
            self.trg_tok_list = self.tokenize("en_32000", self.tokenizer(model="en_32000"), trg)
            self.src_tok_list = self.tokenize("de_32000", self.tokenizer(model="de_32000"), src)
                
            type = "dev"
            datasets = ["IWSLT16.TED.dev2012", "IWSLT16.TED.dev2010"]
            trg = []
            src = []
            for dataset in datasets:
                trg_path = os.path.join(datapath, type, f"{dataset}.{src_lang}-{trg_lang}.{trg_lang}.xml")
                src_path = os.path.join(datapath, type, f"{dataset}.{src_lang}-{trg_lang}.{src_lang}.xml")
                with open(trg_path, encoding = "utf-8") as f:
                    temp_trg = f.read().splitlines()
                with open(src_path, encoding = "utf-8") as f:
                    temp_src = f.read().splitlines()
                assert(len(temp_trg) == len(temp_src))
                trg += temp_trg
                src += temp_src
                
            self.trg_tok_list = self.tokenize("en_32000", self.tokenizer(model="en_32000"), trg)
            self.src_tok_list = self.tokenize("de_32000", self.tokenizer(model="de_32000"), src)
        
        self.src = src
        self.trg = trg
                
    def __len__(self):
        return len(self.src)

    def __getitem__(self,idx):
        en = torch.FloatTensor(self.trg_tok_list[idx])
        de = torch.FloatTensor(self.src_tok_list[idx])
        return en, de
    
    def tokenizer(self, model):
        if model == "en_32000":
            SP_en = sp.SentencePieceProcessor.Load(model_file = 
            "Tokenizer/models/"f"{model}.model")
            return SP_en
            
        elif model =="de_32000":
            SP_de = sp.SentencePieceProcessor.Load(model_file = 
            "Tokenizer/models/"f"{model}.model")
            return SP_de

        
    def tokenize(self, model, tokenizer, vocab):
        model_name = f"tokenized_IWSLT_2016.{model}"
        tokenized = []      
        tokenizer.Load(model_file = 
            "Tokenizer/models/"f"{model_name}.model")
        for lines in vocab:
            tokenized.append(tokenizer.EncodeAsIds(lines))
        
        max_len = max(len(item) for item in tokenized)
                
        for line in tokenized:
            while len(line) < max_len:
                line.append(3)

        return tokenized
    
dataset = IWSLTDataset()

dataloader = DataLoader(
    dataset,
    batch_size = 32, # argparse 사용 예정
    shuffle = True, 
)

for i, batch in enumerate(dataloader): # batch[0]에는 en_list가 들어가고 batch[1]에는 de_list가 들어감
    print(f"{i+1}번째 en_batch", batch[0].int())
    print(f"{i+1}번째 de_batch", batch[1].int())
    break