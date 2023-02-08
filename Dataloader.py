#DataLoader

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import Field, BucketIterator
import sentencepiece as sp
from tqdm import tqdm
import pickle
import tarfile
from urllib.request import urlretrieve
import urllib.request

class CustomDataset(Dataset):
    def __init__(self,
                 src_lang = "de",
                 trg_lang = "en",
                 unk_idx = 0,
                 pad_idx = 1,
                 bos_idx = 2,
                 eos_idx = 3,
                 data_path = "./dataset/"):
        
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.unk_idx = unk_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.data_path = data_path
        self.unk = "<unk>"
        self.pad = "<pad>"
        self.bos = "<bos>"
        self.eos = "<eos>"

        trg, src = self.load_vocab("train")
        self.train_trg_tok_list = self.tokenize(f"{self.trg_lang}_32000", trg, "train")
        self.train_src_tok_list = self.tokenize(f"{self.src_lang}_32000", src, "train")
        print("builded train dataset")

        trg, src = self.load_vocab("test")
        self.test_trg_tok_list = self.tokenize(f"{self.trg_lang}_32000", trg, "test")
        self.test_src_tok_list = self.tokenize(f"{self.src_lang}_32000", src, "test")
        print("builded test dataset")
     
        trg, src = self.load_vocab("dev")
        self.dev_trg_tok_list = self.tokenize(f"{self.trg_lang}_32000", trg, "dev")
        self.dev_src_tok_list = self.tokenize(f"{self.src_lang}_32000", src, "dev")
        print("builded dev dataset")
        
        self.src = src



    def __len__(self):

        return len(self.src)


    def __getitem__(self,idx):
        train_en = torch.IntTensor(self.train_trg_tok_list[idx])
        train_de = torch.IntTensor(self.train_src_tok_list[idx])
        test_en = torch.IntTensor(self.test_trg_tok_list[idx])
        test_de = torch.IntTensor(self.test_src_tok_list[idx])    
        dev_en = torch.IntTensor(self.dev_trg_tok_list[idx])   
        dev_de = torch.IntTensor(self.dev_src_tok_list[idx])        

        return train_en, train_de, test_en, test_de, dev_en, dev_de


    def tokenizer(self, model):
        SP = sp.SentencePieceProcessor()
        SP_temp = SP.Load(model_file =
        "Tokenizer/models/"f"{model}/"f"{model}.model")
        if SP_temp == False:
            self.build_vocab()
            SP_temp = SP.Load(model_file = 
            "Tokenizer/models/"f"{model}/"f"{model}.model")
        
        else:
            pass
                  
        print("Loaded "f"{model[:2]}_Tokenizer")
        return SP


    def tokenize(self, model, vocab, type):
        tokenized = []      
        tokenizer = self.tokenizer(model)

        if os.path.isfile(f'./Tokenizer/EncodeAsIds_{model}_{type}.pickle') == False:
            print("Encoding As Id...") 
            for lines in tqdm(vocab):
                tokenized_temp = tokenizer.EncodeAsIds(lines)
                tokenized_temp.insert(0,self.bos_idx)        # insert <BOS>
                tokenized_temp.append(self.eos_idx)          # insert <EOS>
                tokenized.append(tokenized_temp)
            with open(f'./Tokenizer/EncodeAsIds_{model}_{type}.pickle', 'wb') as f:
                pickle.dump(tokenized, f, pickle.HIGHEST_PROTOCOL)
        
        else:                
            print(f"Loading Encoded {model[:2]} file!")
            with open(f'./Tokenizer/EncodeAsIds_{model}_{type}.pickle', 'rb') as f:
                tokenized = pickle.load(f)
        
        print(f"{model[:2]}_vocab tokenizing Finished!")
        return tokenized


    def load_vocab(self, type):
        trg_vocab, src_vocab = [], []
        vocab_dict = {"train": ["IWSLT16.TED.train"],
                      "test": ["IWSLT16.TEDX.tst2014", "IWSLT16.TEDX.tst2013", "IWSLT16.TED.tst2014", "IWSLT16.TED.tst2013", "IWSLT16.TED.tst2012", "IWSLT16.TED.tst2011", "IWSLT16.TED.tst2010"],
                      "dev": ["IWSLT16.TED.dev2010", "IWSLT16.TEDX.dev2012"]}
        for dataset in vocab_dict.get(type):
            trg_path = os.path.join(self.data_path, type, f"{dataset}.{self.src_lang}-{self.trg_lang}.{self.trg_lang}.xml")
            src_path = os.path.join(self.data_path, type, f"{dataset}.{self.src_lang}-{self.trg_lang}.{self.src_lang}.xml")
            with open(trg_path, encoding = "utf-8") as f:
                temp_trg = f.read().splitlines()
            with open(src_path, encoding = "utf-8") as f:
                temp_src = f.read().splitlines()
            # assert(len(temp_trg) == len(temp_src)), "Vocab size is different!!"
            trg_vocab += temp_trg
            src_vocab += temp_src
        print("Loaded Vocab! \nVocab sizes:")
        print("target vocab : ", len(trg_vocab))
        print("Source vocab : ", len(src_vocab))   

        return trg_vocab, src_vocab

    
    # def my_collate_fn(self, samples):
    #     trg = [sample[0] for sample in samples]
    #     src = [sample[1] for sample in samples]
    #     padded_trg = pad_sequence(trg, batch_first=True, padding_value = int(self.pad_idx))  # Padding
    #     padded_src = pad_sequence(src, batch_first=True, padding_value = int(self.pad_idx))  # Padding
    #     return {'padded_trg': padded_trg.contiguous(),
    #             'padded_src': padded_src.contiguous()}




dataset = CustomDataset()
print("Dataset Loaded")

# dataset.build_vocab()

def my_collate_fn(samples):
    trg = [sample[0] for sample in samples]
    src = [sample[1] for sample in samples]
    padded_trg = pad_sequence(trg, batch_first=True, padding_value = int(1))  # Padding
    padded_src = pad_sequence(src, batch_first=True, padding_value = int(1))  # Padding
    return {'padded_trg': padded_trg.contiguous(),
            'padded_src': padded_src.contiguous()}

dataloader = DataLoader(
    dataset,
    batch_size = 32, # argparse 사용 예정
    shuffle = True, 
    collate_fn = my_collate_fn
)

for i, batch in enumerate(tqdm(dataloader)): # batch[0]에는 en_list가 들어가고 batch[1]에는 de_list가 들어감
    print(f"{i+1}번째 en_batch\n", batch['padded_trg'])
    print(f"{i+1}번째 de_batch\n", batch['padded_src'])
    
    if i > 5:
        break
