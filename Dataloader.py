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
        self.unk = "<unk>"
        self.pad = "<pad>"
        self.bos = "<bos>"
        self.eos = "<eos>"
        self.data_path = data_path

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
        train_en = torch.IntTensor(self.train_trg_tok_list[idx])      # dataset[0] = encoded train_en data
        train_de = torch.IntTensor(self.train_src_tok_list[idx])      # dataset[1] = encoded train_de data
        test_en = torch.IntTensor(self.test_trg_tok_list[idx])        # dataset[2] = encoded test_en data
        test_de = torch.IntTensor(self.test_src_tok_list[idx])        # dataset[3] = encoded test_de data
        dev_en = torch.IntTensor(self.dev_trg_tok_list[idx])          # dataset[4] = encoded dev_en data
        dev_de = torch.IntTensor(self.dev_src_tok_list[idx])          # dataset[5] = encoded dev_de data

        return train_en, train_de, test_en, test_de, dev_en, dev_de


    def tokenizer(self, model):
        SP = sp.SentencePieceProcessor()
        SP_temp = SP.Load(model_file =
        "Tokenizer/vocab/"f"{model}/"f"{model}.model")
        if SP_temp == False:
            self.build_vocab()
            SP_temp = SP.Load(model_file = 
            "Tokenizer/vocab/"f"{model}/"f"{model}.model")
        
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
                      
                      "test": ["IWSLT16.TEDX.tst2014", 
                               "IWSLT16.TEDX.tst2013", 
                               "IWSLT16.TED.tst2014", 
                               "IWSLT16.TED.tst2013", 
                               "IWSLT16.TED.tst2012", 
                               "IWSLT16.TED.tst2011", 
                               "IWSLT16.TED.tst2010"],
                      
                      "dev": ["IWSLT16.TED.dev2010", 
                              "IWSLT16.TEDX.dev2012"]}
        
        for dataset in vocab_dict.get(type):
            trg_path = os.path.join(self.data_path, type, 
                                    f"{dataset}.{self.src_lang}-{self.trg_lang}.{self.trg_lang}.xml")
            src_path = os.path.join(self.data_path, type, 
                                    f"{dataset}.{self.src_lang}-{self.trg_lang}.{self.src_lang}.xml")
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
    
            
    def build_vocab(self):
        # building German Vocab
        Stp_trainer = sp.SentencePieceTrainer
        vocab_size = 32000
        model_type = "bpe"
        character_coverage = 0.9999
        dataset_dir = f"{self.data_path}train/"
        corpus = "IWSLT16.TED.train"
        languages =["en", "de"]
        for lang in languages:
            model_name = f"{lang}_{vocab_size}"
            save_dir = "./Tokenizer/vocab/" + model_name
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            model_path_en = os.path.join(save_dir, model_name)
            data4vocab = os.path.join(dataset_dir, f"{corpus}.{self.src_lang}-{self.trg_lang}.{self.trg_lang}.xml")
            input_argument = ' '.join(['--input=%s',
                                       '--model_prefix=%s',
                                       '--vocab_size=%s',
                                       '--model_type=%s',
                                       '--character_coverage=%s',
                                       '--unk_id=%s',
                                       '--unk_piece=%s',
                                       '--pad_id=%s',
                                       '--pad_piece=%s',
                                       '--bos_id=%s',
                                       '--bos_piece=%s',
                                       '--eos_id=%s', 
                                       '--eos_piece=%s']
                                      )
            input = input_argument % (data4vocab, 
                                      model_path_en, 
                                      vocab_size, 
                                      model_type, 
                                      character_coverage, 
                                      self.unk_idx, 
                                      self.unk, 
                                      self.pad_idx, 
                                      self.pad, 
                                      self.bos_idx, 
                                      self.bos, 
                                      self.eos_idx, 
                                      self.eos)
            
            Stp_trainer.Train(input)
            print(f"{lang} Vocab Generation Finished")
            

    def train_collate_fn(self, samples):
        trg = [sample[0] for sample in samples]
        src = [sample[1] for sample in samples]
        padded_trg_train = pad_sequence(trg, batch_first=True, padding_value = int(self.pad_idx))  # Padding
        padded_src_train = pad_sequence(src, batch_first=True, padding_value = int(self.pad_idx))  # Padding
        return {'padded_trg_train': padded_trg_train.contiguous(),
                'padded_src_train': padded_src_train.contiguous()}
        
        
    def test_collate_fn(self, samples):
        trg = [sample[2] for sample in samples]
        src = [sample[3] for sample in samples]
        padded_trg_test = pad_sequence(trg, batch_first=True, padding_value = int(self.pad_idx))  # Padding
        padded_src_test = pad_sequence(src, batch_first=True, padding_value = int(self.pad_idx))  # Padding
        return {'padded_trg_train': padded_trg_test.contiguous(),
                'padded_src_train': padded_src_test.contiguous()}
        
        
    def dev_collate_fn(self, samples):
        trg = [sample[4] for sample in samples]
        src = [sample[5] for sample in samples]
        padded_trg_dev = pad_sequence(trg, batch_first=True, padding_value = int(self.pad_idx))  # Padding
        padded_src_dev = pad_sequence(src, batch_first=True, padding_value = int(self.pad_idx))  # Padding
        return {'padded_trg_train': padded_trg_dev.contiguous(),
                'padded_src_train': padded_src_dev.contiguous()}
        
        
    def get_iter(self, **kwargs):
        train_iter = DataLoader(CustomDataset, collate_fn=self.train_collate_fn, **kwargs)
        valid_iter = DataLoader(CustomDataset, collate_fn=self.test_collate_fn, **kwargs)
        test_iter = DataLoader(CustomDataset, collate_fn=self.dev_collate_fn, **kwargs)
        return train_iter, valid_iter, test_iter

dataset=CustomDataset()