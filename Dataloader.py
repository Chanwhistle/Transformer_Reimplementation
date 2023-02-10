#DataLoader

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as sp
from tqdm import tqdm
import pickle
from utils import *

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
        self.specials={
                self.unk: self.unk_idx,
                self.pad: self.pad_idx,
                self.bos: self.bos_idx,
                self.eos: self.eos_idx
                }
        self.data_path = data_path
        
        self.vocab_src = None
        self.vocab_tgt = None

        train_file = os.path.join(data_path, "train.pickle")
        dev_file = os.path.join(data_path, "dev.pickle")
        test_file = os.path.join(data_path, "test.pickle")

        if os.path.exists(train_file):
            self.train = load_pkl(train_file)
        else:
            trg_train, src_train = self.load_corpus("train")
            self.train = [(en, de) for en, de in zip(trg_train, src_train)]
            save_pkl(self.train , train_file)
            
        self.train_trg_tok = self.tokenize(f"{self.trg_lang}_32000",
                                                self.train[0], "train")
        self.train_src_tok = self.tokenize(f"{self.src_lang}_32000",
                                                self.train[1], "train")
                
        self.train_t = [(en, de) for en, de in zip(self.to_tensor(self.train_trg_tok), self.to_tensor(self.train_src_tok))]
        print("built train dataset")
        
        
        if os.path.exists(dev_file):
            self.dev = load_pkl(dev_file)
        else:
            trg_dev, src_dev = self.load_corpus("dev")
            self.dev = [(en, de) for en, de in zip(trg_dev, src_dev)]
            save_pkl(self.dev , dev_file)
            
        self.dev_trg_tok = self.tokenize(f"{self.trg_lang}_32000",
                                              self.dev[0], "dev")
        self.dev_src_tok = self.tokenize(f"{self.src_lang}_32000",
                                              self.dev[1], "dev")
        self.dev_t = [(en, de) for en, de in zip(self.to_tensor(self.dev_trg_tok), self.to_tensor(self.dev_src_tok))]
        print("built dev dataset")
            
            
        if os.path.exists(test_file):
            self.test = load_pkl(test_file)
        else:
            trg_test, src_test = self.load_corpus("dev")
            self.test = [(en, de) for en, de in zip(trg_test, src_test)]
            save_pkl(self.test , test_file)
            
        self.test_trg_tok = self.tokenize(f"{self.trg_lang}_32000",
                                               self.test[0], "test")
        self.test_src_tok = self.tokenize(f"{self.src_lang}_32000",
                                               self.test[1], "test")
        self.test_t = [(en, de) for en, de in zip(self.to_tensor(self.test_trg_tok), self.to_tensor(self.test_src_tok))]
        print("built test dataset")
        
        
        with open("./Tokenizer/vocab/en_32000/en_32000.vocab", encoding = "utf-8") as f:
            self.vocab_trg = f.read().splitlines()
        with open("./Tokenizer/vocab/de_32000/de_32000.vocab", encoding = "utf-8") as f:               
            self.vocab_src = f.read().splitlines()
    

    def __len__(self):

        return len(self.train_trg_tok)


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


    def load_corpus(self, type):
        trg_corpus, src_corpus = [], []
        corpus_dict = {"train": ["IWSLT16.TED.train"],
                      
                      "test": ["IWSLT16.TEDX.tst2014", 
                               "IWSLT16.TEDX.tst2013", 
                               "IWSLT16.TED.tst2014", 
                               "IWSLT16.TED.tst2013", 
                               "IWSLT16.TED.tst2012", 
                               "IWSLT16.TED.tst2011", 
                               "IWSLT16.TED.tst2010"],
                      
                      "dev": ["IWSLT16.TED.dev2010", 
                              "IWSLT16.TEDX.dev2012"]}
        
        for corpus in corpus_dict.get(type):
            trg_path = os.path.join(self.data_path, type, 
                                    f"{corpus}.{self.src_lang}-{self.trg_lang}.{self.trg_lang}.xml")
            src_path = os.path.join(self.data_path, type, 
                                    f"{corpus}.{self.src_lang}-{self.trg_lang}.{self.src_lang}.xml")
            with open(trg_path, encoding = "utf-8") as f:
                temp_trg = f.read().splitlines()
            with open(src_path, encoding = "utf-8") as f:
                temp_src = f.read().splitlines()
            # assert(len(temp_trg) == len(temp_src)), "Vocab size is different!!"
            trg_corpus += temp_trg
            src_corpus += temp_src
        print("Loaded Corpus! \nCorpus sizes:")
        print("target Corpus : ", len(trg_corpus))
        print("source Corpus : ", len(src_corpus))   

        return trg_corpus, src_corpus
    
            
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
            
            
    def to_tensor(self, sentences):
        tensor_sent = []
        for sentence in tqdm(sentences):
            T = torch.IntTensor(sentence)
            tensor_sent.append(T)
        return tensor_sent
            
            
    def my_collate_fn(self, samples):
        trg = [sample[0] for sample in samples]
        src = [sample[1] for sample in samples]
        batch_trg = pad_sequence(trg, batch_first=True, padding_value = int(self.pad_idx))  # Padding
        batch_src = pad_sequence(src, batch_first=True, padding_value = int(self.pad_idx))  # Padding
        return (batch_src, batch_trg)       

        
    def get_iter(self, **kwargs):
        train_iter = DataLoader(self.train_t,
                                collate_fn=self.my_collate_fn,
                                **kwargs)
        dev_iter = DataLoader(self.dev_t,
                                collate_fn=self.my_collate_fn,
                                **kwargs)
        test_iter = DataLoader(self.test_t,
                               collate_fn=self.my_collate_fn,
                               **kwargs)
        return train_iter, dev_iter, test_iter