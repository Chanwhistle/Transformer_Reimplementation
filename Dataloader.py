#DataLoader

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as sp
from tqdm import tqdm
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
        
        train_file = os.path.join(self.data_path, "train.pickle")
        dev_file = os.path.join(self.data_path, "dev.pickle")
        test_file = os.path.join(self.data_path, "test.pickle")

        if os.path.exists(train_file):
            self.train = load_pkl(train_file)
        else:
            trg_train, src_train = self.load_corpus("train")
            self.train = [(en, de) for en, de in zip(trg_train, src_train)]
            save_pkl(self.train , train_file)
        self.train_t = self.tokenize(self.train, "train")
        print("built train dataset")        
        
        if os.path.exists(dev_file):
            self.dev = load_pkl(dev_file)
        else:
            trg_dev, src_dev = self.load_corpus("dev")
            self.dev = [(en, de) for en, de in zip(trg_dev, src_dev)]
            save_pkl(self.dev , dev_file)
        self.dev_t = self.tokenize(self.dev, "dev")
        print("built dev dataset")
            
        if os.path.exists(test_file):
            self.test = load_pkl(test_file)
        else:
            trg_test, src_test = self.load_corpus("test")
            self.test = [(en, de) for en, de in zip(trg_test, src_test)]
            save_pkl(self.test , test_file)
        self.test_t = self.tokenize(self.test, "test")
        print("built test dataset")
        
        with open("./Tokenizer/vocab/en_32000/en_32000.vocab", encoding = "utf-8") as f:
            self.vocab_trg = f.read().splitlines()
        with open("./Tokenizer/vocab/de_32000/de_32000.vocab", encoding = "utf-8") as f:               
            self.vocab_src = f.read().splitlines()
                

    def __len__(self):

        return len(self.train_t)


    def tokenizer(self, model):
        SP = sp.SentencePieceProcessor()
        if not os.path.isfile("Tokenizer/vocab/"f"{model}/"f"{model}.model"):
            self.build_vocab()
        SP.Load(model_file = 
        "Tokenizer/vocab/"f"{model}/"f"{model}.model")
        return SP


    def tokenize(self, vocab, type):
        tokenized_trg = []
        tokenized_src = []
        trg_model = self.tokenizer(f"{self.trg_lang}_32000")
        src_model = self.tokenizer(f"{self.src_lang}_32000")
        EncodeAsIds_file = f'./Tokenizer/EncodeAsIds_{type}.pickle'
        if not os.path.isfile(EncodeAsIds_file):
            print("Encoding As Id...") 
            for src, trg in tqdm(vocab):       
                tok_tmp_trg = trg_model.EncodeAsIds(trg); tok_tmp_trg.insert(0,self.bos_idx); tok_tmp_trg.append(self.eos_idx) 
                tok_tmp_src = src_model.EncodeAsIds(src); tok_tmp_src.insert(0,self.bos_idx); tok_tmp_src.append(self.eos_idx)
                tokenized_trg.append(tok_tmp_trg)
                tokenized_src.append(tok_tmp_src)
            assert(len(tokenized_trg) == len(tokenized_src)), "Vocab size is different!!"
            tok_trg_src =[(en, de) for en, de in zip(self.to_tensor(tokenized_trg), self.to_tensor(tokenized_src))]
            save_pkl(tok_trg_src, EncodeAsIds_file)          
        
        else:                
            print(f"Loading Encoded {type} file!")
            tok_trg_src = load_pkl(EncodeAsIds_file)
        
        return tok_trg_src


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
            assert(len(temp_trg) == len(temp_src)), "Vocab size is different!!"
            trg_corpus += temp_trg
            src_corpus += temp_src
        print("Loaded Corpus! \nCorpus sizes:")
        print("target Corpus : ", len(trg_corpus))
        print("source Corpus : ", len(src_corpus))   

        return trg_corpus, src_corpus
    
            
    def build_vocab(self):
        # building Vocab
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
    
    # def translate(self, model, src_sentence: str, decode_func):
    #     model.eval()
    #     src = self.transform_src([self.tokenizer_src(src_sentence)]).view(1, -1)
    #     num_tokens = src.shape[1]
    #     tgt_tokens = decode_func(model,
    #                              src,
    #                              max_len=num_tokens+5,
    #                              start_symbol=self.sos_idx,
    #                              end_symbol=self.eos_idx).flatten().cpu().numpy()
    #     tgt_sentence = " ".join(self.vocab_tgt.lookup_tokens(tgt_tokens))
    #     return tgt_sentence