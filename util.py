#DataLoader

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data import Field, BucketIterator
import sentencepiece as sp
from tqdm import tqdm
import pickle

class IWSLTDataset(Dataset):
    def __init__(self, src_lang = "de", trg_lang = "en", type = "train"):
        super().__init__()

        if type == "train":
            trg, src = self.load_vocab(type, src_lang, trg_lang)
            self.trg_tok_list = self.tokenize(f"{trg_lang}_32000", trg)
            self.src_tok_list = self.tokenize(f"{src_lang}_32000", src)


        elif type == "tst":
            trg, src = self.load_vocab(type, src_lang, trg_lang)
            self.trg_tok_list = self.tokenize(f"{trg_lang}_32000", trg)
            self.src_tok_list = self.tokenize(f"{src_lang}_32000", src)


        elif type == "dev":      
            trg, src = self.load_vocab(type, src_lang, trg_lang)
            self.trg_tok_list = self.tokenize(f"{trg_lang}_32000", trg)
            self.src_tok_list = self.tokenize(f"{src_lang}_32000", src)

        else:
            raise TypeError ("Data type is wrong. Please check data type.")
        
        self.src = src
        

    def __len__(self):

        return len(self.src)


    def __getitem__(self,idx):
        en = torch.IntTensor(self.trg_tok_list[idx])
        de = torch.IntTensor(self.src_tok_list[idx])

        return en, de


    def tokenizer(self, model):
        SP = sp.SentencePieceProcessor()
        SP_temp = SP.Load(model_file =
        "Tokenizer/models/"f"{model}/"f"{model}.model")
        if SP_temp == True:
            pass
        
        else:
            self.build_vocab()
            SP_temp = SP.Load(model_file = 
            "Tokenizer/models/"f"{model}.model")
                  
        print("Loaded "f"{model[:2]}_Tokenizer")
        return SP


    def tokenize(self, model, vocab):
        tokenized = []      
        tokenizer = self.tokenizer(model)

        if os.path.isfile(f'./Tokenizer/EncodeAsIds_{model}.pickle') == False:
            print("Encoding As Id...") 
            for lines in tqdm(vocab):
                tokenized_temp = tokenizer.EncodeAsIds(lines)
                tokenized_temp.insert(0,4)        # insert [BOS]
                tokenized_temp.append(5)          # insert [EOS]
                tokenized.append(tokenized_temp)
            with open(f'./Tokenizer/EncodeAsIds_{model}.pickle', 'wb') as f:
                pickle.dump(tokenized, f, pickle.HIGHEST_PROTOCOL)
        
        else:                
            print(f"Loading Encoded {model[:2]} file!")
            with open(f'./Tokenizer/EncodeAsIds_{model}.pickle', 'rb') as f:
                tokenized = pickle.load(f)
        
        print(f"{model[:2]}_vocab tokenizing Finished!")
        return tokenized


    def load_vocab(self, type, src_lang, trg_lang):
        datapath = "./dataset/"
        trg_vocab, src_vocab = [], []
        vocab_dict = {"train": ["commoncrawl", "europarl-v7"],
                      "tst": ["IWSLT16.TED.tst2014", "IWSLT16.TED.tst2013", "IWSLT16.TED.tst2012", "IWSLT16.TED.tst2011", "IWSLT16.TED.tst2010"],
                      "dev": ["newstest2008", "newstest2009", "newstest2010", "newstest2011", "newstest2012", "newstest2013"]}
        for dataset in vocab_dict.get(type):
            trg_path = os.path.join(datapath, type, f"{dataset}.{src_lang}-{trg_lang}.{trg_lang}")
            src_path = os.path.join(datapath, type, f"{dataset}.{src_lang}-{trg_lang}.{src_lang}")
            with open(trg_path, encoding = "utf-8") as f:
                temp_trg = f.read().splitlines()
            with open(src_path, encoding = "utf-8") as f:
                temp_src = f.read().splitlines()
            assert(len(temp_trg) == len(temp_src)), "Vocab size is different!!"
            trg_vocab += temp_trg
            src_vocab += temp_src
        print("Loaded Vocab! \nVocab sizes:")
        print("target vocab : ", len(trg_vocab))
        print("Source vocab : ", len(src_vocab))   

        return trg_vocab, src_vocab


    def build_vocab(self):
        SP_trainer = sp.SentencePieceTrainer
        user_defined_symbols = '[PAD],[BOS],[EOS],[CLS],[SEP],[MASK],[BOS],[EOS]'
        user_defined_symbols += '[UNK0],[UNK1],[UNK2],[UNK3],[UNK4],[UNK5],[UNK6],[UNK7],[UNK8],[UNK9]'
        vocab_size = 32000
        save_dir = "./Tokenizer/models/"
        character_coverage = 0.9999
        model_type = 'bpe'
        input_argument = "--input=%s , --model_prefix=%s , --vocab_size=%s, --user_defined_symbols=%s, --model_type=%s, --character_coverage=%s"
        
        print("Building German Vocabulary...")
        dataset_dir_de = f"./dataset/train/de_train"
        model_prefix_de = f"de_{vocab_size}"
        model_path = os.path.join(save_dir, model_prefix_de)
        input_de = input_argument % (dataset_dir_de, model_prefix_de, vocab_size, user_defined_symbols, model_type, character_coverage)
        de_model = SP_trainer.Train(input_de)

        print("Building English Vocabulary...")
        dataset_dir_en = f"./dataset/train/en_train"
        model_prefix_en = f"en_{vocab_size}"
        model_path = os.path.join(save_dir, model_prefix_en)
        input_en = input_argument % (dataset_dir_en, model_prefix_en, vocab_size, user_defined_symbols, model_type, character_coverage)
        en_model = SP_trainer.Train(input_en)   
                
        print("Voacb Training Finished!")    




dataset = IWSLTDataset()
print("Dataset Loaded")

def my_collate_fn(samples):
    trg = [sample[0] for sample in samples]
    src = [sample[1] for sample in samples]
    padded_trg = pad_sequence(trg, batch_first=True, padding_value = int(3))  # Padding
    padded_src = pad_sequence(src, batch_first=True, padding_value = int(3))  # Padding
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