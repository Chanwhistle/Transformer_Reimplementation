#DataLoader

# Load StpTokenizer models
import os
import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as sp


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


class IWSLTDataset(Dataset):
    def __init__(self, src_lang = "de", trg_lang = "en", type = "train"):
        super().__init__()
         
        if type == "train":
            trg, src = self.load_vocab(type, src_lang, trg_lang)
            self.trg_tok_list = self.tokenize(f"{trg_lang}_32000", trg)
            self.src_tok_list = self.tokenize(f"{src_lang}_32000", src)
                      
                       
        elif type == "tst":
            trg, src = self.load_vocab(type, src_lang, trg_lang)
            self.trg_tok_list = self.tokenize("en_32000", trg)
            self.src_tok_list = self.tokenize("de_32000", src)
        
        
        elif type == "dev":      
            trg, src = self.load_vocab(type, src_lang, trg_lang)
            self.trg_tok_list = self.tokenize("en_32000", self.tokenizer(model="en_32000"), trg)
            self.src_tok_list = self.tokenize("de_32000", self.tokenizer(model="de_32000"), src)
        
        else:
            raise TypeError ("Data type is wrong. Please check data type.")
        
        self.src = src
        self.trg = trg
                
                
    def __len__(self):
        return len(self.src)


    def __getitem__(self,idx):
        en = torch.FloatTensor(self.trg_tok_list[idx])
        de = torch.FloatTensor(self.src_tok_list[idx])
        return en, de
    
    
    def tokenizer(self, model):
        SP = sp.SentencePieceProcessor()
        SP_temp = SP.Load(model_file = 
        "Tokenizer/models/tokenized_IWSLT_2016."f"{model}.model")
        return SP


    def tokenize(self, model, vocab):
        tokenized = []      
        tokenizer = self.tokenizer(model)
        for lines in vocab:
            tokenized.append(tokenizer.EncodeAsIds(lines))
        
        max_len = max(len(item) for item in tokenized)
                
        for line in tokenized:
            while len(line) < max_len:
                line.append(3)
        return tokenized
    
    
    def load_vocab(self, type, src_lang, trg_lang):
        datapath = "./dataset/"
        trg_vocab = []
        src_vocab = []
        vocab_dict = {"train": ["IWSLT16.TED.train2014"],
                      "tst": ["IWSLT16.TED.tst2014", "IWSLT16.TED.tst2013", "IWSLT16.TED.tst2012", "IWSLT16.TED.tst2011", "IWSLT16.TED.tst2010"],
                      "dev": ["IWSLT16.TED.dev2012", "IWSLT16.TED.dev2010"]}
        for dataset in vocab_dict.get(type):
            trg_path = os.path.join(datapath, type, f"{dataset}.{src_lang}-{trg_lang}.{trg_lang}")
            src_path = os.path.join(datapath, type, f"{dataset}.{src_lang}-{trg_lang}.{src_lang}")
            with open(trg_path, encoding = "utf-8") as f:
                temp_trg = f.read().splitlines()
            with open(src_path, encoding = "utf-8") as f:
                temp_src = f.read().splitlines()
            assert(len(trg_vocab) == len(src_vocab))
                
            print("\nLoaded Vocab! \nVocab sizes:")
            trg_vocab += temp_trg
            print("target vocab : ", len(trg_vocab))
            src_vocab += temp_src
            print("Source vocab : ", len(src_vocab))    
        return trg_vocab, src_vocab
    
    def build_vocab(self):
        SP_trainer = sp.SentencePieceTrainer
        user_defined_symbols = ["<PAD>", "<UNK>", "<EOS>", "<BOS>", "<CLS>", "<SEP>", "<MASK>"]
        vocab_size = 32000
        model_prefix = f"{}"
        trg_lang = "en"
        src_lang = "de"
        input = 
        
        print("Building German Vocabulary...")
        de_model = SP_trainer.Train("--input = , --model_prefix = , --vocab_size = , --user_defined_symbols = , --model_type = , --character_coverage = ")
        
        print("Building English Vocabulary...")
        en_model = SP_trainer.Train("--input = , --model_prefix = , --vocab_size = , --user_defined_symbols = , --model_type = , --character_coverage = ")
        
        
        print("Voacb Training Finished!")

    
dataset = IWSLTDataset(type="tst")

print("Dataset Loaded")

dataloader = DataLoader(
    dataset,
    batch_size = 32, # argparse 사용 예정
    shuffle = True, 
)

for i, batch in enumerate(dataloader): # batch[0]에는 en_list가 들어가고 batch[1]에는 de_list가 들어감
    print(f"{i+1}번째 en_batch", batch[0].int())
    print(f"{i+1}번째 de_batch", batch[1].int())
    break