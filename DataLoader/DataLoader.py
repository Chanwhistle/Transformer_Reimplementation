#DataLoader

# Load StpTokenizer models
import argparse
import os
import torch
from torch.utils.data import Dataset
from DataLoader.CustomDataset import CustomDataset
from torch.utils.data import DataLoader

# set variables
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Please input variables')

    # Required parameter
    parser.add_argument(
        "--batch_size",
        default = 128,
        type = int,
        required = False,
    )    
    
    args = parser.parse_args()


# Tokenizer 

import sentencepiece as spm

# Tokenizer class

class Stp(spm.SentencePieceProcessor):
    def __init__(self):
        super().__init__()


def load_tokenizers():

    Stp_en = Stp.Load(model_file = 
                      "Tokenizer/models/tokenized_train.en_32000/tokenized_train.en_32000.model")
    
    Stp_de = Stp.Load(model_file = 
                      "Tokenizer/models/tokenized_train.de_32000/tokenized_train.de_32000.model")
    
    return Stp_en, Stp_de


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])



def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)
    
    print("Building German Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt

def load_vocab(spacy_de, spacy_en):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt



dataset = CustomDataset()


def my_collate_fn(dataset):
    en_text, de_text, = [], []
  
    for (_en,_de) in dataset:
        en_text.append(_en)
        de_text.append(_de)
  
    return torch.Tensor(en_text), torch.Tensor(de_text)


dataloader = DataLoader(
    dataset, 
    batch_size = 128,
    shuffle = False, 
    collate_fn = my_collate_fn
)