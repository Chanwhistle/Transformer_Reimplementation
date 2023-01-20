# encoder

import sentencepiece as spm
import pandas as pd
import argparse

# set variables
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Please input variables')

    # Required parameter
    parser.add_argument(
        "--input",
        type=str,
        required=True,
    )    
    
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--vocab_size",
        default='32000',
        type=int,
        required=False,
    )

    parser.add_argument(
        "--character_coverage",
        default= 0.9999,
        type=int,
        required=False,
    )
        
    parser.add_argument(
        "--model_type",
        default='bpe',
        type=str,
        required=False,
    )

    
    args = parser.parse_args()

sp = spm.SentencePieceProcessor()

with open ("/home/chanhwi/workspace/Transformer_Reimplementation/Tokenizer/tokenized_sp_train_en/tokenized_%s.model" % (args.model_name), "r") as f:
    txt = f.read()

vocab_file = "/home/chanhwi/workspace/Transformer_Reimplementation/Tokenizer/tokenized_sp_train_en/tokenized_%s.model" % (args.model_name)
sp.load(txt)


encoded_corpus = sp.encode_as_pieces(txt)
print(encoded_corpus)