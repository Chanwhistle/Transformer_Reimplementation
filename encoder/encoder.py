# encoder

import sentencepiece as spm
import os
import argparse

# set variables
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Please input variables')

    # Required parameter 
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
       
    args = parser.parse_args()

file_name = args.model_name
model_name = 'tokenized_%s' % (file_name) + '_' + str(args.vocab_size)

spp = spm.SentencePieceProcessor()
spp.load(model_file = "./../Tokenizer/models/%s/%s.model" % (model_name, model_name))

with open ("./../dataset/%s" % (file_name), "r", encoding="utf-8") as f:
    txt = f.read()

ids = spp.EncodeAsIds(txt)