# Sentencepiece Tokenizer

'''

$ python Sentencepiece_tokenizer \

    --corpus_name             file name of corpus data \

optional arguments:
    --save_dir                directory for trained file \
    --dataset_dir             directory for dataset file \
    --vocab_size              number of vocabulary size (8000, 16000, 32000) \
    --character_coverage      rate of character coverage. small dataset performs well at 1 [0,1] \
    --model_type              type of model (unigram, bpe, char, word) \

'''