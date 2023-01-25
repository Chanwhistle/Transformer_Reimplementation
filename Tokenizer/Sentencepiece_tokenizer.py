import sentencepiece as spm
import os
import argparse


# set variables
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Please input variables')

    # Required parameter
    parser.add_argument(
        "--corpus_name",
        type=str,
        required=True,
    )    

    parser.add_argument(
        "--save_dir",
        default= './models/',
        type=str,
        required=False,
    )
    
    parser.add_argument(
        "--dataset_dir",
        default= './../dataset/',
        type=str,
        required=False,
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



# Tokenizer model
corpus_name = args.corpus_name
vocab_size = args.vocab_size
model_type = args.model_type
character_coverage = args.character_coverage
dataset_dir = args.dataset_dir
model_name = 'tokenized_%s' % (corpus_name) + '_' + str(vocab_size)
save_dir = args.save_dir + model_name

user_defined_symbols = '[PAD],[UNK],[CLS],[SEP],[MASK],[BOS],[EOS]'
user_defined_symbols += '[UNK0],[UNK1],[UNK2],[UNK3],[UNK4],[UNK5],[UNK6],[UNK7],[UNK8],[UNK9]'
unused_token_num = 100
unused_list = [f"<unused{i}>" for i in range(unused_token_num)]
user_defined_symbols += unused_list

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

model_path = os.path.join(save_dir, model_name)

input_argument = '--input=%s%s --model_prefix=%s --vocab_size=%s --user_defined_symbols=%s --model_type=%s --character_coverage=%s'
cmd = input_argument%(dataset_dir, corpus_name, model_path, vocab_size, user_defined_symbols, model_type, character_coverage)

# Train Tokenizer
trained_vocab = spm.SentencePieceTrainer.Train(cmd)

print('Tokenizing Finished!')