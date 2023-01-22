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
user_defined_symbols = '[PAD],[UNK],[CLS],[SEP],[MASK],[BOS],[EOS],[UNK0],[UNK1],[UNK2],[UNK3],[UNK4],[UNK5],[UNK6],[UNK7],[UNK8],[UNK9],[unused0],[unused1],[unused2],[unused3],[unused4],[unused5],[unused6],[unused7],[unused8],[unused9],[unused10],[unused11],[unused12],[unused13],[unused14],[unused15],[unused16],[unused17],[unused18],[unused19],[unused20],[unused21],[unused22],[unused23],[unused24],[unused25],[unused26],[unused27],[unused28],[unused29],[unused30],[unused31],[unused32],[unused33],[unused34],[unused35],[unused36],[unused37],[unused38],[unused39],[unused40],[unused41],[unused42],[unused43],[unused44],[unused45],[unused46],[unused47],[unused48],[unused49],[unused50],[unused51],[unused52],[unused53],[unused54],[unused55],[unused56],[unused57],[unused58],[unused59],[unused60],[unused61],[unused62],[unused63],[unused64],[unused65],[unused66],[unused67],[unused68],[unused69],[unused70],[unused71],[unused72],[unused73],[unused74],[unused75],[unused76],[unused77],[unused78],[unused79],[unused80],[unused81],[unused82],[unused83],[unused84],[unused85],[unused86],[unused87],[unused88],[unused89],[unused90],[unused91],[unused92],[unused93],[unused94],[unused95],[unused96],[unused97],[unused98],[unused99]'

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

model_path = os.path.join(save_dir, model_name)


input_argument = '--input=%s%s --model_prefix=%s --vocab_size=%s --user_defined_symbols=%s --model_type=%s --character_coverage=%s'
cmd = input_argument%(dataset_dir, corpus_name, model_path, vocab_size, user_defined_symbols, model_type, character_coverage)

# Train Tokenizer
trained_vocab = spm.SentencePieceTrainer.Train(cmd)

print('Train Finished!')