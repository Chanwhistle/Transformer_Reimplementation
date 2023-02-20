import pickle
import torch
from torchtext.data.metrics import bleu_score
import sentencepiece as sp

def save_pkl(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_pkl(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def get_bleu_score(output, gt, specials, max_n=4):
    spp = sp.SentencePieceProcessor()
    vocab_file = "./Tokenizer/vocab/en_32000/en_32000.model"
    spp.load(vocab_file)
    
    def itos(x):                 
        xs = x.tolist()
        indexs = [x for x in xs if x not in specials]
        tokens = spp.DecodeIdsWithCheck(indexs)
        return tokens
    
    pred = torch.stack([out.max(dim=-1)[1] for out in output], dim=0)
    pred_str = itos(pred)
    gt_str = itos(gt)
    
    score = bleu_score(pred_str, gt_str, max_n=max_n) * 100
    return  score


def greedy_decode(model, src, max_len, start_symbol, end_symbol):
    src = src.to(model.device)
    src_mask = model.make_src_mask(src).to(model.device)
    memory = model.encode(src, src_mask)

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(model.device)
    for i in range(max_len-1):
        memory = memory.to(model.device)
        trg_mask = model.make_trg_mask(ys).to(model.device)
        src_trg_mask = model.make_src_trg_mask(src, ys).to(model.device)
        out = model.decode(ys, memory, trg_mask, src_trg_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == end_symbol:
            break
    return 