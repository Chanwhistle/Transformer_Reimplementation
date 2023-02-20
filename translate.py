import torch
from torch import nn
from Dataloader import *

DEVICE = torch.device('cuda:3')

model = torch.load("./checkpoint/0019.pt", map_location=DEVICE)


outputs = model(input)
print(outputs)



def translate(self, model, src_sentence: str, decode_func):
    model = torch.load("./checkpoint/0019.pt", map_location=DEVICE)
    model.eval()
    src = self.transform_src([self.tokenizer_src(src_sentence)]).view(1, -1)
    num_tokens = src.shape[1]
    trg_tokens = decode_func(model,
                                src,
                                max_len=num_tokens+5,
                                start_symbol=self.sos_idx,
                                end_symbol=self.eos_idx).flatten().cpu().numpy()
    trg_sentence = " ".join(self.vocab_trg.lookup_tokens(trg_tokens))
    return trg_sentence