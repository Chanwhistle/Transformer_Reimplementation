'''
Transformer
'''

import torch.nn as nn
        
class Transformer(nn.Module):
    def __init__(self, src_embed, trg_embed, encoder, decoder, generator):
        super(Transformer, self).__init__()
        self.src_embed = src_embed   # src언어의 embeding
        self.trg_embed = trg_embed   # trg언어의 embeding
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        
        
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)


    def decode(self, trg, encoder_out, trg_mask, src_trg_mask):
        return self.decoder(self.trg_embed(trg), encoder_out, trg_mask, src_trg_mask)


    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        src_trg_mask = self.make_src_trg_mask(src, trg)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(trg, encoder_out, trg_mask, src_trg_mask)
        out = F.log_softmax(self.generator(decoder_out), dim=-1)
        return out, decoder_out


    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask


    def make_trg_mask(self, trg):
        pad_mask = self.make_pad_mask(trg, trg)
        seq_mask = self.make_subsequent_mask(trg, trg)
        mask = pad_mask & seq_mask
        return pad_mask & seq_mask
    
    
    def make_src_trg_mask(self, src, trg):
        pad_mask = self.make_pad_mask(trg, src)
        return pad_mask
   
    
    def make_pad_mask(self, query, key, pad_idx=3):
        # query: (n_batch, len_query)
        # key: (n_batch, len_key)
        len_query, len_key = query.size(1), key.size(1)

        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)
        key_mask = key_mask.repeat(1, 1, len_query, 1)    # (n_batch, 1, query_seq_len, key_seq_len)

        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, 1, len_key)  # (n_batch, 1, query_seq_len, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False
        return mask


    def make_subsequent_mask(self, query, key):
        len_query, len_key = query.size(1), key.size(1)

        mask = torch.tril(torch.ones(len_query, len_key)).type(torch.BoolTensor).to(self.device)
        return mask