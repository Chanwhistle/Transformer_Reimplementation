import torch
import torch.nn as nn

from model import *

def build_model(src_vocab_size,
                tgt_vocab_size,
                device=torch.device("gpu"),
                max_len = 256,
                n_layer = 6,
                d_model = 512,
                n_head = 8,
                hidden_layer = 2048,
                drop_prob = 0.1,
                norm_eps = 1e-12):
    import copy
    copy = copy.deepcopy

    src_token_embed = TokenEmbedding(
                                     d_model = d_model,
                                     vocab_size = src_vocab_size)
    tgt_token_embed = TokenEmbedding(
                                     d_model = d_model,
                                     vocab_size = tgt_vocab_size)
    pos_embed = PositionalEncoding(
                                   d_model = d_model,
                                   max_len = max_len,
                                   device = device)

    src_embed = TransformerEmbedding(
                                     token_embed = src_token_embed,
                                     pos_embed = copy(pos_embed))
    tgt_embed = TransformerEmbedding(
                                     token_embed = tgt_token_embed,
                                     pos_embed = copy(pos_embed))

    attention = MultiHeadAttention(
                                    d_model = d_model,
                                    n_head = n_head)
    
    position_ff = PositionwiseFeedForward(
                                            d_model = d_model,
                                            hidden = hidden_layer,
                                            drop_prob = drop_prob)
    norm = LayerNorm(d_model, eps = norm_eps)

    encoder_layer = EncoderLayer(
                                 size = d,
                                 self_attn = copy(attention),
                                 feed_forward = copy(position_ff),
                                 drop_prob = drop_prob)
    decoder_layer = DecoderLayer(
                                 size = d,
                                 self_attn = copy(attention),
                                 cross_attn = copy(attention),
                                 feed_forward = copy(position_ff),
                                 norm = copy(norm),
                                 drop_prob = drop_prob)

    encoder = Encoder(
                      layer = encoder_layer,
                      n_layers = n_layer)
    
    decoder = Decoder(
                      layer = decoder_layer,
                      n_layers = n_layer)
    
    generator = nn.Linear(d_model, tgt_vocab_size)

    model = Transformer(
                        src_embed = src_embed,
                        tgt_embed = tgt_embed,
                        encoder = encoder,
                        decoder = decoder,
                        generator = generator).to(device)
    model.device = device

    return model