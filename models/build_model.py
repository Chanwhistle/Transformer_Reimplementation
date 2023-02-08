import torch
import torch.nn as nn
from .embedding.embedding import *
from .embedding.positional_encoding import *
from .layers.multi_head_attention_layer import *
from .layers.positionwise_feed_forward_layer import *
from .model.encoder_decoder import *
from .model.transformer import *

from models.layers.positionwise_feed_forward_layer import *

def build_model(src_vocab_size,
                tgt_vocab_size,
                device=torch.device("gpu"),
                max_len = 5000,
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
    trg_embed = TransformerEmbedding(
                                     token_embed = tgt_token_embed,
                                     pos_embed = copy(pos_embed))

    attention = MultiHeadAttention(
                                    d_model = d_model,
                                    n_head = n_head)
    
    position_ff = PositionwiseFeedForward(
                                            d_model = d_model,
                                            hidden = hidden_layer,
                                            drop_prob = drop_prob)
    
    norm = nn.LayerNorm(d_model, eps = norm_eps)
    
    encoder_layer = EncoderLayer(
                                 size = d_model,
                                 self_attn = copy(attention),
                                 feed_forward = copy(position_ff),
                                 drop_prob = drop_prob,
                                 n_layers = n_layer)
    decoder_layer = DecoderLayer(
                                 size = d_model,
                                 self_attn = copy(attention),
                                 cross_attn = copy(attention),
                                 feed_forward = copy(position_ff),
                                 drop_prob = drop_prob,
                                 n_layers = n_layer)

    encoder = Encoder(
                      layer = encoder_layer,
                      n_layers = n_layer)
    
    decoder = Decoder(
                      layer = decoder_layer,
                      n_layers = n_layer)
    
    generator = nn.Linear(d_model, tgt_vocab_size)

    model = Transformer(
                        src_embed = src_embed,
                        trg_embed = trg_embed,
                        encoder = encoder,
                        decoder = decoder,
                        generator = generator).to(device)
    model.device = device

    return model