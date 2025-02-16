import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module import Encoder, Embedding, PostionalEncoding
from mask_generate import mask_generate

class BERT(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, n_head, n_layer, dropout, max_len):
        self.emb = Embedding(vocab_size, d_model)
        self.pos_enc = PostionalEncoding(d_model, max_len)
        self.encoder = Encoder(d_model, d_ff, n_head, dropout, n_layer)

    def forward(self, x):
        '''
        Args:
            x: (batch_size, seq_len)
        Return:
            o: (batch_size, seq_len, d_model)
        '''
        attn_mask = mask_generate(x)
        x = self.emb(x)
        x = self.pos_enc(x)
        o = self.encoder(x, attn_mask)
        return x
