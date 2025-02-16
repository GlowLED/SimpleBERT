import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module import Encoder, TokenEmbedding, SegmentEmbedding, PostionalEncoding
from BERT.utils import attn_mask_generate, generate_segment_ids

class BERT(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, n_head, n_layer, dropout, max_len):
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.segment_emb = SegmentEmbedding(d_model)
        self.pos_enc = PostionalEncoding(d_model, max_len)
        
        self.encoder = Encoder(d_model, d_ff, n_head, dropout, n_layer)
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, pad_idx=0, cls_idx=1, sep_idx=2):
        '''
        Args:
            x: (batch_size, seq_len)
        Return:
            o: (batch_size, seq_len, d_model)
        '''
        
        attn_mask = attn_mask_generate(x)
        segment_ids = generate_segment_ids(x, sep_idx)
        
        x = self.token_emb(x) + self.segment_emb(segment_ids)
        x = self.pos_enc(x)
        
        x = self.encoder(x, attn_mask)
        
        o = self.linear(x)
        
        return o
        
