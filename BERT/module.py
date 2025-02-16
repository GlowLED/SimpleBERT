import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_emb):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_emb)
    
    def forward(self, x):
        '''
        Args:
            x: (batch_size, seq_len)
        Return:
            o: (batch_size, seq_len, d_model)
        '''
        o = self.embedding(x)
        return o


class PostionalEncoding(nn.Module):
    '''
    Postional Encoding for BERT. It's a learnable embedding.
    Combine Sequence encoding.
    '''
    def __init__(self, d, max_len):
        super().__init__()
        self.d_model = d
        self.max_len = max_len
        
        self.pos_emb = nn.Embedding(max_len, d)
    
    def forward(self, x):
        '''
        Args:
            x: (batch_size, seq_len, d)
        Return:
            o: (batch_size, seq_len, d)
        '''
        batch_size, seq_len = x.size(0), x.size(1)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device)        
        emb = torch.repeat_interleave(self.pos_emb(pos).unsqueeze(0), batch_size, dim=0)
        o = x + emb
        return o
        

class SegmentEmbedding(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.emb = nn.Embedding(2, d)
    
    def forward(self, segment_ids):
        '''
        Args:
            x: (batch_size, seq_len, d)
            segment_ids: (batch_size, seq_len)
        Return:
            o: (batch_size, seq_len, d)
        '''
        return self.emb(segment_ids)
        

class AddNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
    
    def forward(self, x, y):
        '''
        Args:
            x: (batch_size, seq_len, d)
            y: (batch_size, seq_len, d)
        Return:
            o: (batch_size, seq_len, d)
        '''
        o = self.norm(x + y)
        return o


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.addnorm1 = AddNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.addnorm2 = AddNorm(d_model)
    
    def forward(self, x, attn_mask):
        '''
        Args:
            x: (batch_size, seq_len, d_model)
            attn_mask: (batch_size, seq_len, seq_len)
        Return:
            o: (batch_size, seq_len, d_model)
        '''
        
        y, _ = self.attn(x, x, x, attn_mask)
        x = self.addnorm1(x, y)
        y = self.ffn(x)
        o = self.addnorm2(x, y)
        return o

class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout, n_layer):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, d_ff, n_head, dropout)
            for _ in range(n_layer)
        ])
    
    def forward(self, x, attn_mask):
        '''
        Args:
            x: (batch_size, seq_len, d_model)
            attn_mask: (batch_size, seq_len, seq_len)
        Return:
            o: (batch_size, seq_len, d_model)
        '''
        for layer in self.layers:
            x = layer(x, attn_mask)
        o = x
        return o



    
