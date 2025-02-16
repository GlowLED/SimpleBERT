import torch
import torch.nn as nn

class Embedding(nn.Module):
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
        seq_len = x.size(1)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device)        
        pos_emb = self.pos_emb(pos).unsqueeze(0)    #(1, seq_len, d_model)
        o = x + pos_emb     # pos_emb will be broadcasted to (batch_size, seq_len, d_model).
        return o


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
        
        y = self.attn(x, x, x, attn_mask)[0]
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



    
