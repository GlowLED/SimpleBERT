import torch

def attn_mask_generate(x: torch.Tensor, pad_idx=0):
    '''
    Args:
        x: [batch_size, seq_len]
    
    Returns:
        mask: [batch_size, seq_len, seq_len]
    '''
    
    # example:
    # False False False True
    # False False False True
    # False False False True
    
    mask = torch.repeat_interleave(x.eq(pad_idx).unsqueeze(1), x.size(1), dim=1)
    return mask

def generate_segment_ids(x, sep_idx=2):
    '''
    Args:
        x: [batch_size, seq_len]
        sep_idx: int
    Returns:
        segment_ids: [batch_size, seq_len]
    '''
    batch_size, seq_len = x.size(0), x.size(1)
    
    idx = torch.where(x == sep_idx, 1, 0).nonzero()[0::2, 1, None]
    mask = torch.arange(seq_len, device=x.device).unsqueeze(0) > idx
    
    segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=x.device)
    segment_ids = segment_ids.masked_fill(mask, 1)
    
    return segment_ids

