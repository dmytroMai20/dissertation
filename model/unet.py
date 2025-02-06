import torch
import torch.nn as nn

def t_embedding(time_steps, temb_dim):
    half_dim = temb_dim // 2
    exponent = torch.arange(half_dim, dtype=torch.float32, device=time_steps.device) / half_dim
    freqs = torch.exp(-torch.log(torch.tensor(10000.0, device=time_steps.device)) * exponent)
    
    # Compute sinusoidal embeddings
    args = time_steps[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    #  sin(pos/10000**2i) = exp(-log(10000)*exponent)
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim):
        super().__init__()
        self.norm1
        self.conv1
        self.norm2
        self.conv2
        self.time_emb_proj
        self.activation
        self.skip_connections