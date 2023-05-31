import torch
import math
from torch import nn
from torch import Tensor

from math import sqrt
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PositionalEncoding(nn.Module):
    def __init__(self,
                 maxlen: int,
                 emb_size: int,
                 dropout: float = 0.1,):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(maxlen, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embedding: Tensor, src_mask):
        re = self.dropout(
            token_embedding + self.pos_embedding.weight[:token_embedding.size(1), :])
        return re


class AttentalSum(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.w = nn.Linear(input_dim, 1)
        self.act = nn.Tanh()
        self.soft = nn.Softmax(dim=0)

    def forward(self, x: Tensor, src_mask: Tensor = None):
        # x: S B D, src_mask: B S
        weight = self.w(x)
        weight = self.act(weight).clone()

        if src_mask is not None:
            weight[src_mask.transpose(0, 1)] = -torch.inf
        weight = self.soft(weight)

        weighted_embed = torch.sum(x*weight, dim=0)
        return weighted_embed


class PositionalEncoding_fix(nn.Module):
    "Implement the PE function."

    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1,):
        super(PositionalEncoding_fix, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, src_mask=None):
        x = x + self.pe[:, :x.size(1)]
        x[src_mask] = 0
        return self.dropout(x)


class MaskSum(nn.Module):
    def __init__(self, dim=1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x, src_mask):
        mask = torch.ones_like(x).masked_fill(src_mask.unsqueeze(-1), 0)
        return torch.sum(x*mask, dim=self.dim)


class MaskMean(nn.Module):
    def __init__(self, dim=1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x, src_mask):
        mask = torch.ones_like(x).masked_fill(src_mask.unsqueeze(-1), 0)
        summ = torch.sum(x*mask, dim=self.dim)
        # print(x.shape, summ.shape)
        # print(torch.sum(mask, dim=self.dim).shape)
        # exit()
        meann = summ/(torch.sum(mask, dim=self.dim) + 1e-7)
        return meann


class FragAttention(nn.Module):
    def __init__(self, pooling="sum", max_length=30):
        super().__init__()
        if pooling == 'sum':
            self.pooling = MaskSum(dim=2)
        # Ticky!, 0 in the mask means KEEP, 1 means DISCARD
        self.right_mask = torch.ones(
            max_length, max_length).triu().t().bool()[:-1]
        self.left_mask = torch.ones(max_length, max_length).triu().bool()[1:]

    def forward(self, x, src_mask):
        # x: S B D
        # src_mask: B S
        # left_mask, right_mask: G, S
        G = self.right_mask.shape[0]
        S, B, D = x.shape

        if self.right_mask.device != x.device:
            self.right_mask = self.right_mask.to(x.device)
            self.left_mask = self.left_mask.to(x.device)

        x = x.transpose(0, 1)
        new_dim_x = x.unsqueeze(1).expand(B, G, S, D)
        new_dim_mask = src_mask.unsqueeze(1).expand(B, G, S)

        # Back to normal
        left_tower_mask = new_dim_mask | self.left_mask
        right_tower_mask = new_dim_mask | self.right_mask

        left_frag = self.pooling(new_dim_x, left_tower_mask)
        right_frag = self.pooling(new_dim_x, right_tower_mask)

        new_frag = torch.cat([left_frag, right_frag], dim=2)
        return new_frag


class Deepmatch(nn.Module):
    def __init__(self, max_length=30):
        super().__init__()

    def forward(self, x, src_mask):
        S, B, D = x.shape

        left_tower = x[:S-1]
        right_tower = x[1:]
        x_new = torch.cat([left_tower, right_tower], dim=2)
        return x_new.transpose(0, 1)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class ExcludeCLS(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        cls_token, x = x[:, :1], x[:, 1:]
        x = self.fn(x, **kwargs)
        return torch.cat((cls_token, x), dim=1)

# prenorm


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# feed forward related classes


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride=1, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size,
                      padding=padding, groups=dim_in, stride=stride, bias=bias),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.Hardswish(),
            DepthWiseConv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        h = w = int(sqrt(x.shape[-2]))
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.net(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

# attention


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads,
                         dim_head=dim_head, dropout=dropout))),
                ExcludeCLS(
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x
