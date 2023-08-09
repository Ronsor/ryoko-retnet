# Copyright (C) 2023 Ronsor Labs.
# This software is licensed to you under the terms of the Free Development Public License 1.0-US.

import torch
from torch import nn, Tensor
from torch.nn import functional as F

class XPos(nn.Module):
  def __init__(self, dim: int, theta: int = 10000, scale_base: int = 512):
    super().__init__()

    self.dim = dim
    self.scale_base = scale_base
    self.theta = theta

    self.register_buffer('scale', (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim))

  def forward(self, x: Tensor, offset: int = 0, inverse_scale: bool = False) -> Tensor:
    length, dim = x.size(-2), x.size(-1)
    assert dim <= self.dim

    scale = self.scale ** (torch.arange(offset, offset + length) / self.scale_base)[:, None]
    scale_length, scale_dim = scale.size()

    freqs    = 1. / (self.theta ** (torch.arange(scale_dim) / scale_dim))
    sin_in   = torch.einsum('i, j -> i j', torch.arange(offset, offset + scale_length), freqs)
    sin, cos = torch.sin(sin_in), torch.cos(sin_in)

    scale = scale[-length:]
    sin   = sin[-length:]
    cos   = cos[-length:]

    if inverse_scale: scale = 1 / scale

    sin = XPos.duplicate_interleave(sin * scale)
    cos = XPos.duplicate_interleave(cos * scale)

    y_1 = x * cos
    y_2 = XPos.rotate_half(x) * sin
    y = y_1 + y_2

    return y

  @staticmethod
  def rotate_half(x: Tensor) -> Tensor:
    return torch.stack((-x[:, :, 1::2], x[:, :, ::2]), dim=-1).flatten(-2)

  @staticmethod
  def duplicate_interleave(x: Tensor) -> Tensor:
    dim0 = x.size(0)
    return x.view(-1, 1).repeat(1, 2).view(dim0, -1)

class Retention(nn.Module):
  def __init__(self, embed_dim: int, head_dim: int = None,
               gamma: float = 1.0, kdim: int = None, vdim: int = None, head_vdim: int = None,
               layer_norm: bool = False, add_bias_kv: bool = False):
    super().__init__()

    self.embed_dim = embed_dim
    self.head_dim  = head_dim if head_dim is not None else embed_dim

    self.kdim      = kdim if kdim is not None else embed_dim
    self.vdim      = vdim if vdim is not None else embed_dim
    self.head_vdim = head_vdim if head_vdim is not None else self.vdim

    self.gamma = gamma

    self.query = nn.Linear(embed_dim, self.head_dim, bias=False)
    self.key   = nn.Linear(self.kdim, self.head_dim, bias=add_bias_kv)
    self.value = nn.Linear(self.vdim, self.head_vdim, bias=add_bias_kv)

    self.xpos = XPos(self.head_dim)

    self.layer_norm = nn.LayerNorm() if layer_norm else nn.Identity()

  def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None,
              offset: int = 0, state: Tensor = None, need_state: bool = False) -> Tensor:
    length_is_one = query.size(-2) == 1

    query = self.query(query)
    key   = self.key(key)
    value = self.value(value)

    query, key = self.xpos(query, offset), self.xpos(key, offset, True)

    if mask is None and not (need_state and length_is_one):
      mask = Retention.decay_mask(query.size(-2), self.gamma)

    if need_state: # Recurrent or Chunkwise Retention
      if state is None:
        state = self.empty_state.repeat(query.size(-3), 1, 1)

      if length_is_one and mask is None: # Recurrent Retention
        state  = self.gamma * state + (key.transpose(-1, -2) @ value)
        output = query @ state
      else: # Chunkwise Retention
        inner_retention = (query @ key.transpose(-1, -2)) * mask.unsqueeze(0)
        inner_retention = inner_retention @ value

        power = (self.gamma ** torch.arange(1, query.size(-2) + 1)).view(1, query.size(-2), 1).repeat(query.size(-3), 1, 1)
        cross_retention = (query @ state) * power

        state  = (self.gamma ** query.size(-2)) * state + (key.transpose(-1, -2) @ (value * mask[-1].view(1, -1, 1)))
        output = inner_retention + cross_retention

    else: # Parallel Retention
      retention = (query @ key.transpose(-1, -2)) * mask.unsqueeze(0)
      output = retention @ value

    output = self.layer_norm(output)

    return (output, state) if need_state else output

  @property
  def empty_state(self) -> Tensor:
    return torch.zeros(1, self.head_dim, self.head_vdim)

  @staticmethod
  def decay_mask(length, gamma) -> Tensor:
    u, v = torch.arange(length).view(-1, 1), torch.arange(length).view(1, -1)
    w = (gamma ** (u - v)) * (u >= v)
    return torch.nan_to_num(w)

class MultiscaleRetention(nn.Module):
  def __init__(self, embed_dim: int, num_heads: int,
               gammas: list = None, kdim: int = None, vdim: int = None,
               batch_first: bool = True):
    super().__init__()
    assert batch_first, "Only batch_first=True is supported"

    self.embed_dim = embed_dim
    self.kdim = kdim if kdim is not None else embed_dim
    self.vdim = vdim if vdim is not None else embed_dim

    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
    assert self.kdim % num_heads == 0, "kdim must be divisible by num_heads"
    assert self.vdim % num_heads == 0, "vdim must be divisible by num_heads"

    self.num_heads = num_heads
    self.head_dim  = embed_dim // num_heads

    self.gammas = gammas if gammas is not None else (1 - 2 ** (-5. - torch.arange(0, num_heads))).tolist()

    self.heads = nn.ModuleList([
      Retention(embed_dim, self.head_dim, gamma,
                kdim=self.kdim, vdim=self.vdim,
                head_vdim=self.vdim // num_heads) for gamma in self.gammas
    ])

    self.group_norm = nn.GroupNorm(num_heads, self.vdim)

    self.group  = nn.Linear(embed_dim, self.vdim, bias=False)
    self.output = nn.Linear(self.vdim, embed_dim, bias=False)

  def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None,
              offset: int = 0, state: list = None, need_state: bool = False,
              _always_return_state: bool = False):
    if need_state or state is not None:
      if state is None: state = []
      output, state = tuple(zip(*[
        head(query, key, value, mask, offset=offset,
             state=state[i] if i < len(state) else None, need_state=True
            ) for i, head in enumerate(self.heads)
      ]))
      output, state = list(output), list(state)
    else:
      output = [head(query, key, value, mask, offset=offset) for head in self.heads]

    output = torch.cat(output, dim=-1)
    shape  = output.size()
    output = self.group_norm(output.view(-1, self.vdim)).view(shape)
    group  = F.silu(self.group(query))
    output = self.output(group * output)

    return (output, state) if need_state or _always_return_state else output

def try_me():
  x = torch.randn(1, 3, 8)
  m = MultiscaleRetention(8, 2)
  y1 = m(x, x, x)
  print(y1)

  print('---')

  xx = x.transpose(0, 1)[0].unsqueeze(0)
  y2, z2 = m.forward(xx, xx, xx, need_state=True)
  print(y2)
  xx = x.transpose(0, 1)[1].unsqueeze(0)
  y3, z3 = m.forward(xx, xx, xx, offset=1, state=z2, need_state=True)
  print(y3)
  xx = x.transpose(0, 1)[2].unsqueeze(0)
  y4, z4 = m.forward(xx, xx, xx, offset=2, state=z3, need_state=True)
  print(y4)

  print('---')

  xx = torch.stack([x.transpose(0, 1)[0], x.transpose(0, 1)[1]]).transpose(0, 1)
  y5, z5 = m.forward(xx, xx, xx, offset=0, need_state=True)
  print(y5)

  xx = x.transpose(0, 1)[2].unsqueeze(0)
  y6, z6 = m.forward(xx, xx, xx, offset=2, state=z5, need_state=True)
  print(y6)

if __name__ == '__main__':
  try_me()
