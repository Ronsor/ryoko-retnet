# Copyright (C) 2023 Ronsor Labs.
# This software is licensed to you under the terms of the Free Development Public License 1.0-US.

import dataclasses, torch
from torch import nn, Tensor
from torch.nn import functional as F

from .retention import MultiscaleRetention

def _get_activation_fn(x, default: str = 'gelu'):
  _activation_fn = {
    'relu': F.relu,
    'gelu': F.gelu,
    'silu': F.silu,
  }
  return _activation_fn.get(x if x is not None else default, x)

@dataclasses.dataclass
class ModelArgs:
  dim: int        = 24
  vocab_size: int = 8
  num_layers: int = 6
  num_heads: int  = 6

  dim_ffn: int    = None
  norm_eps: float = 1e-6
  dropout: float  = 0.0

  layer_args: dict = None

class FunctionToModule(nn.Module):
  def __init__(self, f):
    super().__init__()
    self.f = f

  def forward(self, *args):
    return self.f(*args)

class RetNetDecoderLayer(nn.Module):
  def __init__(self, d_model: int, nhead: int, dim_feedforward: int = None,
               dropout: float = 0.0, activation: any = 'gelu', layer_norm_eps: float = 1e-6,
               batch_first: bool = True, norm_first: bool = True, only_self_rttn: bool = True,
               feed_forward: nn.Module = None):
    super().__init__()

    self._all_args = dict(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                          dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps,
                          norm_first=norm_first, only_self_rttn=only_self_rttn)

    if dim_feedforward is None:
      dim_feedforward = d_model * 4

    assert norm_first
    self.norm_first = norm_first
    self.only_self_rttn = only_self_rttn

    self.norm = nn.ModuleList([
      nn.LayerNorm(d_model, eps=layer_norm_eps) for _ in range(2 if only_self_rttn else 3)
    ])

    self.dropout = nn.ModuleList([
      nn.Dropout(dropout) for _ in range(2 if only_self_rttn else 3)
    ])

    self.self_retention = MultiscaleRetention(d_model, nhead, batch_first=batch_first)
    if not only_self_rttn:
      self.cross_retention = MultiscaleRetention(d_model, nhead, batch_first=batch_first)

    self.feed_forward = feed_forward if feed_forward is not None else nn.Sequential(
      nn.Linear(d_model, dim_feedforward),
      FunctionToModule(_get_activation_fn(activation)),
      nn.Linear(dim_feedforward, d_model)
    )

  def forward(self, tgt: Tensor, memory: Tensor = None, tgt_mask: Tensor = None, memory_mask: Tensor = None,
              offset: int = 0, state: list = None, need_state: bool = False) -> Tensor:
    x = tgt

    if state is None:
      tgt_state, mem_state = (None, None)
    elif self.only_self_rttn:
      tgt_state, mem_state = (state, None)
    else:
      tgt_state, mem_state = tuple(state)

    if self.norm_first:
      y, tgt_state = self._sr_block(self.norm[0](x), tgt_mask, offset, tgt_state, need_state)
      x = x + y
      if not self.only_self_rttn:
        y, mem_state = self._xr_block(self.norm[1](x), memory, memory_mask, offset, mem_state, need_state)
        x = x + y
      x = x + self._ff_block(self.norm[-1](x))
    else:
      y, tgt_state = self._sr_block(x, tgt_mask, offset, tgt_state, need_state)
      x = self.norm[0](x + y)
      if not self.only_self_rttn:
        y, mem_state = self._xr_block(x, memory, memory_mask, offset, mem_state, need_state)
        x = self.norm[1](x + y)
      x = self.norm[-1](x + self._ff_block(x))

    if self.only_self_rttn:
      state = tgt_state
    else:
      state = (tgt_state, mem_state)

    return (x, state) if need_state else x

  def _sr_block(self, x: Tensor, mask: Tensor = None,
                offset: int = 0, state: list = None, need_state: bool = False) -> (Tensor, Tensor):
    x, state = self.self_retention(x, x, x, mask,
                                   offset=offset, state=state, need_state=need_state, _always_return_state=True)
    return self.dropout[0](x), state

  def _xr_block(self, x: Tensor, memory: Tensor, mask: Tensor = None,
                offset: int = 0, state: list = None, need_state: bool = False) -> (Tensor, Tensor):
    x, state = self.cross_retention(x, memory, memory, mask,
                                    offset=offset, state=state, need_state=need_state, _always_return_state=True)
    return self.dropout[1](x), state

  def _ff_block(self, x: Tensor) -> Tensor:
    return self.dropout[-1](self.feed_forward(x))

class RetNetDecoder(nn.Module):
  def __init__(self, decoder_layer: RetNetDecoderLayer, num_layers: int, norm: any = None):
    super().__init__()
    decoder_layer_args = getattr(decoder_layer, '_all_args', decoder_layer)

    self.layers = nn.ModuleList([
      RetNetDecoderLayer(**decoder_layer_args) for _ in range(num_layers)
    ])
    self.num_layers = num_layers

    self.norm = norm

  def forward(self, tgt: Tensor, memory: Tensor = None,
              tgt_mask: Tensor = None, memory_mask: Tensor = None,
              offset: int = 0, state: list = None, need_state: bool = False) -> Tensor:
    output = tgt

    if need_state or state is not None:
      if state is None:
        state = [None] * self.num_layers

      for i, mod in enumerate(self.layers):
        output, state[i] = mod(output, memory, tgt_mask, memory_mask,
                               offset=offset, state=state[i], need_state=True)
    else:
      for mod in self.layers:
        output = mod(output, memory, tgt_mask, memory_mask,
                     offset=offset)

    if self.norm is not None:
      output = self.norm(output)

    return (output, state) if need_state else output

class CausalRetNet(nn.Module):
  def __init__(self, params: ModelArgs):
    super().__init__()
    self.params = params

    self.embedding = nn.Embedding(params.vocab_size, params.dim)
    self.decoder = RetNetDecoder(
      RetNetDecoderLayer(params.dim, params.num_heads, params.dim_ffn,
                         dropout=params.dropout, layer_norm_eps=params.norm_eps,
                         only_self_rttn=True, activation='gelu',
                         **(params.layer_args if params.layer_args is not None else {})),
      params.num_layers, norm=nn.LayerNorm(params.dim, eps=params.norm_eps))

    self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

  def forward(self, x: Tensor, offset: int = 0, state: list = None,
              need_state: bool = False, want_logits: bool = True):
    embed = self.embedding(x)

    if need_state:
      embed, state = self.decoder(embed, offset=offset, state=state, need_state=True)
    else:
      embed = self.decoder(embed, offset=offset, state=state)

    output = self.output(embed) if want_logits else embed
    output = output.float()

    return (output, state) if need_state else output

def try_me():
  rn = CausalRetNet(ModelArgs())

  sample = torch.tensor([
    [1, 2, 3, 4, 5, 6],
  ])
  print(rn(sample))

  sample_a = torch.tensor([[1, 2, 3]])
  sample_b = torch.tensor([[4, 5]])
  sample_c = torch.tensor([[6]])

  y, s = rn(sample_a, need_state=True)
  print(y)

  y, s = rn(sample_b, offset=3, state=s, need_state=True)
  print(y)

  y, s = rn(sample_c, offset=5, state=s, need_state=True)
  print(y)

if __name__ == '__main__':
  try_me()
