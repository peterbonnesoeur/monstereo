

import torch
import torch.nn as nn
import numbers
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from torch.nn import init

from torch.nn import  Module

from torch import Tensor, Size
from typing import Union, List

_shape_t = Union[int, List[int], Size]

class LayerNormMasked(Module):

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(LayerNormMasked, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor, mask = None) -> Tensor:
        raise ValueError
        if mask is None:
            return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            return F.layer_norm(input, self.normalized_shape, self.weight*mask, self.bias, self.eps)
    def extra_repr(self) -> Tensor:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_hid, d_inner_hid, dropout=0.1):

        super().__init__()
        self.w_1 = nn.Linear(d_hid, d_inner_hid)
        self.w_2 = nn.Linear(d_inner_hid ,d_hid)
        self.layer_norm = nn.LayerNorm(d_hid)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        output = self.w_1(x)
        output = nn.functional.relu(output)
        output = self.w_2(output)
        output = self.dropout(output)
        output = output + x
        return self.layer_norm(output)
