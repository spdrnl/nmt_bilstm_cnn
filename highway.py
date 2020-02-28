#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn

class Highway(nn.Module):

    def __init__(self, H, size):
        super(Highway, self).__init__()
        self.H = H
        self.Wh = nn.Linear(size, size, bias=False)
        self.Wt = nn.Linear(size, size, bias=True)  # includes a bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Tx = torch.sigmoid(self.Wt(x))
        y = self.H(self.Wh(x)) * Tx + x * (1 - Tx)
        return y


### END YOUR CODE 

