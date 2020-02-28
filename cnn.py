#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i

import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, max_word_length, kernel_size, in_channels, out_channels):
        super(CNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.max_pool = nn.MaxPool1d(max_word_length - kernel_size + 1, stride=1, padding=0, dilation=1,)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv(input)
        x = torch.nn.functional.relu(x)
        x = self.max_pool(x)
        out = x.squeeze(2)
        assert len(list(out.size()))==2, "Conv output axis does not equal 2."

        return out

### END YOUR CODE

