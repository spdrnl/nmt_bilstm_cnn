import unittest
import torch
from highway import Highway
import numpy as np
import torch.nn as nn


class TestHighway(unittest.TestCase):
    def test_output_size(self):
        x = torch.zeros(3, 5)
        h = Highway(torch.nn.Identity(), 5)
        y = h(x)
        self.assertEqual(x.size(), y.size())

    def test_input_too_small(self):
        x = torch.zeros(3, 4)
        h = Highway(torch.nn.Identity(), 5)
        with self.assertRaises(RuntimeError):
            y = h(x)

    def test_input_too_big(self):
        x = torch.zeros(3, 6)
        h = Highway(torch.nn.Identity(), 5)
        with self.assertRaises(RuntimeError):
            y = h(x)

    def test_zero_values(self):
        x = torch.zeros(3, 5)
        hw = Highway(torch.nn.Identity(), 5)
        y = hw(x)
        # hw = x = 0
        self.assertTrue(np.array_equal(x.numpy(), y.detach().numpy()))

    def test_one_values(self):
        x = torch.ones(3, 5)
        h = Highway(torch.nn.Identity(), 5)

        h.Wh.weight = nn.Parameter(0.2 * torch.ones_like(h.Wt.weight))

        # 50-50
        h.Wt.weight = nn.Parameter(torch.zeros_like(h.Wt.weight))
        h.Wt.bias = nn.Parameter(torch.zeros_like(h.Wt.bias))

        y = h(x)

        # sigmoid(0)=0.5, 1 * 0.5 + 1 * 0.5 = 1
        self.assertTrue(np.array_equal(np.ones((3, 5)), y.detach().numpy()))

    def test_H(self):
        x = torch.ones(3, 5)
        h = Highway(lambda x: x + 1, 5)

        h.Wh.weight = nn.Parameter(0.2 * torch.ones_like(h.Wt.weight))

        # 50-50
        h.Wt.weight = nn.Parameter(torch.zeros_like(h.Wt.weight))
        h.Wt.bias = nn.Parameter(torch.zeros_like(h.Wt.bias))

        y = h(x)

        # sigmoid(0)=0.5, (1+1) * 0.5 + 1 * 0.5 = 1.5
        self.assertTrue(np.array_equal(np.ones((3, 5)) + 0.5, y.detach().numpy()))

    def test_large_T(self):
        x = torch.ones(3, 5)
        h = Highway(lambda x: -x, 5)

        h.Wh.weight = nn.Parameter(0.2 * torch.ones_like(h.Wt.weight))

        # 1 - 0
        h.Wt.weight = nn.Parameter(torch.zeros_like(h.Wt.weight))
        h.Wt.bias = nn.Parameter(1000 * torch.ones_like(h.Wt.bias))

        y = h(x)

        # sigmoid(1000)=1,  -1 * 1 + 1 * (1-1) = -1
        self.assertTrue(np.allclose(-np.ones((3, 5)), y.detach().numpy()))

    def test_small_T(self):
        x = torch.ones(3, 5)
        h = Highway(lambda x: -x, 5)

        h.Wh.weight = nn.Parameter(0.2 * torch.ones_like(h.Wt.weight))

        # 0 - 1
        h.Wt.weight = nn.Parameter(torch.zeros_like(h.Wt.weight))
        h.Wt.bias = nn.Parameter(-1000 * torch.ones_like(h.Wt.bias))

        y = h(x)

        # sigmoid(1000)=0,  -1 * 0 + 1 * (1-0) = 1
        self.assertTrue(np.allclose(np.ones((3, 5)), y.detach().numpy()))


if __name__ == '__main__':
    unittest.main()
