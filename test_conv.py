import unittest

import torch

from cnn import CNN


class TestPConv(unittest.TestCase):
    def test_output_size(self):
        words = 10
        in_channels = 3
        out_channels = 3
        max_word_length = 21
        kernel_size = 5

        x = torch.ones(words, in_channels, max_word_length)
        conv = CNN(max_word_length, kernel_size, in_channels, out_channels)
        y = conv(x)
        s = torch.zeros(words, out_channels)
        self.assertEqual(s.size(), y.size())

    def test_change_words(self):
        words = 5
        in_channels = 3
        out_channels = 3
        max_word_length = 21
        kernel_size = 5

        x = torch.ones(words, in_channels, max_word_length)
        conv = CNN(max_word_length, kernel_size, in_channels, out_channels)
        y = conv(x)
        s = torch.zeros(words, out_channels)
        self.assertEqual(s.size(), y.size())

    def test_change_out_channels(self):
        words = 5
        in_channels = 3
        out_channels = 4
        max_word_length = 21
        kernel_size = 5

        x = torch.ones(words, in_channels, max_word_length)
        conv = CNN(max_word_length, kernel_size, in_channels, out_channels)
        y = conv(x)
        s = torch.zeros(words, out_channels)
        self.assertEqual(s.size(), y.size())

    def test_word_length(self):
        words = 5
        in_channels = 3
        out_channels = 4
        max_word_length = 15
        kernel_size = 5

        x = torch.ones(words, in_channels, max_word_length)
        conv = CNN(max_word_length, kernel_size, in_channels, out_channels)
        y = conv(x)
        s = torch.zeros(words, out_channels)
        self.assertEqual(s.size(), y.size())

    def test_word_length_too_big(self):
        words = 5
        in_channels = 3
        out_channels = 4
        max_word_length = 15
        kernel_size = 5

        x = torch.ones(words, in_channels, max_word_length + 1)
        conv = CNN(max_word_length, kernel_size, in_channels, out_channels)
        with self.assertRaises(AssertionError):
            y = conv(x)

    def test_word_length_too_small(self):
        words = 5
        in_channels = 3
        out_channels = 4
        max_word_length = 21
        kernel_size = 5

        x = torch.ones(words, in_channels, max_word_length - 1)
        conv = CNN(max_word_length, kernel_size, in_channels, out_channels)
        with self.assertRaises(RuntimeError):
            y = conv(x)

    def test_kernel_to_large(self):
        words = 5
        in_channels = 3
        out_channels = 4
        max_word_length = 21
        kernel_size = 22

        x = torch.ones(words, in_channels, max_word_length)
        conv = CNN(max_word_length, kernel_size, in_channels, out_channels)
        with self.assertRaises(RuntimeError):
            y = conv(x)

    def test_input_channels_to_small(self):
        words = 5
        in_channels = 3
        out_channels = 4
        max_word_length = 21
        kernel_size = 5

        x = torch.ones(words, in_channels - 1, max_word_length)
        conv = CNN(max_word_length, kernel_size, in_channels, out_channels)
        with self.assertRaises(RuntimeError):
            y = conv(x)

    def test_input_channels_to_big(self):
        words = 5
        in_channels = 3
        out_channels = 4
        max_word_length = 21
        kernel_size = 5

        x = torch.ones(words, in_channels + 1, max_word_length)
        conv = CNN(max_word_length, kernel_size, in_channels, out_channels)
        with self.assertRaises(RuntimeError):
            y = conv(x)


if __name__ == '__main__':
    unittest.main()
