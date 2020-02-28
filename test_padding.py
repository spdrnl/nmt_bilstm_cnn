import unittest
from utils import *

class TestPadding(unittest.TestCase):
    def test_empty_word(self):
        self.assertEqual(pad_char_word([], 0, 3), [0, 0, 0])

    def test_single_char(self):
        self.assertEqual(pad_char_word([1], 0, 3), [1, 0, 0])

    def test_full_word(self):
        self.assertEqual(pad_char_word([1, 2, 3], 0, 3), [1, 2, 3])

    def test_long_word(self):
        self.assertEqual(pad_char_word([1, 2, 3, 4], 0, 3), [1, 2, 3])

    def test_empty_sent(self):
        self.assertEqual(pad_char_sent([], 2, 0, 3), [[0, 0, 0], [0, 0, 0]])

    def test_single_word1(self):
        self.assertEqual(pad_char_sent([[1]], 2, 0, 3), [[1, 0, 0], [0, 0, 0]])

    def test_full_sent(self):
        self.assertEqual(pad_char_sent([[1, 2, 3], [1, 2, 3]], 2, 0, 3), [[1, 2, 3], [1, 2, 3]])

    def test_long_sent(self):
        self.assertEqual(pad_char_sent([[1, 2, 3, 4], [1, 2], [], [5, 6]], 3, 0, 3), [[1, 2, 3], [1, 2, 0], [0, 0, 0]])

if __name__ == '__main__':
    unittest.main()