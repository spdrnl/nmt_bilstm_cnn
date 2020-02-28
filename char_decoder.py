#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        self.charDecoder = nn.LSTM(input_size=char_embedding_size, hidden_size=hidden_size)
        self.char_output_projection = nn.Linear(in_features=hidden_size, out_features=len(target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(num_embeddings=len(target_vocab.char2id),
                                           embedding_dim=char_embedding_size,
                                           padding_idx=target_vocab.char2id['<pad>'])
        self.target_vocab = target_vocab
        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        st_scores = []
        # column will contain the first, second, third characters over all entries of the batch
        for column in torch.split(input, split_size_or_sections=1, dim=0):
            embedded_input = self.decoderCharEmb(column)
            _, dec_hidden = self.charDecoder(embedded_input, dec_hidden)
            st_new = self.char_output_projection(dec_hidden[0])
            st_scores.append(st_new)

        st_scores = torch.cat(st_scores, dim=0)
        return st_scores,  dec_hidden
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).

        # Leave out the last character of the input (being end or pad)
        inputs = char_sequence[:-1, :]
        # Create targets, shifted one: _joli should predict joli_
        targets = char_sequence[1:, :]

        # Get the scores
        st_scores, _ = self.forward(inputs, dec_hidden)

        # Apply softmax
        softmax_scores = nn.functional.log_softmax(st_scores, dim=2)

        # Get the relevant values
        selected_values = torch.gather(softmax_scores, index=targets.unsqueeze(2), dim=2).squeeze(2)

        # Mask the paddings, i.e mask is zero there
        target_masks = (targets != self.target_vocab.char2id['<pad>']).float()

        # Sum and minus to get the loss
        loss = -(selected_values * target_masks).sum()

        return loss
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        # Create vector of start values
        batch_size = list(initialStates[0].size())[1]
        start_character_idx = self.target_vocab.char2id['{']
        start_value_idxs = torch.Tensor([start_character_idx], device=device).expand(1, batch_size).long()

        # Loop max_length to generate characters
        input_values = start_value_idxs
        dec_states = initialStates
        idx_predictions = []
        for i in range(max_length):
            st_scores, dec_states = self.forward(input_values, dec_states)
            input_values = st_scores.argmax(dim=2)
            idx_predictions.append(input_values)

        # Decode into characters
        import itertools
        words = []
        idx_predictions = torch.cat(idx_predictions, dim=0).detach().numpy()
        for i in range(batch_size):
            word_idxs = idx_predictions[:,i]
            chars = [self.target_vocab.id2char[idx] for idx in list(word_idxs)]
            chars = list(itertools.takewhile(lambda x: x != '}', chars))
            words.append("".join(chars))

        return words
        
        ### END YOUR CODE

