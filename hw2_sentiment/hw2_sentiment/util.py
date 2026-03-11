import csv
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple


class Vocabulary:
    """
    A vocabulary class for managing word-to-index mappings in NLP tasks.

    This class maintains bidirectional mappings between words and their corresponding
    indices, with special tokens for padding (<PAD>) and unknown words (<UNK>).
    """

    def __init__(self):
        """
        Initialize the vocabulary with special tokens.

        Sets up the vocabulary with:
        - <PAD> token at index 0 (for padding sequences)
        - <UNK> token at index 1 (for unknown/out-of-vocabulary words)
        """
        self.word_to_index = {"<PAD>": 0, "<UNK>": 1}
        self.index_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.next_index = 2

    def add_word(self, word: str) -> int:
        """
        Add a word to the vocabulary and return its index.

        If the word already exists in the vocabulary, returns its existing index.
        If the word is new, assigns it the next available index.

        @param word: The word to add to the vocabulary
        @return: The index assigned to the word
        """
        if word not in self.word_to_index:
            self.word_to_index[word] = self.next_index
            self.index_to_word[self.next_index] = word
            self.next_index += 1
        return self.word_to_index[word]

    def get_index(self, word: str) -> int:
        """
        Get the index of a word in the vocabulary.

        @param word: The word to look up
        @return: The index of the word, or 1 (<UNK> token index) if word not found
        """
        return self.word_to_index.get(word, 1)  # Return <UNK> if not found

    def get_vector(self, word: str, embedding_layer: nn.Embedding) -> torch.Tensor:
        """
        Get embedding vector for a word using the embedding layer.

        @param word: The word to get the embedding vector for
        @param embedding_layer: PyTorch embedding layer containing word vectors
        @return: The embedding vector for the word (detached from computation graph)
        """
        word_idx = self.get_index(word)
        return embedding_layer.weight[word_idx].detach()  # Detach to avoid autograd issues

    def size(self) -> int:
        """
        Get the size of the vocabulary.

        @return: The total number of words in the vocabulary (including special tokens)
        """
        return self.next_index


def read_tweet_data(filename: str) -> Tuple[List[str], np.ndarray]:
    """
    Read tweet emotion dataset from CSV file.

    @param filename: Path to the CSV file
    @return: Tuple of (texts, labels) where labels is a one-hot encoded matrix
    """
    texts = []
    labels = []

    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row['text']
            emotion_vector = [
                int(row['joy']),
                int(row['anger']),
                int(row['fear'])
            ]
            texts.append(text)
            labels.append(emotion_vector)

    return texts, np.array(labels)