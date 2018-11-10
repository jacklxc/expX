import os
import sys
import numpy as np
#from tqdm import tqdm

class EmbeddingReader:
    def __init__(self, EMBEDDING_DIR, MAX_NUM_WORDS=600000):
        print('Indexing word vectors.')
        self.MAX_NUM_WORDS = MAX_NUM_WORDS
        self.embeddings_index = {}
        word_count = 0
        with open(EMBEDDING_DIR) as f:
            for line in f:
                word_count += 1 
                values = line.split()
                word = values[0]
                try:
                    coefs = np.asarray(values[1:], dtype='float32')
                except ValueError:
                    continue
                self.embeddings_index[word] = coefs
                if word_count>self.MAX_NUM_WORDS:
                    break
            self.EMBEDDING_DIM = coefs.size
        print('Found %s word vectors.' % len(self.embeddings_index))
    
    def make_embedding_matrix(self,word_index):
        print('Preparing embedding matrix.')
        # prepare embedding matrix
        num_words = min(self.MAX_NUM_WORDS, len(word_index) + 1)
        self.embedding_matrix = np.zeros((num_words, self.EMBEDDING_DIM))
        for word, i in word_index.items():
            if i >= self.MAX_NUM_WORDS:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector
        return self.embedding_matrix
