import string
import re

import numpy as np

import h5py

import tensorflow as tf

sess = tf.Session()

import keras
from keras import optimizers
from keras import backend as K
K.set_session(sess)

from keras import regularizers
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout, Flatten, LSTM, Bidirectional, Lambda, Input
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras import regularizers
from keras.utils import plot_model
from keras.optimizers import RMSprop, Adam
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from keras.callbacks import TensorBoard
import pandas as pd
import random

def reset_random_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class RunFile:

    def __init__(self, runFile) :

        self.runs = []
        tsv = pd.read_csv(runFile, sep='\t')
        for i, row in tsv.iterrows():
            run = {}
            run['text_column'] = row['text_column']
            run['label_column'] = row['label_column']
            run['model_name'] = row['model_name']
            run['embedding_type'] = row['embedding_type']
            
            if row.get('keras_file',None) is not None:
                run['keras_file'] = row['keras_file']
            self.runs.append(run)

    def print(self, outFile):
        df = pd.DataFrame(data=self.runs)
        df.to_csv(outFile, sep="\t")

class SpreadsheetClassificationExecution:

    def __init__(self, sd, embedding_matrix, classifier_type) :

        #training params
        batch_size = 128
        num_epochs = 100

        #model parameters
        num_filters = 64
        embed_dim = 100
        weight_decay = 1e-4

        if classifier_type == 'SuperSimpleLSTMClassifier':
            classifier = SuperSimpleLSTMClassifier(embedding_matrix, sd.max_seq_len, sd.n_classes)
        elif classifier_type == 'SuperSimpleLSTMClassifierRandomEmbedding':
            num_epochs = 20
            batch_size = 64
            classifier = SuperSimpleLSTMClassifierRandomEmbedding(sd.max_seq_len, sd.n_classes)
        elif classifier_type == 'CNNRandomEmbedding':
            num_epochs = 20
            batch_size = 64
            classifier = CNNRandomEmbedding(sd.max_seq_len, sd.n_classes)
        elif classifier_type == 'FancyConvolutionNetworkClassifier':
            classifier = FancyConvolutionNetworkClassifier(embedding_matrix, sd.max_seq_len, sd.n_classes)
        elif classifier_type == 'BidirectionalLSTMClassifier':
            classifier = BidirectionalLSTMClassifier(embedding_matrix, sd.max_seq_len, sd.n_classes)
        else:
            raise ValueError("Incorrect Classifier Type: %s"%(classifier_type))

        model = classifier.model
        print("Begin training")
        #early_stopping = EarlyStopping(patience = 10)
        hist = model.fit(sd.x_train, sd.y_train, batch_size=batch_size,
                         epochs=num_epochs, validation_data=(sd.x_dev, sd.y_dev),
                         shuffle=True, verbose=2)
        
        score = model.evaluate(sd.x_test, sd.y_test, verbose=2)
        self.loss = score[0]
        self.accuracy = score[1]
        print("Test accuracy:",score[1])

class SpreadsheetData:

    def __init__(self, trainFile, devFile, testFile, textColumn, labelColumn, vocab, MAX_NUM_WORDS = 100000):
        self.MAX_NB_WORDS = 600000
        self.vocab = vocab
        self.textColumn = textColumn
        self.labelColumn = labelColumn
        self.x_train, self.y_train = self.read_data(trainFile, train=True)
        self.x_dev, self.y_dev = self.read_data(devFile)
        self.x_test, self.y_test = self.read_data(testFile)

    def read_data(self, inFile, train = False):
        df = pd.read_csv(inFile, sep='\t', header=0, index_col=0,engine='python')

        # Remove records with missing data.
        df = df[pd.notnull(df[self.textColumn])]
        labels = df[self.labelColumn].unique().tolist()
        y_base = [labels.index(i) for i in df[self.labelColumn]]

        if train:
        # analyze word distribution
            df['doc_len'] = df[self.textColumn].apply(lambda words: len(words.split(" ")))
            self.mean_seq_len = np.round(df['doc_len'].mean()).astype(int)
            self.max_seq_len = np.round(df['doc_len'].mean() + df['doc_len'].std()).astype(int)

        np.random.seed(0)

        raw_docs = df[self.textColumn].tolist()

        print("pre-processing train data...")        
        processed_docs = []
        for doc in raw_docs:
            filtered = []
            tokens = doc.split()
            for word in tokens:
                word = self._clean_url(word)
                word = self._clean_num(word)
                if word not in self.vocab:
                    word = "<UNK>"
                filtered.append(word)
            processed_docs.append(" ".join(filtered))
        if train:
            self.tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS, lower=True, char_level=False)
            self.tokenizer.fit_on_texts(processed_docs)  #leaky

        print("tokenizing input data...")
        word_seq_train = self.tokenizer.texts_to_sequences(processed_docs)

        if train:
            self.word_index = self.tokenizer.word_index
            print("dictionary size: ", len(self.word_index))

        #pad sequences
        X = sequence.pad_sequences(word_seq_train, maxlen=self.max_seq_len)
        if train:
            self.n_classes = len(labels)
            self.labels = labels
        Y = keras.utils.to_categorical(y_base, num_classes=self.n_classes)
        return X, Y

    def _clean_url(self,word):
        """
            Clean specific data format from social media
        """
        # clean urls
        word = re.sub(r'https? : \/\/.*[\r\n]*', '<URL>', word)
        word = re.sub(r'exlink', '<URL>', word)
        return word


    def _clean_num(self,word):
        # check if the word contain number and no letters
        if any(char.isdigit() for char in word):
            try:
                num = float(word.replace(',', ''))
                return '<NUM>'
            except:
                if not any(char.isalpha() for char in word):
                    return '<NUM>'
        return word
    
class FancyConvolutionNetworkClassifier:

    rep_max = -100000.0
    rep_size = 0

    def __init__(self, embedding_matrix, max_seq_len, n_classes, num_filters = 64, weight_decay = 1e-4):

        nb_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]

        self.model = Sequential()
        self.model.add(Embedding(nb_words, embed_dim,
            weights=[embedding_matrix], input_length=max_seq_len, trainable=True))
        self.model.add(Dropout(0.25))
        self.model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
        self.model.add(MaxPooling1D(2))
        self.model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Dense(n_classes, activation='sigmoid'))  #multi-label (k-hot encoding)

        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        self.model.summary()

class CNNRandomEmbedding:

    rep_max = -100000.0
    rep_size = 0

    def __init__(self, max_seq_len, n_classes, num_filters = 64, weight_decay = 1e-4):
        nb_words = 17490
        embed_dim = 1024
        self.model = Sequential()
        self.model.add(Embedding(nb_words, embed_dim,
                         input_length=max_seq_len, trainable=True))
        self.model.add(Dropout(0.25))
        self.model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
        self.model.add(MaxPooling1D(2))
        self.model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(Dense(n_classes, activation='sigmoid'))  #multi-label (k-hot encoding)

        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        self.model.summary()
        
class SuperSimpleLSTMClassifierRandomEmbedding:

    def __init__(self, max_seq_len, n_classes):

        nb_words = 17490
        embed_dim = 1024

        self.model = Sequential()
        self.model.add(Embedding(nb_words, embed_dim,
                             input_length=max_seq_len, trainable=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(128))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(n_classes, activation='sigmoid'))
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        self.model.summary()        
        
class SuperSimpleLSTMClassifier:

    def __init__(self, embedding_matrix, max_seq_len, n_classes):

        nb_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]

        self.model = Sequential()
        self.model.add(Embedding(nb_words, embed_dim,
                            weights=[embedding_matrix], input_length=max_seq_len, trainable=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(128))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(n_classes, activation='sigmoid'))
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=0.001),
                      metrics=['accuracy'])
        self.model.summary()

class BidirectionalLSTMClassifier:
    def __init__(self, embedding_matrix, max_seq_len, n_classes):
        nb_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]

        self.model = Sequential()
        self.model.add(Embedding(nb_words, embed_dim,
                                 weights=[embedding_matrix], input_length=max_seq_len, trainable=True))
        self.model.add(Dropout(0.5))
        self.model.add(Bidirectional(LSTM(128)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(n_classes, activation='sigmoid'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
        self.model.summary()



