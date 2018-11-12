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

    def __init__(self, runFile, randomize) :

        self.randomize = randomize
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
            run['randomize'] = self.randomize
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
        elif classifier_type == 'PreComputedElmoBidirectionalLSTMClassifier':
            batch_size = 16
            num_epochs = 100
            classifier = PreComputedElmoBidirectionalLSTMClassifier(sd.n_classes)
        else:
            raise ValueError("Incorrect Classifier Type: %s"%(classifier_type))

        model = classifier.model
        print("Begin training")
        #callbacks = [TensorBoard(log_dir='./logs/elmo_hub')]
        #early_stopping = EarlyStopping(patience = 10)
        hist = model.fit(sd.x_train, sd.y_train, batch_size=batch_size,
                         epochs=num_epochs, validation_split=0.1,
                         shuffle=True, verbose=2)
        
        score = model.evaluate(sd.x_test, sd.y_test, verbose=2)
        self.loss = score[0]
        self.accuracy = score[1]
        print("Test accuracy:",score[1])

class SpreadsheetData:

    x_train = None
    x_test = None
    n_classes = 0
    y_train = None
    y_test = None
    labels = None
    word_index = None
    randomized = False

    def __init__(self, inFile, textColumn, labelColumn, testSize, randomizeTestSet=False, preComputedEmbedding=False, MAX_NUM_WORDS = 100000):
        vocabFile="/nas/home/xiangcil/bio_corpus/abstracts/bioVocab50.txt"
        df = pd.read_csv(inFile, sep='\t', header=0, index_col=0,engine='python')

        # Remove records with missing data.
        df = df[pd.notnull(df[textColumn])]
        n_rec = df.shape[0]

        if randomizeTestSet :
            test_ids = sorted(random.sample(range(n_rec), int(testSize)))
            self.randomized = True
        else:
            test_ids = range(int(testSize))
        train_ids = []
        for i in range(n_rec):
            if i not in test_ids:
                train_ids.append(i)

        df_train = df.iloc[train_ids,:]
        df_test = df.iloc[test_ids,:]

        labels = df[labelColumn].unique().tolist()

        y_train_base = [labels.index(i) for i in df_train[labelColumn]]
        y_test_base = [labels.index(i) for i in df_test[labelColumn]]

        # analyze word distribution
        df_train['doc_len'] = df_train[textColumn].apply(lambda words: len(words.split(" ")))
        self.mean_seq_len = np.round(df_train['doc_len'].mean()).astype(int)
        self.max_seq_len = np.round(df_train['doc_len'].mean() + df_train['doc_len'].std()).astype(int)

        np.random.seed(0)
        
        if preComputedEmbedding:
            if textColumn=="evd_frg":
                embedding_file = "/nas/home/xiangcil/figure_classification/evidence_fragment.hdf5"
            elif textColumn=="text":
                embedding_file = "/nas/home/xiangcil/figure_classification/text.hdf5"
            else:
                assert False, "Wrong text column!"
            self.x_train = np.zeros((len(train_ids),self.max_seq_len,1024*3))
            self.x_test = np.zeros((len(test_ids),self.max_seq_len,1024*3))
            self.train_ids = train_ids
            self.test_ids=test_ids
            with h5py.File(embedding_file, 'r') as fin:
                for index,i in tqdm(enumerate(train_ids)):
                    #matrix = np.max(fin[str(i)][()],axis=0) # Max pooling
                    matrix = np.concatenate((fin[str(i)][()][0,:,:],fin[str(i)][()][1,:,:],fin[str(i)][()][2,:,:]),axis=1)
                    sentence_length = np.min([matrix.shape[0],self.max_seq_len])
                    self.x_train[index,:sentence_length,:] = matrix[:sentence_length,:]
                for index,i in tqdm(enumerate(test_ids)):
                    #matrix = np.max(fin[str(i)][()],axis=0)
                    matrix = np.concatenate((fin[str(i)][()][0,:,:],fin[str(i)][()][1,:,:],fin[str(i)][()][2,:,:]),axis=1)
                    sentence_length = np.min([matrix.shape[0],self.max_seq_len])
                    self.x_test[index,:sentence_length,:] = matrix[:sentence_length,:]
            self.n_classes = len(labels)
            self.labels = labels
            self.y_train = keras.utils.to_categorical(y_train_base, num_classes=self.n_classes)
            self.y_test = keras.utils.to_categorical(y_test_base, num_classes=self.n_classes)
            self.train_size = self.y_train.shape[0]
            self.test_size = self.y_test.shape[0]
        else:
            self.MAX_NB_WORDS = 600000

            raw_docs_train = df_train[textColumn].tolist()
            raw_docs_test = df_test[textColumn].tolist()
            self.raw_docs_train = np.array(raw_docs_train, dtype=str)[:, np.newaxis]
            self.raw_docs_test = np.array(raw_docs_test, dtype=str)[:, np.newaxis]

            print("pre-processing train data...")
            
            vocab = []
            with open(vocabFile,"r") as vfile:
                for line in vfile:
                    vocab.append(line.strip())
            self.vocab = set(vocab)
            
            processed_docs_train = []
            for doc in raw_docs_train:
                filtered = []
                tokens = doc.split()
                for word in tokens:
                    word = self._clean_url(word)
                    word = self._clean_num(word)
                    if word not in self.vocab:
                        word = "<UNK>"
                    filtered.append(word)
                processed_docs_train.append(" ".join(filtered))

            processed_docs_test = []
            for doc in raw_docs_test:
                filtered = []
                tokens = doc.split()
                for word in tokens:
                    word = self._clean_url(word)
                    word = self._clean_num(word)
                    if word not in self.vocab:
                        word = "<UNK>"
                    filtered.append(word)
                processed_docs_test.append(" ".join(filtered))

            print("tokenizing input data...")
            tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS, lower=True, char_level=False)
            tokenizer.fit_on_texts(processed_docs_train + processed_docs_test)  #leaky
            word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)
            word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)
            self.word_index = tokenizer.word_index
            print("dictionary size: ", len(self.word_index))

            #pad sequences
            self.x_train = sequence.pad_sequences(word_seq_train, maxlen=self.max_seq_len)
            self.x_test = sequence.pad_sequences(word_seq_test, maxlen=self.max_seq_len)

            self.n_classes = len(labels)
            self.labels = labels
            self.y_train = keras.utils.to_categorical(y_train_base, num_classes=self.n_classes)
            self.y_test = keras.utils.to_categorical(y_test_base, num_classes=self.n_classes)

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

class PreComputedElmoBidirectionalLSTMClassifier:
    def __init__(self, n_classes):
        self.model = Sequential()
        self.model.add(Dropout(0.25))
        self.model.add(Bidirectional(LSTM(128,batch_input_shape=(None,None,1024*3))))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(n_classes, activation='sigmoid'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        #self.model.summary()
        
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



