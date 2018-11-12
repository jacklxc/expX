import re
import glob
import numpy as np

import h5py
import pickle

import tensorflow as tf

sess = tf.Session()

import keras
from keras import optimizers
from keras import backend as K
K.set_session(sess)

from keras import regularizers
from keras.engine.topology import Layer
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout, Flatten, LSTM, Bidirectional, Lambda, Input,concatenate,Multiply
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras import regularizers
from keras.utils import plot_model
from keras.optimizers import RMSprop, Adam
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
#from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from keras.callbacks import TensorBoard
import pandas as pd
import random

def reset_random_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)

class RunFile:

    def __init__(self, runFile) :

        self.runs = []
        tsv = pd.read_csv(runFile, sep='\t')
        for i, row in tsv.iterrows():
            run = {}
            run['model_name'] = row['model_name']
            run['embedding_type'] = row['embedding_type']
            run['doc_type'] = row['doc_type']
            run['repeat'] = row['repeat']
            self.runs.append(run)

    def print(self, outFile):
        df = pd.DataFrame(data=self.runs)
        df.to_csv(outFile, sep="\t")

class SpreadsheetClassificationExecution:

    def __init__(self, sd, embedding_matrix, classifier_type, doc_type):

        #training params
        batch_size = 256
        num_epochs = 10 

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
        elif classifier_type == 'SimpleCNN':
            num_epochs = 100
            repeat = 10
            classifier = SimpleCNN(embedding_matrix, sd.max_seq_len, sd.n_classes)
        elif classifier_type == 'BidirectionalLSTMClassifier':
            classifier = BidirectionalLSTMClassifier(embedding_matrix, sd.max_seq_len, sd.n_classes)
        
        elif classifier_type == 'CNNBiLSTM':
            num_epochs = 20
            classifier = CNNBiLSTM(embedding_matrix, sd.max_seq_len, sd.n_classes)
        else:
            raise ValueError("Incorrect Classifier Type: %s"%(classifier_type))

        model = classifier.model
        print("Begin training",flush=True)
        #callbacks = [TensorBoard(log_dir='./logs/elmo_hub')]
        early_stopping = EarlyStopping(patience = 5)
        hist = model.fit(sd.x_train, sd.y_train, batch_size=batch_size,
                         epochs=num_epochs, validation_split=0.1,
                         shuffle=True, verbose=2, callbacks=[early_stopping])

        score = model.evaluate(sd.x_test, sd.y_test, verbose=2)
        self.loss = score[0]
        self.accuracy = score[1]
        self.Y_pred_train = model.predict(sd.x_train)
        self.Y_pred = model.predict(sd.x_test)
        print("Test accuracy:",score[1],flush=True)
        
        if not(doc_type == "abstract" or doc_type == "MeSH" or doc_type == "title" or doc_type == "title_abstract"):
            # Voter
            print("Training voter")
            inp = model.input                                          
            for layer in model.layers:
                if "bidirectional" in layer.name:
                    output = layer.output
                    break
            functor = K.function([inp, K.learning_phase()], [output])
            # Run all x_train once to obtain train_LSTM_state will cause memory problem, so batch x_train to the size of x_test
            # and run batches one after another
            # Get LSTM state for x_train
            batched_Xtrain = self.sliceX(sd.x_train, 40)
            train_LSTM_state = []
            for batch in batched_Xtrain:
                batch_train_LSTM_state = functor([batch, 0])
                train_LSTM_state.append(batch_train_LSTM_state[0])
            train_LSTM_state = np.concatenate(train_LSTM_state,axis=0)
            X_vote_train, pred_train, y_train = self.make_input(train_LSTM_state, 
                self.Y_pred_train,sd.train_paper_name, sd.y_train, sd.max_paragraph)
            
            # Get LSTM state for x_test
            test_LSTM_state = functor([sd.x_test, 0])
            X_vote_test, pred_test, y_test = self.make_input(test_LSTM_state[0], 
                self.Y_pred,sd.test_paper_name, sd.y_test, sd.max_paragraph)
            
            # Train voter
            early_stopping = EarlyStopping(patience = 10)
            voter = Voter(X_vote_train.shape)
            voter_hist = voter.model.fit(X_vote_train,y_train, batch_size=30, 
                            epochs=100, validation_split=0.1, shuffle=True, verbose=2, callbacks=[early_stopping])

            score_voter = voter.model.evaluate(X_vote_test, y_test, verbose=2)
            self.loss_voter = score_voter[0]
            self.accuracy_voter = score_voter[1]
            self.Y_pred_voter = model.predict(sd.x_test)
            print("Test voter accuracy:",score_voter[1],flush=True)
    
    def sliceX(self, X,n_batch=10):
        data_size = X.shape[0]
        batch_size = int(data_size / n_batch) + 1
        Xs = []
        for i in range(n_batch-1):
            Xs.append(X[i*batch_size:(i+1)*batch_size,:])
        Xs.append(X[(i+1)*batch_size:,:])
        return Xs
    
    def make_input(self,LSTMstate,pred,papers,label,max_paragraph):
        state_dim = LSTMstate.shape[-1]
        papers = np.array(papers)
        unique_papers = np.unique(papers)
        paper_count = unique_papers.size
        X = np.zeros((paper_count,max_paragraph,state_dim))
        pred_out = np.zeros((paper_count,max_paragraph,2))
        y = np.zeros((paper_count,2))
        for index, paper in enumerate(unique_papers):
            this_paper = papers == paper
            this_paper_length = np.sum(this_paper)
            fill_in_length = np.min([this_paper_length,max_paragraph])
            if label is not None:
                identical_labels = label[this_paper,:]
                y[index,:] = identical_labels[0,:]
            pred_to_fill = pred[this_paper,:]
            state_to_fill = LSTMstate[this_paper,:]
            pred_out[index,:fill_in_length,:] = pred_to_fill[:fill_in_length,:]
            X[index,:fill_in_length,:] = state_to_fill[:fill_in_length,:]
        return X,pred_out,y
    
class Attention(Layer):
    def __init__(self,**kwargs):
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.paper, self.paragraph, self.state_dim = input_shape
        self.attention = self.add_weight(name='attention', 
                                      shape=(self.state_dim,),
                                      initializer='glorot_normal',
                                      trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        # input_shape: (paper,paragraph,state_dim)
        paragraph_score = tf.tensordot(x,self.attention, axes=[[2],[0]])
        attention_score = K.softmax(paragraph_score,axis=1)
        reshaped = K.expand_dims(attention_score,axis=2)
        duplicate = K.repeat_elements(reshaped,self.state_dim,2)
        return duplicate

    def compute_output_shape(self, input_shape):
        return (self.paper, self.paragraph,self.state_dim)
    
class Summation(Layer):
    def __init__(self,**kwargs):
        super(Summation, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(Summation, self).build(input_shape)

    def call(self, x):
        # input shape: paper, paragraph, state_dim
        out = K.sum(x,axis=1)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

class Voter:
    def __init__(self,biLSTMState_shape):
        n_paper, n_paragraph, LSTMdim = biLSTMState_shape
        x = Input(shape=(n_paragraph,LSTMdim))
        attention = Attention()(x)
        with_attention = Multiply()([x,attention])
        state = Summation()(with_attention)
        p = Dense(2, activation='sigmoid')(state)
        #p = Dropout(0.25)(p)
        self.model = Model(inputs = x, outputs = p)
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
        self.model.compile(loss = "categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
        self.model.summary()
    
class SpreadsheetData:

    def __init__(self, doc_type, embedding_type, testSize=100, randomizeTestSet=False,\
                 vocabFile, dataPath, MAX_NB_WORDS=600000, n_rec=None):
        self.doc_type = doc_type
        self.dataPath = dataPath
        # Read text and label from DarkSpace
        label_file = dataPath+"all_pmids_eval.tsv"
        label_df = pd.read_csv(label_file, sep='\t', header=0, index_col=None,engine='python')
        if n_rec is None:
            n_rec = label_df.shape[0]
        if randomizeTestSet :
            test_ids = sorted(random.sample(range(n_rec), int(testSize)))
            self.randomized = True
        else:
            test_ids = range(int(testSize))
        train_ids = []
        for i in range(n_rec):
            if i not in test_ids:
                train_ids.append(i)
        
        if doc_type == "all" or doc_type == "evd_frg":
            train_paragraphs, train_paper_name, train_label = \
                self._get_text_label(label_df, train_ids,doc_type)

            test_paragraphs, test_paper_name, test_label = \
                self._get_text_label(label_df, test_ids,doc_type)
        elif doc_type == "caption":
            train_paragraphs, train_paper_name, train_label = \
                self._get_caption_label(label_df, train_ids,doc_type)

            test_paragraphs, test_paper_name, test_label = \
                self._get_caption_label(label_df, test_ids,doc_type)
        
        elif doc_type == "abstract" or doc_type == "MeSH" or doc_type == "title":
            train_paragraphs, train_paper_name, train_label = \
                self._get_single_label(label_df, train_ids,doc_type)

            test_paragraphs, test_paper_name, test_label = \
                self._get_single_label(label_df, test_ids,doc_type)
        elif doc_type == "caption_evdfrg":
            train_captions, train_caption_paper_name, train_caption_label = \
                self._get_caption_label(label_df, train_ids, "caption")

            test_captions, test_caption_paper_name, test_caption_label = \
                self._get_caption_label(label_df, test_ids, "caption")

            train_texts, train_text_paper_name, train_text_label = \
                self._get_text_label(label_df, train_ids, "evd_frg")

            test_texts, test_text_paper_name, test_text_label = \
                self._get_text_label(label_df, test_ids, "evd_frg")
            
            train_paragraphs, train_paper_name, train_label = self._combine_doc_type(train_captions,\
                train_caption_paper_name,train_caption_label, train_texts, train_text_paper_name, train_text_label)
            test_paragraphs, test_paper_name, test_label = self._combine_doc_type(test_captions,\
                test_caption_paper_name,test_caption_label, test_texts, test_text_paper_name, test_text_label)
        elif doc_type == "title_abstract":
            train_title, train_title_paper_name, train_title_label = \
                self._get_single_label(label_df, train_ids,"title")

            test_title, test_title_paper_name, test_title_label = \
                self._get_single_label(label_df, test_ids,"title")
            
            train_abstract, train_abstract_paper_name, train_abstract_label = \
                self._get_single_label(label_df, train_ids,"abstract")

            test_abstract, test_abstract_paper_name, test_abstract_label = \
                self._get_single_label(label_df, test_ids,"abstract")
            
            train_paragraphs, train_paper_name, train_label = self._combine_doc_type(train_title,\
                train_title_paper_name,train_title_label, train_abstract, train_abstract_paper_name, train_abstract_label)
            test_paragraphs, test_paper_name, test_label = self._combine_doc_type(test_title,\
                test_title_paper_name, test_title_label, test_abstract, test_abstract_paper_name, test_abstract_label)
        else:
            assert 0,"Wrong doc_type!"

        labels = list(set(train_label + test_label))

        y_train_base = [labels.index(i) for i in train_label]
        y_test_base = [labels.index(i) for i in test_label]

        # analyze word distribution
        seq_len = np.array([len(paragraph.split(" ")) for paragraph in train_paragraphs])
        self.mean_seq_len = np.round(seq_len.mean()).astype(int)
        self.max_seq_len = np.round(seq_len.mean() + seq_len.std()).astype(int)
        
        self.MAX_NB_WORDS = MAX_NB_WORDS

        raw_docs_train = train_paragraphs
        raw_docs_test = test_paragraphs
        
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
        self.train_paper_name = train_paper_name
        self.test_paper_name = test_paper_name
        self.max_paragraph = self._paragraph_distribution(self.train_paper_name) 
        # max number of paragraph in the paper collections

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
    
    def _get_text_label(self, label_df, paper_indices, doc_type):
        folderpath = self.dataPath+"tsv_clause_evfrgs/"
        all_paragraphs = []
        paper_name_per_paragraph = []
        label = []
        for paper_index in paper_indices:
            paper_id = str(label_df["pmid"][paper_index])
            searchkey = folderpath+paper_id+"*.tsv"
            found_files = glob.glob(searchkey)
            if len(found_files)>0:
                filename = found_files[0]
                paragraph = ""
                df = pd.read_csv(filename, sep='\t', header=0, index_col=0,engine='python')
                df = df[pd.notnull(df["Clause Text"])]
                prev_paragraph = ""
                for i in range(df.shape[0]):
                    try:
                        #if df["Discourse Label"][i] == "method":
                        if df["Paragraph"][i][0]=="p" or "article" in df["Paragraph"][i]: # e.g. "p1"
                            if doc_type=="all" or isinstance(df["fig_spans"][i], str): # evidence fragment
                                if df["Paragraph"][i]!=prev_paragraph:
                                    prev_paragraph = df["Paragraph"][i]
                                    if len(paragraph)>0:
                                        all_paragraphs.append(paragraph)
                                        paper_name_per_paragraph.append(paper_id)
                                        label.append(label_df["contains_interactions"][paper_index])
                                    paragraph = ""
                                paragraph+= df["Clause Text"][i].lower() + " " # Lower case!
                    except KeyError:
                        pass
        return all_paragraphs, paper_name_per_paragraph, label
    
    def _get_single_label(self, label_df, paper_indices, doc_type):
        all_texts = []
        paper_names = []
        label = []
        filename = self.dataPath+'abstracts.tsv'
        df = pd.read_csv(filename, sep='\t', header=0, index_col=0,engine='python')
        for paper_index in paper_indices:
            paper_id = label_df["pmid"][paper_index]
            try:
                text = df[doc_type][paper_id].lower()
            except:
                continue
            all_texts.append(text)
            paper_names.append(str(paper_id))
            label.append(label_df["contains_interactions"][paper_index])
        return all_texts, paper_names, label
    
    def _get_caption_label(self, label_df, paper_indices, doc_type):
        all_captions = []
        paper_name_per_caption = []
        label = []
        filename = self.dataPath+'captions.tsv'
        df = pd.read_csv(filename, sep='\t', header=0, index_col=None,engine='python')
        df = df[pd.notnull(df["Caption"])]
        for paper_index in paper_indices:
            paper_id = label_df["pmid"][paper_index]
            caption_idx = df["DocId"] == paper_id
            captions = df["Caption"][caption_idx].tolist()
            length = len(captions)
            if length>0:
                captions = [caption.lower() for caption in captions]
                all_captions.extend(captions)
                paper_name_per_caption.extend([str(paper_id)]*length)
                label.extend([label_df["contains_interactions"][paper_index]]*length)
        return all_captions, paper_name_per_caption, label
    
    def _combine_doc_type(self, doc1,doc1_paper,doc1_label, doc2, doc2_paper, doc2_label):
        doc1 = np.array(doc1)
        doc1_paper = np.array(doc1_paper)
        doc1_label = np.array(doc1_label)
        doc2 = np.array(doc2)
        doc2_paper = np.array(doc2_paper)
        doc2_label = np.array(doc2_label)

        all_papers = np.unique(np.append(doc1_paper,doc2_paper))
        X = np.array([],dtype="|S1")
        y = np.array([],dtype="|S1")
        papers = np.array([],dtype="|S1")
        for paper in all_papers:
            this_paper_doc1 = doc1_paper==paper
            X = np.append(X,doc1[this_paper_doc1])
            y = np.append(y,doc1_label[this_paper_doc1])
            papers = np.append(papers,doc1_paper[this_paper_doc1])
            this_paper_doc2 = doc2_paper==paper
            X = np.append(X,doc2[this_paper_doc2])
            y = np.append(y,doc2_label[this_paper_doc2])
            papers = np.append(papers,doc2_paper[this_paper_doc2])
        return X.tolist(), papers.tolist(), y.tolist()
    
    def _paragraph_distribution(self,papers):
        paper_freq = {x:papers.count(x) for x in papers}
        max_freq = np.max(list(paper_freq.values()))
        return max_freq 

class FancyConvolutionNetworkClassifier:

    rep_max = -100000.0
    rep_size = 0

    def __init__(self, embedding_matrix, max_seq_len, n_classes, num_filters = 16, weight_decay = 1e-4):
        dropout = 0.5
        reg = 1e-4
        nb_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]
        
        self.model = Sequential()
        self.model.add(Embedding(nb_words, embed_dim,
            weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
        self.model.add(Dropout(dropout))
        self.model.add(Conv1D(num_filters, 5, strides=3, activation='relu', padding='valid',\
                              kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg)))
        self.model.add(Dropout(dropout))
        self.model.add(MaxPooling1D(4))
        self.model.add(Conv1D(num_filters, 5, strides=3, activation='relu', padding='valid',\
                              kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg)))
        self.model.add(Dropout(dropout))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(reg)))
        self.model.add(Dense(n_classes,kernel_regularizer=regularizers.l2(reg),activation='sigmoid'))  #multi-label (k-hot encoding)

        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        self.model.summary()

class SimpleCNN:

    rep_max = -100000.0
    rep_size = 0

    def __init__(self, embedding_matrix, max_seq_len, n_classes, num_filters = 8, weight_decay = 1e-4):
        dropout = 0.4
        reg = 1e-4
        nb_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]
        
        self.model = Sequential()
        self.model.add(Embedding(nb_words, embed_dim,
            weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
        self.model.add(Dropout(dropout))
        self.model.add(Conv1D(num_filters, 5, strides=2, activation='relu', padding='valid',\
                              kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg)))
        self.model.add(Dropout(dropout))
        self.model.add(MaxPooling1D())
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dense(4, activation='relu', kernel_regularizer=regularizers.l2(reg)))
        self.model.add(Dense(n_classes,kernel_regularizer=regularizers.l2(reg),activation='sigmoid'))  #multi-label (k-hot encoding)

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

        adam = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
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
                            weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(32))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(n_classes, activation='sigmoid'))
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=0.001),
                      metrics=['accuracy'])
        self.model.summary()

class BidirectionalLSTMClassifier:
    def __init__(self, embedding_matrix, max_seq_len, n_classes):
        reg = 1e-4
        dropout = 0.5
        nb_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]

        self.model = Sequential()
        self.model.add(Embedding(nb_words, embed_dim,
                                 weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
        self.model.add(Dropout(dropout))
        self.model.add(Bidirectional(LSTM(32,kernel_regularizer=regularizers.l2(reg),\
                                          recurrent_regularizer=regularizers.l2(reg), \
                                          bias_regularizer=regularizers.l2(reg))))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(n_classes, activation='sigmoid'))
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=adam,
                           metrics=['accuracy'])
        self.model.summary()


class CNNBiLSTM:
    def __init__(self, embedding_matrix, max_seq_len, n_classes, num_filters = 16, weight_decay = 1e-4):

        reg = 1e-4
        dropout = 0.4
        nb_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]

        self.model = Sequential()
        self.model.add(Embedding(nb_words, embed_dim,
            weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
        self.model.add(Dropout(dropout))
        self.model.add(Conv1D(num_filters, 5, strides=1, activation='relu', padding='valid',\
                              kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg)))
        self.model.add(Dropout(dropout))
        self.model.add(MaxPooling1D(4))
        self.model.add(Conv1D(num_filters, 5, strides=1, activation='relu', padding='valid',\
                              kernel_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg)))
        self.model.add(Dropout(dropout))
        self.model.add(Bidirectional(LSTM(64,kernel_regularizer=regularizers.l2(reg),\
                                  recurrent_regularizer=regularizers.l2(reg), \
                                  bias_regularizer=regularizers.l2(reg))))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(n_classes, activation='sigmoid'))  #multi-label (k-hot encoding)

        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        self.model.summary()
