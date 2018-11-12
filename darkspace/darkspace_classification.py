from __future__ import print_function, division
import numpy as np
from embedding_reader import EmbeddingReader
from spreadsheet_classifier import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--embedding_name", "-n", default=sys.stdin)
parser.add_argument("--data_path", "-d", default=sys.stdin)
parser.add_argument("--embedding_path", "-e", default=sys.stdin)
parser.add_argument("--vocab_file", "-v", default=sys.stdin)
args = parser.parse_args()

runFile = "run/"+args.embedding_name+"_run.tsv"
outFile = "performance/"+args.embedding_name+"_performance.tsv"
vocabFile = args.vocab_file
dataPath = args.data_path
embeddingPath = args.embedding_path
testSize = 100
randomizeTestSet = True
MAX_NUM_WORDS = 600000
rf = RunFile(runFile)

for i, run in enumerate(rf.runs):
    print("###New Model########################################################################", flush=True)
    reset_random_seed(7)
    model_name = run["model_name"]
    embedding_type = run["embedding_type"]
    doc_type = run["doc_type"]
    repeat = run["repeat"]
    print("model_name:", model_name)
    print("Embedding type:",embedding_type)
    print("Document type:", doc_type)
    if embedding_type=="glove300":
        repFile = embeddingPath+"glove.6B.300d.txt"
    elif embedding_type=="glove200":
        repFile = embeddingPath+"glove.6B.200d.txt"
    elif embedding_type=="glove100":
        repFile = embeddingPath+"glove.6B.100d.txt"
    elif embedding_type=="glove50":
        repFile = embeddingPath+"glove.6B.50d.txt"
    elif embedding_type=="bio_glove1024":
        repFile = embeddingPath+"bio_GloVe_1024.txt"
    elif embedding_type=="bio_glove300":
        repFile = embeddingPath+"bio_GloVe_300.txt"
    elif embedding_type=="bio_glove100":
        repFile = embeddingPath+"bio_GloVe_100.txt"
    elif embedding_type=="bio_glove50":
        repFile = embeddingPath+"bio_GloVe_50.txt"
    elif embedding_type=="bio_fasttext100":
        repFile = embeddingPath+"allMolecularBioCorpus2_fasttext.model.vec"
    elif embedding_type=="fasttext300":
        repFile = embeddingPath+"wiki.en.vec"        
    else:
        repFile = None
    
    accuracies = []
    vote_accuracies = []
    for iteration in range(1,repeat+1):
        print("###Iteration "+str(iteration)+"#################################")
        sd = SpreadsheetData(doc_type, embedding_type, testSize, randomizeTestSet,
                             vocabFile,dataPath, MAX_NB_WORDS = MAX_NUM_WORDS)
        # embedding matrix
        embedding_matrix = None
        if "glove" in embedding_type or "fasttext" in embedding_type:
            ER = EmbeddingReader(repFile,MAX_NUM_WORDS=MAX_NUM_WORDS)
            embedding_matrix = ER.make_embedding_matrix(sd.word_index)
        sce = SpreadsheetClassificationExecution(sd, embedding_matrix, model_name, doc_type)

        if doc_type == "abstract" or doc_type == "MeSH" or doc_type == "title" or doc_type == "title_abstract":
            vote_accuracy = np.nan
            print("Model: %s, Accuracy:%f"%(model_name,sce.accuracy))
        else:
            vote_accuracy = sce.accuracy_voter
            print("Model: %s, Accuracy:%f, Vote Accuracy:%f"%(model_name,sce.accuracy, vote_accuracy))
        accuracies.append(sce.accuracy)
        vote_accuracies.append(vote_accuracy)
    run['mean_accuracy'] = np.mean(accuracies)
    run['accuracy_se'] = np.std(accuracies) / np.sqrt(len(accuracies))

    if doc_type == "abstract" or doc_type == "MeSH" or doc_type == "title" or doc_type == "title_abstract":
        run['accuracy_vote_se'] = "N/A"
        run['mean_accuracy_vote'] = "N/A"
    else:
        run['accuracy_vote_se'] = np.std(vote_accuracies) / np.sqrt(len(vote_accuracies))
        run['mean_accuracy_vote'] = np.mean(vote_accuracies)
rf.print(outFile)