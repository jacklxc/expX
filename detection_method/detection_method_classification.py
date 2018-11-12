from __future__ import print_function, division

from embedding_reader import EmbeddingReader
from spreadsheet_classifier import *
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--embedding_name", "-n", default=sys.stdin)
parser.add_argument("--data_file", "-d", default=sys.stdin)
parser.add_argument("--vocab_file", "-v", default=sys.stdin)
parser.add_argument("--embedding_path", "-e", default=sys.stdin)

args = parser.parse_args()

runFile = "run/"+args.embedding_name+"_run.tsv"
outFile = "performance/"+args.embedding_name+"_performance.tsv"
vocabFile = args.vocab_file
dataFile = args.data_file
embeddingPath = args.embedding_path

testSize = 400
randomizeTestSet = True
MAX_NUM_WORDS = 600000
rf = RunFile(runFile, randomizeTestSet)
repeat = 5

for i, run in enumerate(rf.runs):
    reset_random_seed(0)
    print("###New Model########################################################################", flush=True)
    model_name = run["model_name"]
    embedding_type = run["embedding_type"]
    text_column = run["text_column"]
    label_column = run["label_column"]
    print("model_name:", model_name)
    print("Embedding type:",embedding_type)
    print("text column:", text_column)
    print("label column:", label_column)
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
    for iteration in range(repeat):
        if "glove" in embedding_type or "fasttext" in embedding_type:
            ER = EmbeddingReader(repFile,MAX_NUM_WORDS)

        sd = SpreadsheetData(dataFile, text_column, label_column,
                             testSize, randomizeTestSet,preComputedEmbedding="PreComputedElmo" in model_name, 
                             MAX_NUM_WORDS = MAX_NUM_WORDS)
        # embedding matrix
        embedding_matrix = None
        if "glove" in embedding_type or "fasttext" in embedding_type:
            embedding_matrix = ER.make_embedding_matrix(sd.word_index)
        sce = SpreadsheetClassificationExecution(sd, embedding_matrix, model_name)
        accuracies.append(sce.accuracy)
        print("Text: %s, Label: %s, Model: %s, Accuracy:%f"%
              (text_column, label_column,model_name,sce.accuracy))
    run['accuracy_mean'] = np.mean(accuracies)
    run['accuracy_se'] = np.std(accuracies) / np.sqrt(len(accuracies))
rf.print(outFile)
