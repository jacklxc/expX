from __future__ import print_function, division
import sys

from embedding_reader import EmbeddingReader
from spreadsheet_classifier_split import reset_random_seed, RunFile, SpreadsheetClassificationExecution, SpreadsheetData
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--embedding_name", "-n", default=sys.stdin)
parser.add_argument("--embedding_path", "-e", default=sys.stdin)

args = parser.parse_args()

runFile = "runText/"+args.embedding_name+"_run.tsv"
outFile = "performance/"+args.embedding_name+"_performance.tsv"
embeddingPath = args.embedding_path

MAX_NUM_WORDS = 600000
rf = RunFile(runFile)
repeat = 10

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

    base_path = "data/filtered_records_new_"
    trainFile = base_path + label_column + "_train.tsv"
    devFile = base_path + label_column + "_val.tsv"
    testFile = base_path + label_column + "_test.tsv"

    accuracies = []
    for iteration in range(repeat):
        ER = EmbeddingReader(repFile,MAX_NUM_WORDS)
        vocab = set(ER.embeddings_index.keys())
        sd = SpreadsheetData(trainFile, devFile, testFile, text_column, label_column, vocab,
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
