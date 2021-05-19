# expX
Darkspace document classifier & Detection method classifier
This work is developed by [Xiangci Li](https://github.com/jacklxc) (lixiangci8@gmail.com) on top of Dr. Gully Burns's code. Read the [paper](https://doi.org/10.1093/database/baz034) for more details.
## Requirements
* Tensorflow (1.9.0)
* Keras (GPU version, 2.2.2. Non-GPU version should also work.)
* h5py (2.7.1)
* Numpy (1.14.3)
* Pandas (0.23.0)
* tqdm (4.26.0)

## Embeddings
* GloVe 50, 100, 200, 300 dimensions (http://nlp.stanford.edu/data/glove.6B.zip)
* bio_GloVe 50, 100, 300, 1024 dimensions
* FastText 300 dimensions (https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip)
* bio_FastText 100 dimensions 

## Darkspace document classifier
This is a document-level classifier for Darkspace dataset, whose binary label is showing whether a document mentioned interaction method or not.

To replicate the training results, run
```
$ cd darkspace
$ python darkspace_classification.py -n EMBEDDING_NAME -d DATA_PATH -e EMBEDDING_PATH -v VOCAB_FILE
```
`EMBEDDING_NAME` refers to the name of run files in `run/EMBEDDING_NAME_run.tsv` format. You can change the training settings in the run files.
`DATA_PATH` refers to the path that the dataset is stored. 
`EMBEDDING_PATH` refers to the path that all the pretrained word embeddings are stored.
`VOCAB_FILE` refers to the file that the vocabulary file that used when training word embeddings.

## Detection method classifier
This is a paragraph-level classifier for INTACT dataset.

To replicate the training results, run
```
$ cd detection_method
$ python detection_method_classification.py -n EMBEDDING_NAME -d DATA_FILE -e EMBEDDING_PATH -v VOCAB_FILE
```
`EMBEDDING_NAME` refers to the name of run files in `run/EMBEDDING_NAME_run.tsv` format. You can change the training settings in the run files.
`DATA_FILE` refers to the dataset file. 
`EMBEDDING_PATH` refers to the path that all the pretrained word embeddings are stored.
`VOCAB_FILE` refers to the file that the vocabulary file that used when training word embeddings.

### Note
The only major difference in _split.py is it takes pre-splitted train, dev, test data, while the original code splits train and test during execution.

## Cite our paper
Please use the following BibTeX citation to cite our paper:
```
@article{burns2019building,
  title={Building deep learning models for evidence classification from the open access biomedical literature},
  author={Burns, Gully A and Li, Xiangci and Peng, Nanyun},
  journal={Database},
  volume={2019},
  year={2019},
  publisher={Oxford Academic}
}
```
