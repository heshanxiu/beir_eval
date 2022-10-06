import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util, models, evaluation, losses, InputExample
from sentence_transformers import SentenceTransformer
from sbert import SentenceTransformerA
import logging
from datetime import datetime
import gzip
import os
import tarfile
from collections import defaultdict
from torch.utils.data import IterableDataset
import tqdm
from torch.utils.data import Dataset
import random
import pickle
import argparse
import loss
import model
import evaluator
from data import MSMARCODataset

model_name = "output/train_adv_bi-encoder-mnrl-distilbert-base-uncased-margin_3.0-lambda0.01-2022-10-02_20-54-54/gen_iter_29/0_TransformerClassifier" #"../../sentence-transformers/examples/training/ms_marco/output/train_bi-encoder-mnrl-distilbert-base-uncased-margin_3.0-2022-09-15_06-13-41"
data_folder = '../../msmarco'

word_embedding_model = model.TransformerClassifier(model_name, max_seq_length=180)
dim = word_embedding_model.get_word_embedding_dimension()
discrim = model.Discriminator(in_features=dim, out_features = 512, num_layers = 2)
model = SentenceTransformerA(modules=[word_embedding_model, discrim])

for n in [1,2,3,4,5]:
    sample_file = f"samples_robust04_f{n}_bm25_tops_for_sbert.json" #'samples_dev_bm25_tops_for_sbert.json'
    eval_samples = json.loads(open(os.path.join(data_folder, sample_file)).readline())
    eval = evaluator.RerankingEvaluator(eval_samples, batch_size=2)

    print(f"{n}th fold:")
    print(eval(model))