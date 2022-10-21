#!/usr/bin/env python
# coding: utf-8

import json

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from transformers import AutoTokenizer
import model
from sbert import SentenceTransformerA
import os 
from typing import List, Dict
import numpy as np
import sys

class BEIRSbertModel:
    def __init__(self, model, tokenizer, max_length=256):
        self.max_length = max_length
        self.model = model

    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        X = self.model.encode(queries)
        return X

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + ' ' + doc["text"]).strip() for doc in corpus]
        return self.model.encode(sentences)


# set the dir for trained weights
# NOTE: this version only works for max agg in SPLADE, so the two directories below !
# If you want to use old weights ("../weights/flops_best" and "../weights/flops_efficient") for BEIR benchmark,
# change the SPLADE aggregation in SPLADE forward in models.py
model_type_or_dir = sys.argv[1]
outfile = sys.argv[2]

# loading model and tokenizer:
word_embedding_model = model.TransformerClassifier(model_type_or_dir, max_seq_length=500)
dim = word_embedding_model.get_word_embedding_dimension()
discrim = model.Discriminator(in_features=dim, out_features = 512, num_layers = 2)
model = SentenceTransformerA(modules=[word_embedding_model, discrim])
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
beir_model = BEIRSbertModel(model, tokenizer)
all_results = dict()

for dataset in ["trec-covid", "arguana", "dbpedia-entity", "scidocs", "scifact", "webis-touche2020"]: #, "nfcorpus", "quora", "climate-fever", "fiqa", "fever", "hotpotqa", "nq"]:
    print("start:", dataset)
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    
    out_dir = "/home/ec2-user/efs/beir_dataset/{}/{}".format(dataset, dataset)
    if not os.path.exists(out_dir):
        data_path = util.download_and_unzip(url, "/home/ec2-user/efs/beir_dataset/{}".format(dataset))
    corpus, queries, qrels = GenericDataLoader(data_folder=out_dir).load(split="test")
    dres = DRES(beir_model)
    retriever = EvaluateRetrieval(dres, score_function="cos_sim")
    results = retriever.retrieve(corpus, queries)
    ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, results, [1, 10, 100, 1000])
    results2 = EvaluateRetrieval.evaluate_custom(qrels, results, [1, 10, 100, 1000], metric="r_cap")
    res = {"NDCG@10": ndcg["NDCG@10"],
           "Recall@100": recall["Recall@100"],
           "R_cap@100": results2["R_cap@100"]}
    print("res for {}:".format(dataset), res)
    all_results[dataset] = res
json.dump(all_results, open(outfile, "w"))
