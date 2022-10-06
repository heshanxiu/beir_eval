"""
This examples show how to train a Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, that were mined using different dense embedding methods and lexical search methods.
Each positive and negative passage comes with a score from a Cross-Encoder. This allows denoising, i.e. removing false negative
passages that are actually relevant for the query.

With a distilbert-base-uncased model, it should achieve a performance of about 33.79 MRR@10 on the MSMARCO Passages Dev-Corpus

Running this script:
python train_bi-encoder-v3.py
"""
import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util, models, evaluation, losses, InputExample
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

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--model_name", required=False)
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--epochs", default=10, type=int)

parser.add_argument("--negs_to_use", default=None, help="From which systems should negatives be used? Multiple systems seperated by comma. None = all")
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=5, type=int)
parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
parser.add_argument("--use_all_queries", default=False, action="store_true")
parser.add_argument("--ce_score_margin", default=3.0, type=float)
parser.add_argument("--lambda_uni", default=0.01,type=float,help="the weight for kl loss")
parser.add_argument("--uni_q", default=0.01,type=float,help="the weight for query uniformity")
args = parser.parse_args()

print(args)

# The  model we want to fine-tune
model_name = args.model_name

train_batch_size = args.train_batch_size           #Increasing the train batch size improves the model performance, but requires more GPU memory
max_seq_length = args.max_seq_length            #Max length for passages. Increasing it, requires more GPU memory
ce_score_margin = args.ce_score_margin             #Margin for the CrossEncoder score between negative and positive passages
num_negs_per_system = args.num_negs_per_system         # We used different systems to mine hard negatives. Number of hard negatives to add from each system
num_epochs = args.epochs                 # Number of epochs we want to train

# Load our embedding model
logging.info("Create new SBERT model")
word_embedding_model = model.TransformerClassifier(model_name, max_seq_length=max_seq_length)
dim = word_embedding_model.get_word_embedding_dimension()
discrim = model.Discriminator(in_features=dim, out_features = 512, num_layers = 2)
model = SentenceTransformerA(modules=[word_embedding_model, discrim])
model_save_path = 'output/train_adv_bi-encoder-mnrl-{}-margin_{:.1f}-lambda{}-uniq{}-{}'.format(model_name.replace("/", "-"), ce_score_margin, args.lambda_uni,args.uni_q, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

### Now we read the MS Marco dataset
data_folder = '../../msmarco'

#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages
collection_filepath = os.path.join(data_folder, 'collection.tsv')
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download collection.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

logging.info("Read corpus: collection.tsv")
with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        pid = int(pid)
        corpus[pid] = passage


### Read the train queries, store in queries dict
queries = {}        #dict in the format: query_id -> query. Stores all training queries
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)


with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        qid = int(qid)
        queries[qid] = query


# Load a dict (qid, pid) -> ce_score that maps query-ids (qid) and paragraph-ids (pid)
# to the CrossEncoder score computed by the cross-encoder/ms-marco-MiniLM-L-6-v2 model
ce_scores_file = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')
if not os.path.exists(ce_scores_file):
    logging.info("Download cross-encoder scores file")
    util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz', ce_scores_file)

logging.info("Load CrossEncoder scores dict")
with gzip.open(ce_scores_file, 'rb') as fIn:
    ce_scores = pickle.load(fIn)

# As training data we use hard-negatives that have been mined using various systems
hard_negatives_filepath = os.path.join(data_folder, 'msmarco-hard-negatives.jsonl.gz')
if not os.path.exists(hard_negatives_filepath):
    logging.info("Download cross-encoder scores file")
    util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz', hard_negatives_filepath)


logging.info("Read hard negatives train file")
train_queries = {}
negs_to_use = None
with gzip.open(hard_negatives_filepath, 'rt') as fIn:
    for line in tqdm.tqdm(fIn):
        data = json.loads(line)

        #Get the positive passage ids
        qid = data['qid']
        pos_pids = data['pos']

        if len(pos_pids) == 0:  #Skip entries without positives passages
            continue

        pos_min_ce_score = min([ce_scores[qid][pid] for pid in data['pos']])
        ce_score_threshold = pos_min_ce_score - ce_score_margin

        #Get the hard negatives
        neg_pids = set()
        if negs_to_use is None:
            if args.negs_to_use is not None:    #Use specific system for negatives
                negs_to_use = args.negs_to_use.split(",")
            else:   #Use all systems
                negs_to_use = list(data['neg'].keys())
            logging.info("Using negatives from the following systems: {}".format(", ".join(negs_to_use)))

        for system_name in negs_to_use:
            if system_name not in data['neg']:
                continue

            system_negs = data['neg'][system_name]
            negs_added = 0
            for pid in system_negs:
                if ce_scores[qid][pid] > ce_score_threshold:
                    continue

                if pid not in neg_pids:
                    neg_pids.add(pid)
                    negs_added += 1
                    if negs_added >= num_negs_per_system:
                        break

        if args.use_all_queries or (len(pos_pids) > 0 and len(neg_pids) > 0):
            train_queries[data['qid']] = {'qid': data['qid'], 'query': queries[data['qid']], 'pos': pos_pids, 'neg': neg_pids}

del ce_scores

logging.info("Train queries: {}".format(len(train_queries)))

# load eval samples"
eval_samples = json.loads(open(os.path.join(data_folder, 'samples_robust04_f1_bm25_tops_for_sbert.json')).readline())


# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(train_queries, corpus=corpus)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = loss.MultipleNegativesRankingGeneratorLoss(model=model, lambda_uni = args.lambda_uni, uni_q = args.uni_q)
discrim_loss = loss.CrossEntropyDiscriminator(model=model)

evaluator = evaluator.RerankingEvaluator(eval_samples)
lr_decay = 0.95
lr_dis = 5e-3
lr_gen = 2e-5
for i in range(30):
    os.makedirs(f"{model_save_path}/dis_iter_{i}", exist_ok=True)

    # Train discriminator
    for param in model[0].parameters(): #freeze BERT parameters
        param.requires_grad = False

    for param in model[1].parameters(): # unfreeze discriminater
        param.requires_grad = True

    model.fit(train_objectives=[(train_dataloader, discrim_loss)],
            epochs=1,
            steps_per_epoch = 1000,
            warmup_steps=0,
            use_amp=True,
            checkpoint_path=f"{model_save_path}/dis_iter_{i}",
            checkpoint_save_steps= 1000,
            optimizer_params = {'lr':lr_dis, 'params': model[1].named_parameters()},
            evaluator=evaluator, 
            evaluation_steps=1000)
    
    logging.info(f"finish {i} iteration discrimator training...")
    # Save model
    model.save(f"{model_save_path}/dis_iter_{i}")
    

    # Train the ranking
    os.makedirs(f"{model_save_path}/gen_iter_{i}", exist_ok=True)
    for param in model[0].parameters():
        param.requires_grad = True

    for param in model[1].parameters():
        param.requires_grad = False


    model.fit(train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=args.warmup_steps,
            use_amp=True,
            checkpoint_path=f"{model_save_path}/gen_iter_{i}",
            checkpoint_save_steps=2000,
            optimizer_params = {'lr':lr_gen, 'params': model[0].named_parameters()},
            evaluator=evaluator, 
            evaluation_steps=2000
            )

    logging.info(f"finish {i} iteration ranker training...")
    model.save(f"{model_save_path}/gen_iter_{i}")

    lr_dis = lr_dis * lr_decay
    lr_gen = lr_gen * lr_decay

