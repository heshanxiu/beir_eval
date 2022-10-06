import sys
import torch 
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util, models, evaluation, losses, InputExample
from sentence_transformers import SentenceTransformer
from sbert import SentenceTransformerA
from datetime import datetime
from collections import defaultdict
from torch.utils.data import IterableDataset
import tqdm
from torch.utils.data import Dataset
import numpy as np
import pytrec_eval
import model
from sentence_transformers.util import cos_sim
all_mrr_scores = []
model_name = "output/train_adv_bi-encoder-mnrl-distilbert-base-uncased-margin_3.0-lambda0.1-uniq0.05-2022-10-06_08-02-47/gen_iter_16/0_TransformerClassifier" 

mrr_at_k = 10
dataset = "robust"
if dataset == "clueweb":
    data_folder = '../../clueweb'
else:
    data_folder = '../../trec45'
qrels = defaultdict(dict)
if dataset == "clueweb":
    qrelf = "/home/ec2-user/efs/clueweb/qrels.clueweb09b.txt"
else:
    qrelf = "/home/ec2-user/efs/trec45/qrels.txt"
with open(qrelf) as f: 
    for line in f:
        qid, _, did, rel = line.strip().split(" ")
        if int(rel) > 0:
            qrels[qid][did] = int(rel)

fname = [1,2,3,4,5]

queries = dict()
collection = dict()
if dataset == "clueweb":
    with open("/home/ec2-user/efs/clueweb/queries_proc.tsv") as f:
        for line in f:
            _, qid, qtext = line.strip().split("\t")
            queries[qid] = qtext

    with open("/home/ec2-user/efs/clueweb/documents_proc_top150.tsv") as f:
        for line in f:
            _, did, dtext = line.strip().split("\t")
            collection[did] = dtext

else:
    with open("/home/ec2-user/efs/trec45/bm25.test.run.top150.tsv") as f:
        for line in f:
            qid, did, qtext, dtext = line.strip().split("\t")
            queries[qid] = qtext
            collection[did] = dtext




run = defaultdict(list)

if dataset == 'clueweb':
    runf = "/home/ec2-user/efs/clueweb/bm25.f%d.test.run.top150"
else:
    runf = "/home/ec2-user/efs/trec45/bm25.f%d.test.run.top150"


for fn in fname:
    with open(runf %fn) as f:
        for line in f:
            qid, _, did, _, _, _ = line.strip().split(" ")
            run[qid].append(did)


word_embedding_model = model.TransformerClassifier(model_name, max_seq_length=500)
dim = word_embedding_model.get_word_embedding_dimension()
discrim = model.Discriminator(in_features=dim, out_features = 512, num_layers = 2)
model = SentenceTransformerA(modules=[word_embedding_model, discrim])
model.eval()

batch_size = 16

run_scores = defaultdict(dict)
docs = []
with torch.no_grad():
    for qid in tqdm.tqdm(run):
        query_embs = model.encode([queries[qid]],
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=False)

        docs_embs = model.encode([collection[did] for did in run[qid]],
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=False)

    #Compute scores
        pred_scores = cos_sim(query_embs, docs_embs)
        
        if len(pred_scores.shape) > 1:
            pred_scores = pred_scores[0]

        for did, score in zip(run[qid], pred_scores):
            run_scores[qid][did] = score.tolist()

        '''
        pred_scores_argsort = torch.argsort(-pred_scores)  #Sort in decreasing order
        #Compute MRR score
        mrr_score = 0
        for rank, index in enumerate(pred_scores_argsort[0:mrr_at_k]):
            if run[qid][index] in qrels[qid] and qrels[qid][run[qid][index]] > 0:
                mrr_score = 1 / (rank+1)
                break
        all_mrr_scores.append(mrr_score)
        '''

metrics = ["ndcg_cut_20", "P_20"]
for VALIDATION_METRIC in metrics:
    trec_eval = pytrec_eval.RelevanceEvaluator(qrels, {VALIDATION_METRIC})
    eval_scores = trec_eval.evaluate(run_scores)
    print(VALIDATION_METRIC, np.mean([d[VALIDATION_METRIC] for d in eval_scores.values()]))
    #print(len(eval_scores), eval_scores)
#print("mrr", np.mean(all_mrr_scores))