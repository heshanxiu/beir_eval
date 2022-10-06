import torch
from torch import nn, Tensor
from typing import Iterable, Dict
import torch.nn.functional as F
import logging
from sentence_transformers import util, SentenceTransformer

class UNIFORM: 
    def __call__(self, x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


class MultipleNegativesRankingGeneratorLoss(nn.Module):
    
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct = util.cos_sim, lambda_uni = 0.01, uni_q = 0.0):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(MultipleNegativesRankingGeneratorLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.lambda_uni = lambda_uni
        self.kl_criterion = nn.KLDivLoss(reduction="batchmean")
        self.uni_q = uni_q
        self.uni = UNIFORM()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        features = [self.model(sentence_feature) for sentence_feature in sentence_features]
        reps = [feat['sentence_embedding'] for feat in features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])

        class_query = F.log_softmax(features[0]['class_pred'], dim = 1)
        class_pos = F.log_softmax(features[1]['class_pred'], dim = 1)
        class_neg = F.log_softmax(features[2]['class_pred'], dim = 1)


        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        overall_loss = self.cross_entropy_loss(scores, labels)
        class_pred = torch.cat([class_query, class_pos, class_neg], dim = 0)
        class_targets = torch.ones_like(class_pred) * 0.5 
        
        assert class_targets.shape[1] == 2
        klclass = self.lambda_uni * self.kl_criterion(class_pred, class_targets)
        #logging.info(f"marginmse: {overall_loss}, klclass: {klclass}")
        overall_loss +=  self.lambda_uni * klclass
        

        if self.uni_q > 0:
            uni_q_loss = self.uni(torch.nn.functional.normalize(embeddings_a,dim=1))
            overall_loss +=  self.uni_q * uni_q_loss
           

        return overall_loss

    def get_config_dict(self):
        return {'scale': self.scale, 'lambda_uni': self.lambda_uni, 'similarity_fct': self.similarity_fct.__name__}


class CrossEntropyDiscriminator(nn.Module):
    """
    Compute the MSE loss between the |sim(Query, Pos) - sim(Query, Neg)| and |gold_sim(Q, Pos) - gold_sim(Query, Neg)|
    By default, sim() is the dot-product
    For more details, please refer to https://arxiv.org/abs/2010.02666
    """
    def __init__(self, model):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use
        """
        super(CrossEntropyDiscriminator, self).__init__()
        self.model = model
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        
        features = [self.model(sentence_feature) for sentence_feature in sentence_features]
        
        class_query = features[0]['class_pred']
        class_pos = features[1]['class_pred']
        class_neg = features[2]['class_pred']

        class_pred = torch.cat([class_query, class_pos, class_neg], dim = 0)
        class_target = torch.cat([torch.zeros(class_query.shape[0],dtype=torch.long), torch.ones(class_query.shape[0] * 2,dtype=torch.long)]).to(class_pred.device)
        # query is class 0, doc is class 1
        overall_loss = self.loss_fct(class_pred, class_target)
        #logging.info(f"classification loss: {overall_loss}")

        return overall_loss
