from torch import nn
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, AutoConfig, T5Config
import json
from typing import List, Dict, Optional, Union, Tuple
import os
import torch
from torch import Tensor
import torch.nn.functional as F



class TransformerClassifier(nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.

    :param model_name_or_path: Huggingface models name (https://huggingface.co/models)
    :param max_seq_length: Truncate any inputs longer than max_seq_length
    :param model_args: Arguments (key, value pairs) passed to the Huggingface Transformers model
    :param cache_dir: Cache dir for Huggingface Transformers to store/load models
    :param tokenizer_args: Arguments (key, value pairs) passed to the Huggingface Tokenizer model
    :param do_lower_case: If true, lowercases the input (independent if the model is cased or not)
    :param tokenizer_name_or_path: Name or path of the tokenizer. When None, then model_name_or_path is used
    """
    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = None,
                 model_args: Dict = {}, cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, do_lower_case: bool = False,
                 tokenizer_name_or_path : str = None):
        super(TransformerClassifier, self).__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case

        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        self._load_model(model_name_or_path, config, cache_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path, cache_dir=cache_dir, **tokenizer_args)

        #No max_seq_length set. Try to infer from model
        if max_seq_length is None:
            if hasattr(self.auto_model, "config") and hasattr(self.auto_model.config, "max_position_embeddings") and hasattr(self.tokenizer, "model_max_length"):
                max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length

        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__


    def _load_model(self, model_name_or_path, config, cache_dir):
        """Loads the transformer model"""
        if isinstance(config, T5Config):
            self._load_t5_model(model_name_or_path, config, cache_dir)
        else:
            self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)

    def _load_t5_model(self, model_name_or_path, config, cache_dir):
        """Loads the encoder model from T5"""
        from transformers import T5EncoderModel
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = T5EncoderModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)

    def __repr__(self):
        return "Transformer({}) with Transformer model: {} ".format(self.get_config_dict(), self.auto_model.__class__.__name__)

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})
        
        # mean pooling of token embeddings
        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)

        sum_mask = torch.clamp(sum_mask, min=1e-9)

        output_vectors.append(sum_embeddings / sum_mask)
        output_vector = torch.cat(output_vectors, 1)

        features.update({'sentence_embedding': output_vector})
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output['text_keys'] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output['text_keys'].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        #strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        #Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(self.tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length))
        return output


    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return TransformerClassifier(model_name_or_path=input_path, **config)


class Discriminator(nn.Module):
    def __init__(self, in_features=30522, out_features=256, num_layers=2, dropout=0.1):
        super(Discriminator, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        hidden_layers = []
        for i in range(num_layers):
            if i == 0:
                input_dim = in_features
            else:
                input_dim = out_features
            hidden_layers.append(nn.Sequential(nn.Linear(input_dim, out_features), nn.GELU(), nn.Dropout(dropout)))

        hidden_layers.append(nn.Linear(out_features, 2))
        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, features: Dict[str, Tensor]):
        x = torch.nn.functional.normalize(features['sentence_embedding'], dim = 1)
        
        for i in range(self.num_layers + 1):
            x = self.hidden_layers[i](x)

        features.update({'class_pred': x})
        return features
    
    def get_sentence_embedding_dimension(self) -> int:
        return self.out_features

    def get_config_dict(self):
        return {'in_features': self.in_features, 'out_features': self.out_features, 'num_layers': self.num_layers}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    def __repr__(self):
        return "Discriminator({})".format(self.get_config_dict())

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        model = Discriminator(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))
        return model