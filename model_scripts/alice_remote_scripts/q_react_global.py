
# Importing stock libraries
import os
import sys
import warnings
import random
from typing import Dict, List, Optional
from types import SimpleNamespace
from itertools import chain, combinations
import numpy as np
import pandas as pd
import re
import time
from tqdm import tqdm
import h5py

# Importing the ML and transformers libraries
import torch
from torch import cuda
from torch import nn
from torchmetrics import MeanSquaredError
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler

from sentence_transformers import SentenceTransformer, models as sentence_models, losses, evaluation, datasets, InputExample

#from datasets import Dataset
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification, RobertaConfig, Trainer, TrainingArguments, AutoModel, AutoTokenizer, EarlyStoppingCallback
#require sentencepiece

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
import wandb

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def nested_detach(tensors):
    """
    Detach nested tensor in-place.
    """
    if isinstance(tensors, torch.Tensor):
        return tensors.detach()
    elif isinstance(tensors, dict):
        return {k: nested_detach(v) for k, v in tensors.items()}
    elif isinstance(tensors, list):
        return [nested_detach(v) for v in tensors]
    elif isinstance(tensors, tuple):
        return tuple(nested_detach(v) for v in tensors)
    return tensors


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    #print(f"GPU memory occupied: {info.used//1024**2} MB.")
    return info.used//1024**2

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def parse_liststr_to_floatarray(liststr):
    return np.array(list(map(float, liststr[1:-1].split(","))))

def normalize(arr):
    return arr / np.sum(arr) #/ np.linalg.norm(arr)

def preprocess_article_text(article_bodies):
    #Remove multiple newlines
    article_bodies = article_bodies.apply(lambda s: re.sub(r'(\n+\s*\n+)+' , r'\n', s))
    
    #Remove multiple whitespaces, while keeping single whitespaces
    article_bodies = article_bodies.apply(lambda s: re.sub(r'(\s\s+)' , r' ', s))
        
    #Remove newlines:
    article_bodies = article_bodies.apply(lambda s: re.sub(r'\n' , r' ', s))
    return article_bodies

def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

def non_increasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def monotonic(L):
    return non_decreasing(L) or non_increasing(L)


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() #expands the size to (batch_size, sequence_length, hidden_size), repeating the attention mask along the new dimension.
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def cls_pooling(token_embeddings, attention_mask):
    #return token_embeddings[:, 0, :]*attention_mask[:, 0].unsqueeze(-1).expand(token_embeddings[:, 0, :].size())
    return token_embeddings[:, 0, :]

def max_pooling(token_embeddings, attention_mask):
    print("Pooling attn mask in: ", attention_mask.shape, token_embeddings.size())
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() #expands the size to (batch_size, sequence_length, hidden_size), repeating the attention mask along the new dimension.
    ###
    #for t_i in range(input_mask_expanded.shape[1]):
    #    np.savetxt(os.path.join("./output_and_results/pooling_attn_masks/", "poolingMask"+str(t_i)+".csv"), 
    #        input_mask_expanded[:, t_i, :].cpu().numpy(), delimiter=";")
    #input_mask_expanded.cuda()
    ###
    
    
    return torch.max(token_embeddings * input_mask_expanded, 1)[0]

class ProbDistribCrossEntropyLoss(nn.Module):
    def __init__(self, eps=1e-15):
        super().__init__()
        self.eps = eps
        
    def forward(self, pred, target, keep_batch_dim=False):
        # Clamping values for stability
        pred = pred.clamp(min=self.eps, max=1. - self.eps)
        target = target.clamp(min=self.eps, max=1. - self.eps)

        # Calculating cross entropy
        cross_entropy = -torch.sum(target * torch.log(pred), dim=1)
        if keep_batch_dim:
            return cross_entropy
        else:
            return cross_entropy.mean()
        
class NonLogKLDivLoss(nn.Module):
    def __init__(self, eps=1e-15):
        super().__init__()
        self.eps = eps
        
    def forward(self, pred, target, keep_batch_dim=False):
        # Clamping values for stability
        pred = pred.clamp(min=self.eps, max=1. - self.eps)
        target = target.clamp(min=self.eps, max=1. - self.eps)

        # Calculating cross entropy
        reduction = 'none' if keep_batch_dim else 'batchmean'
        kl_div =  nn.functional.kl_div(pred.log(), target.log(), log_target=True, reduction=reduction)
        
        if reduction == 'none':
            return torch.sum(kl_div, dim=-1)
        elif reduction == 'batchmean':
            return kl_div
        
        
class JSDivLoss(nn.Module):
    def __init__(self, eps=1e-15):
        super().__init__()
        self.eps = eps
        
    def forward(self, pred, target, keep_batch_dim=False):
        '''
        Pred are target are expected to be just normalized (non-log) probability vectors in form of tensors of shape (batch_size, num_classes)
        '''
        
        # Clamping values for stability
        pred = pred.clamp(min=self.eps, max=1. - self.eps)
        target = target.clamp(min=self.eps, max=1. - self.eps)

        # Computing the average distribution
        avg_dist = 0.5 * (pred + target)

        # Calculating JS Divergence
        reduction = 'none' if keep_batch_dim else 'batchmean'
        js_div = 0.5 * (nn.functional.kl_div(pred.log(), avg_dist.log(), log_target=True,reduction=reduction) +
                        nn.functional.kl_div(target.log(), avg_dist.log(), log_target=True,reduction=reduction))
        
        if reduction == 'none':
            return torch.sum(js_div, dim=-1)
        elif reduction == 'batchmean':
            return js_div


class CustomMSELossWithDispersion(nn.Module):
    def __init__(self, dispersion_weight=0.04):
        self.dispersion_weight = dispersion_weight
        super().__init__()
        
    def forward(self, pred, target, keep_batch_dim=False):
        # Calculating cross entropy
        
        mse = nn.functional.mse_loss(pred, target, reduction='none')
        dispersion_term = - self.dispersion_weight * torch.mean(torch.abs(pred - target) * torch.abs(pred.t() - target))
        mse += dispersion_term

        if keep_batch_dim:
            return mse
        else:
            return mse.mean()  

class CustomMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, keep_batch_dim=False):
        # Calculating cross entropy
        
        mse = nn.functional.mse_loss(pred, target, reduction='none')
        
        if keep_batch_dim:
            return mse
        else:
            return mse.mean()    

class CustomMAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, keep_batch_dim=False):
        # Calculating cross entropy
        
        mae = nn.functional.mae_loss(pred, target, reduction='none')
        
        if keep_batch_dim:
            return mae
        else:
            return mae.mean()    


def compute_metrics(EvalPred):
    preds, labels = EvalPred.predictions, EvalPred.label_ids

    print(preds, labels)
    print(preds.shape, labels.shape)
    
    preds, labels = torch.from_numpy(preds), torch.from_numpy(labels)
    loss_fct = CustomMSELoss()
    mse = loss_fct(preds, labels).item()
    loss_fct = CustomMAELoss()
    mae = loss_fct(preds, labels).item()
    return {"mse_loss": mse, "mae_loss": mae}

def compute_simple_regression_metrics(EvalPred):
    preds, labels = EvalPred
    preds = preds.squeeze()
    labels = labels.squeeze()
    
    #print(preds, labels)
    #print(preds.shape, labels.shape)
    
    mse_loss = nn.functional.mse_loss(preds, labels).item()
    mae_loss = nn.functional.l1_loss(preds, labels).item()
    print("Finished computing metrics")
    return {"mse_loss": mse_loss, "mae_loss": mae_loss}

class PairwissDistanceLoss(nn.Module):
    def __init__(self, all_pairs_in_batch, distance, distance_loss):
        '''
        If all_batch_pairs is True:
            - Pairs are constructed between all samples in the batch
        If all_batch_pairs is False:
            - Neighboring samples i and i+1 are asummed to be a pair
        '''
        super().__init__()
        assert all_pairs_in_batch in [True, False]
        
        self.all_pairs_in_batch = all_pairs_in_batch
        self.distance = distance
        self.distance_loss = distance_loss
        

    def forward(self, pred, target):
        assert pred.shape == target.shape
        
        if self.all_pairs_in_batch:
            pair_indices = list(combinations(range(pred.shape[0]), 2))
            pred_distances = [self.distance(pred[[idx0,], ...], pred[[idx1,], ...]).item() for idx0, idx1 in pair_indices]
            target_distances =  [self.distance(target[[idx0,], ...], target[[idx1,], ...]).item() for idx0, idx1 in pair_indices]
        else:
            pred_distances = self.distance(pred[:-1, ...], pred[1:, ...], keep_batch_dim=True)
            #pred_distances = [self.distance(pred[[i,], ...], pred[[i+1,], ...]).item() for i in range(pred.shape[0]-1)]
            target_distances = self.distance(target[:-1, ...], target[1:, ...], keep_batch_dim=True)
            #target_distances = [self.distance(target[[i,], ...], target[[i+1,], ...]).item() for i in range(target.shape[0]-1)]
            return self.distance_loss(pred_distances, target_distances)

        #THe following unsqueeze is needed because the batch dimension should be the each pair is a sample in this context
        return self.distance_loss(torch.tensor(pred_distances).unsqueeze(-1), torch.tensor(target_distances).unsqueeze(-1))

  
class CustomArticleSentencesDataset(Dataset):
    def __init__(self, article_df, feature_label, tokenizer, max_source_len, device, target_type="mean_score", comment_df=None):
        assert feature_label in ["engagement", "sentiment"], "Feature label should be 'engagement' or 'sentiment'"
        
        self.device = device
        self.feature_label = feature_label
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len

        self.article_ids = np.unique(article_df["article_id"])
        self.num_articles = len(self.article_ids)
        
        self.tokenizer = tokenizer
        self.input_texts = [".".join([subtxt for subtxt in c_art_body.split(".")[:min(25, len(c_art_body.split(".")))]]) for c_art_body in article_df["body"].values]
        #self.input_texts = article_df["headline"].values
        
        
        self.feature_scores = np.array([parse_liststr_to_floatarray(liststr) for liststr in article_df[feature_label + "_scores"]])
        
        #self.input_subtexts = [[tokenizer.cls_token + " " + subtxt + " " + tokenizer.sep_token for subtxt in txt.split(".")] for txt in self.input_texts]
        self.input_texts = [tokenizer.cls_token + " " + txt + " " + tokenizer.sep_token for txt in self.input_texts]
        
        tokenizer_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=self.input_texts, 
                                                            max_length=self.max_source_len, 
                                                            padding='longest',
                                                            truncation=True,
                                                            return_tensors='pt',
                                                            )
            
            
        self.tokenized_input_subtexts = tokenizer_output["input_ids"]
        self.input_texts_attention_masks = tokenizer_output["attention_mask"]

        if feature_label == "engagement":
            num_bins = 5
        elif feature_label == "sentiment":
            num_bins = 9
        
        #Expected logits for each article
        #Index 0 of the histogram gives the hist weights. Index 1 gives the bin edges
        bin_edges = np.linspace(0.0-(1.0-0.0)/(2*(num_bins-1)), 1.0 + (1.0-0.0)/(2*(num_bins-1)), num=num_bins+1)
        print("Bin edges: ", bin_edges)
        self.targets = torch.concatenate([torch.tensor(normalize(np.histogram(parse_liststr_to_floatarray(liststr),  bins=bin_edges, density=True)[0])).reshape(1, -1) for liststr in article_df[feature_label + "_scores"]], axis=0)
        
        if target_type == "mean_score":
            self.targets = torch.tensor([torch.from_numpy(parse_liststr_to_floatarray(liststr)).mean(dtype=torch.float32) for liststr in article_df[feature_label + "_scores"]], dtype=torch.float32).reshape(-1, 1)


    def __len__(self):
        return len(self.article_ids)

    def __getitem__(self, index):
        item = {'input_texts': self.input_texts[index],
                'input_ids': self.tokenized_input_subtexts[index, :],
                'attention_mask': self.input_texts_attention_masks[index, :],
                'targets': self.targets[index, :],
                'feature_scores': self.feature_scores[index, :],
                'article_ids': self.article_ids[index], 
                #'trunc_article_body': self.trunc_art_bodies[index],
                #'comment_bodies': self.comment_bodies[index],
                }   
        
        return item
    
    def select(self, indices):
        selected_subset = torch.utils.data.Subset(self, indices)
        
        selected_subset.device = self.device
        return selected_subset
    
    def sort(self, key, reverse=False):
        sorted_subset = torch.utils.data.Subset(self, sorted(range(len(self)), key=lambda x: self[x]['subtexts_counts'], reverse=reverse))
        
        sorted_subset.device = self.device
        return sorted_subset  
    
    @staticmethod
    def sort(dataset_or_subset, key, reverse=False):
        sorted_subset = torch.utils.data.Subset(dataset_or_subset, sorted(range(len(dataset_or_subset)), key=lambda x: dataset_or_subset[x]['subtexts_counts'], reverse=reverse))
        
        sorted_subset.device = dataset_or_subset.device
        return sorted_subset
    
    def train_test_split(self, train_frac, seed):
        """Split the dataset randomly"""
        assert 0.0 < train_frac < 1.0
        
        train_size = int(train_frac*len(self))
        test_size = len(self) - train_size
        
        dataset_ids = list(range(len(self)))
        random.shuffle(dataset_ids)
        train_dataset_ids = dataset_ids[:train_size]
        test_dataset_ids = dataset_ids[train_size:]
        
        #train_dataset_ids = random.sample(range(len(self)), train_size)
        #test_dataset_ids = [ds_idx for ds_idx in range(len(self)) if ds_idx not in train_dataset_ids]
        print("Preparing subsets...")
        return self.select(train_dataset_ids), self.select(test_dataset_ids)

    def train_test_val_split(self, train_frac, val_frac, seed):
        """Split the dataset randomly"""
        assert 0.0 < train_frac < 1.0
        assert 0.0 < val_frac < 1.0
        assert train_frac + val_frac < 1.0
        
        train_size = int(train_frac*len(self))
        val_size = int(val_frac*len(self))
        test_size = len(self) - train_size - val_size
        
        dataset_ids = list(range(len(self)))
        random.shuffle(dataset_ids)
        train_dataset_ids = dataset_ids[:train_size]
        val_dataset_ids = dataset_ids[train_size:train_size+val_size]
        test_dataset_ids = dataset_ids[train_size+val_size:]
        
        print("Preparing subsets...")
        return self.select(train_dataset_ids), self.select(val_dataset_ids), self.select(test_dataset_ids)
        


class CustomContrastiveArticleSentencesDataset(Dataset):
    def __init__(self, articlesentences_dataset, device):
        self.original_dataset = articlesentences_dataset
        self.index_pairs = list(combinations(range(len(articlesentences_dataset)), 2))
        self.device = device
    
    def __len__(self):
        return len(self.index_pairs)

    def __getitem__(self, index):
        return self.original_dataset[self.index_pairs[index][0]], self.original_dataset[self.index_pairs[index][1]]
    
    def collate(self, batch):
        '''
        Chains a list (length=batch_size) of pair tuples (length=2) into a batch of size batch_size*2
        Note that it is important to not shuffle or re-arrange the elements such that pairs stay next to each other
        In loss computation, pairs are re-extracted.
        '''
        
        batch = list(chain(*batch))
        return batch
    

class CustomModelForArticleEmbedding(nn.Module):
    def __init__(self, text_embedder_model_or_hf_id, device):
        super(CustomModelForArticleEmbedding, self).__init__()
        
        #Subtext embedding
        if isinstance(text_embedder_model_or_hf_id, nn.Module):
            self.embedder_base = text_embedder_model_or_hf_id.to(device)
        elif isinstance(text_embedder_model_or_hf_id, str):    
            if "roberta" in text_embedder_model_or_hf_id:
                self.embedder_base = RobertaModel.from_pretrained(text_embedder_model_or_hf_id) #all-mpnet-base-v2, all-distilroberta-v1, all-MiniLM-L12-v2
                self.embedder_base = self.embedder_base.to(device)
            else:
                self.embedder_base = AutoModel.from_pretrained(text_embedder_model_or_hf_id) #all-mpnet-base-v2, all-distilroberta-v1, all-MiniLM-L12-v2
                self.embedder_base = self.embedder_base.to(device)
        else:
            raise ValueError("Text embedder model must be HF_model_id or nn.Module model.")
        
        self.text_embedding_size = self.embedder_base.config.hidden_size
        self.device = device

    def get_text_embedding_size(self):
        return self.text_embedding_size
    
    def forward(self, input_ids, attention_mask,targets=None, return_attention_weights=False,**kwargs):
        
        self.embedder_base.to(self.device)
        text_embeddings = self.embedder_base(input_ids=input_ids, 
                                            attention_mask=attention_mask)[0] #.last_hidden_state is equivalent to [0]
                                          
        text_embeddings = mean_pooling(text_embeddings, attention_mask) 
        # Take the CLS token of each output sequence:
        #text_embeddings = text_embeddings[:,0,:]

        #####
        #np.savetxt(os.path.join("./output_and_results/", "raw_text_embeddings.csv"), text_embeddings.cpu().detach().numpy() , delimiter=";")
        #text_embed_cosine_sim_mat = cosine_similarity(text_embeddings.cpu().detach().numpy(), text_embeddings.cpu().detach().numpy())
  
        #np.savetxt(os.path.join("./output_and_results/", "raw_text_embeds_cosine_sim_mat.csv"), text_embed_cosine_sim_mat , delimiter=";")
        #text_embeddings.to(self.device)
        #####
        
        #Normalize text embeddings across subtexts
        #     text_subtext_embeddings = nn.functional.normalize(text_subtext_embeddings, p=2, dim=1)
        #     subtext_embeddings_attention_masks = subtext_embeddings_attention_masks.unsqueeze(1).expand(-1, subtext_embeddings_attention_masks.shape[-1], -1)
        #    text_embeddings, attention_weights = self.subtext_pooler(text_subtext_embeddings, subtext_embeddings_attention_masks, return_attention=True)
        
        #elif self.subtext_pooling_mode == "gated":
        #    #Normalize text embeddings across subtexts
        #    text_subtext_embeddings = nn.functional.normalize(text_subtext_embeddings, p=2, dim=1)
        #    gate_weights = nn.functional.softmax(self.subtext_pooler(text_subtext_embeddings), dim=1) #Shape: (batch_size, num_subtexts, 1)
        #    gate_weights = gate_weights * subtext_embeddings_attention_masks.unsqueeze(-1) #Set gate weights to 0 for subtexts that are not present in the batch (i.e. padding subtexts)
        #    text_embeddings = torch.sum(text_subtext_embeddings * gate_weights, dim=1)
        
        
        #if self.subtext_pooling_mode == "attention" and return_attention_weights:
        #    return (nn.functional.normalize(text_embeddings, p=2, dim=1), attention_weights)
        #elif self.subtext_pooling_mode == "gated" and return_attention_weights:
        #    return (nn.functional.normalize(text_embeddings, p=2, dim=1), gate_weights)
        #else:
            #print(self.subtext_pooling_mode, return_attention_weights)
            #print("Neither attention weights nor gate weights are returned.")
            #return nn.functional.normalize(text_embeddings, p=2, dim=1)
        
        return text_embeddings



'''
def finetune_embedding_model(embedding_model, device):
    
    # Define the objective/loss for the task:
    #TODO: Before any iteration, pre-compute target losses (independing of embeddings/model state) -> Per iteration, pre-compute embeddings for each text. -> Backpropate difference between embeddings and targets distances
    # For embedding distance use cosine similarity. For target distance use JS-divergence. 
    # Probabiy need to normalize the distances to make them comparable. 
    embedding_loss_function = JSDivLoss(text_A_target, text_B_target)
    
    # Fine-tune the classification model on the reference comments:
    torch.set_grad_enabled(True)
    
    #embedding_model.fit(train_objectives=[(data, embedding_loss_function)],
    #            epochs=4,
    #            steps_per_epoch=None,
    #            show_progress_bar=True,
    #            )
    
    return finetuned_embedding_model
'''
    
    
#===================================================================================================
# From here on, everything is about prediction
#===================================================================================================

class CustomModelForMultinomialRegression(nn.Module):
    def __init__(self, input_embedding_size, num_reg_targets, device):
        super().__init__()
        self.input_embedding_size = input_embedding_size
        
        self.num_reg_targets = num_reg_targets
        ''' 2-layer
        self.mlp = nn.Sequential(nn.Linear(input_embedding_size, 
                                            int(input_embedding_size/2), 
                                            device=device),
                                nn.ReLU(),
                                nn.Linear(int(input_embedding_size/2),
                                            num_reg_targets,
                                            device=device),
                                )
        '''

        self.mlp = nn.Sequential(nn.Linear(input_embedding_size, 
                                            int(128), 
                                            device=device),
                                nn.ReLU(),
                                nn.Linear(int(128),
                                          int(64),
                                            device=device),
                                nn.ReLU(),
                                nn.Linear(int(64),
                                          int(32),
                                            device=device),
                                nn.ReLU(),
                                nn.Linear(int(32),
                                          int(16),
                                            device=device),
                                nn.ReLU(),
                                nn.Linear(int(16),
                                            num_reg_targets,
                                            device=device),     
                                )

        
        self.device = device

    def forward(self, input_embeddings, targets=None, **kwargs):
        self.mlp = self.mlp.to(self.device)
        x = self.mlp(input_embeddings)
        return nn.functional.softmax(x, dim=-1)

class CustomModelForSimpleRegression(CustomModelForMultinomialRegression):
    def __init__(self, input_embedding_size, device):
        super().__init__(input_embedding_size, 1, device)
    
    def forward(self, input_embeddings, targets=None, **kwargs):
        #return self.mlp(input_embeddings)
        return self.mlp(input_embeddings)


class CustomModelForEnd2EndRegression(nn.Module):
    
    def __init__(self, text_embedder, embedding_regressor, device):
        super().__init__()
        self.text_embedder = text_embedder
        self.text_embedding_size = text_embedder.get_text_embedding_size()
        
        self.reg_head = embedding_regressor
        self.config = self.text_embedder.embedder_base.config
        
        self.device = device

    def forward(self, input_ids, attention_mask, targets=None, return_attention_weights=False, **kwargs):

        #First compute the embedding of each text (from their subtexts)
        #if self.text_embedder.subtext_pooling_mode != "mean" and return_attention_weights:
        if return_attention_weights:
            x, attention_weights = self.text_embedder(input_ids, attention_mask, return_attention_weights=return_attention_weights)
            #Next compute the regression head	
            x = self.reg_head(x)
            return x, attention_weights
        else:
            x = self.text_embedder(input_ids, attention_mask)
            print("Embedder output shape:", x.shape)

            wandb.log({"Text Embeddings Cosine Similarity": wandb.plots.HeatMap(matrix_values=cosine_similarity(x.cpu().detach().numpy(), x.cpu().detach().numpy()), 
                                                                                x_labels=[str(i) for i in range(x.shape[0])],
                                                                                y_labels=[str(i) for i in range(x.shape[0])],
                                                                                show_text=False),
                                                                                })
            x.to(self.device)
            #Next compute the regression head
            x = self.reg_head(x)
            print("Regression head output shape:", x.shape)
            return x


class SklearnWrapperModule(nn.Module):
    def __init__(self, sklearn_model, input_embedding_size, num_reg_targets, loss_func, do_train, device):
        super().__init__()
        
        self.input_embedding_size = input_embedding_size
        self.num_reg_targets = num_reg_targets
        
        self.sklearn_model = sklearn_model
        self.loss_func = loss_func
        self.do_train = do_train
        self.device = device

    def forward(self, input_embeddings, targets, **kwargs):
        #MODULE CALLED VIA COMPUTE_LOSS IN TRAINER! FOR INFERENCE EXTRACT FITTED SKLEARN MODEL AND CALL PREDICT ON IT!
        preds = self.sklearn_model.predict(input_embeddings.cpu().numpy()) #For loss computation
        loss = self.loss_func(preds, targets)
        if self.do_train:
            self.sklearn_model.partial_fit(input_embeddings.cpu().numpy(), targets.cpu().numpy()) #For training
        
        return torch.tensor(loss)



class CustomEnd2EndMultinomialRegressionTrainer(Trainer):
    def __init__(self, *args, loss_func_or_wrapper,device,  **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.loss_func_or_wrapper = loss_func_or_wrapper
        
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # implement custom logic here
        outputs = model(input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        #return_attention_weights = False,
                        )
        
        inputs['targets'] = inputs['targets'].to(outputs.dtype)
        
        loss = self.loss_func_or_wrapper(outputs, inputs['targets'].to(self.device))
        
        print(outputs, " || ", inputs['targets'])
        print(outputs.dtype, " || ", inputs['targets'].dtype)
        print(loss, type(loss))
        return (loss, outputs) if return_outputs else loss
'''
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):	
        loss, logits, labels = super().prediction_step(model, inputs, False, ignore_keys)
        
        if logits is not None and labels is not None:
            wandb.log({"Predictions_targets_plot": wandb.plot.scatter(wandb.Table(data=[[x, y] for (x, y) in zip(logits.cpu().numpy(),labels.cpu().numpy())], columns = ["Predictions", "Targets"]),
                                                                        x = "Predictions", 
                                                                        y = "Targets",
                                                                        ),
                        })
        
        if prediction_loss_only:
            return (loss, None, None)
        else:
            return loss, logits, labels
'''

#================ Contrastive fine-tuning of End-2-End embedding model ====================

def contrastive_finetune_end2end_model(train_dataset, val_dataset, pair_batching_mode, model_args,  training_args, device):    
    required_model_args = ["text_embedder", "embedding_regressor", "num_reg_targets", "loss_func"]
    assert all([key in model_args.keys() for key in required_model_args]), f"Missing one or more of the required model arguments: {required_model_args}"
    assert pair_batching_mode in ["all_pairs_in_dataset", "pairs_per_batch"]
    
    if isinstance(model_args["embedding_regressor"], nn.Module):
        end2end_model = CustomModelForEnd2EndRegression(text_embedder=model_args["text_embedder"],
                                                                embedding_regressor=model_args["embedding_regressor"],
                                                                device=device)
                                                                
        if pair_batching_mode == "all_pairs_in_dataset": 
            contrastive_train_dataset = CustomContrastiveArticleSentencesDataset(train_dataset)
           #batch_collator = contrastive_train_dataset.collate
            contrastive_loss = PairwissDistanceLoss(all_pairs_in_batch=False, distance=model_args["loss_func"], distance_loss=nn.MSELoss())
        elif pair_batching_mode == "pairs_per_batch":
            contrastive_train_dataset = train_dataset
            #batch_collator = train_dataset.dynamic_batch_padding
            contrastive_loss = PairwissDistanceLoss(all_pairs_in_batch=True, distance=model_args["loss_func"], distance_loss=nn.MSELoss())
        
        #val_loader = DataLoader(val_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=data_collator ,shuffle=False)
        #validate_end2end_model(end2end_model, val_loader, 1, model_args["loss_func"], device)
        
        # define the loss & trainer and start training
        
        trainer = CustomEnd2EndMultinomialRegressionTrainer(model=end2end_model,
                                                            args=training_args,
                                                            loss_func_or_wrapper=contrastive_loss,
                                                            device=device,
                                                            train_dataset=contrastive_train_dataset,
                                                            #eval_dataset=val_dataset,
                                                            #data_collator=batch_collator,
                                                            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
                                                            #compute_metrics=compute_metrics,
                                                            )
        trainer.train()


    
#===================================================================================================
#=============== Fine-tuning of End-2-End predicton model ====================================================

def train_end2end_regression_model(train_dataset, val_dataset, data_collator,model_args,  training_args, device, full_dataset=None):    

    required_model_args = ["text_embedder", "embedding_regressor", "num_reg_targets", "loss_func"]
    assert all([key in model_args.keys() for key in required_model_args]), f"Missing one or more of the required model arguments: {required_model_args}"
    
    # Make sure we are logged in to wandb
    os.environ["WANDB_API_KEY."]="925f0f3c8de42f022a0a5a390aab9845cb5c92cf"
    #wandb.login()
    # WandB – Initialize a new run
    wandb.init(project="article_mean_score_regression")
    # save the trained model checkpoint to wandb
    #os.environ["WANDB_LOG_MODEL"]="true"
    # turn off watch to log faster
    os.environ["WANDB_WATCH"]="false"
    
    # Store some key paramters to the WandB config for later reference
    config = wandb.config          # Initialize config
    
    config.TRAIN_BATCH_SIZE = training_args.per_device_train_batch_size  
    config.VALID_BATCH_SIZE = training_args.per_device_eval_batch_size   
    config.TRAIN_EPOCHS = training_args.num_train_epochs           
    #config.VAL_EPOCHS = training_args.num_eval_epochs      
    config.LEARNING_RATE = training_args.learning_rate    
    config.SEED = training_args.seed
    
    def hyp_param_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 5, 10,40),
            }
    
    if isinstance(model_args["embedding_regressor"], nn.Module):
        end2end_model = CustomModelForEnd2EndRegression(text_embedder=model_args["text_embedder"],
                                                                embedding_regressor=model_args["embedding_regressor"],
                                                                device=device)
        
        
        
        val_loader = DataLoader(val_dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=False)
        #validate_end2end_model(end2end_model, val_loader, 1, model_args["loss_func"], device)
        
        
        # define the trainer and start training
        wandb.watch(end2end_model, log="all")
        
        
        trainer = CustomEnd2EndMultinomialRegressionTrainer(model=end2end_model,
                                                            args=training_args,
                                                            loss_func_or_wrapper=model_args["loss_func"],
                                                            device=device,
                                                            train_dataset=train_dataset,
                                                            eval_dataset=val_dataset,
                                                            #data_collator=data_collator,
                                                            #compute_metrics=compute_metrics,
                                                        )
        
        
        """
        best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize", hp_space=hyp_param_space)

        for n, v in best_run.hyperparameters.items():
            setattr(training_args, n, v)
        
        """

        trainer.train()
        
        evaluate_end2end_model(trainer.model, val_loader, model_args["loss_func"], "./output_and_results/temp_predictions.csv", device, target_type="mean_score")
        
        #------------------Other experiments that should happen after training the end2end model:-------------------------
    
        #capture_embedding_similarity(end2end_model.text_embedder, full_dataset, "./output_and_results/", "distilroberta_zero_embeddings.hdf5", device)
        #capture_embedding_similarity(end2end_model.text_embedder, train_dataset, "./embed_checkpoints/", "train_roberta_e2e_embeddings.hdf5", device)
        #capture_embedding_similarity(end2end_model.text_embedder, val_dataset, "./embed_checkpoints/", "test_roberta_e2e_embeddings.hdf5", device)
    
        
        '''
        if training_args.deepspeed:
            checkpoint_dir = os.path.join(trainer.args.output_dir, "checkpoint-final")
            trainer.deepspeed.save_checkpoint(checkpoint_dir)
            end2end_model.cpu()
            trainer.save_model(os.path.join(trainer.args.output_dir, "model-final"))
        else:
            trainer.save_model(os.path.join(trainer.args.output_dir, "model-final"))
        '''

    elif isinstance(model_args["embedding_regressor"], sklearn.base.BaseEstimator):
        
        train_loader = DataLoader(training_args.train_dataset, batch_size=len(training_args.train_dataset), shuffle=False)
        sklearn_head = GaussianProcessRegressor(kernel=RBF(), random_state=0)
    
        SklearnWrapperModule(sklearn_head, input_embedding_size, num_reg_targets, do_train=True, loss_func=loss_func)
        for (b_i, batch) in enumerate(train_loader,0):
            art_embeddings = torch.mean(batch['input_subtext_embeddings'], dim=1).numpy()
            
            scaled_art_embeddings = StandardScaler().fit_transform(art_embeddings)
            sklearn_head.fit(scaled_art_embeddings, batch['targets'].numpy())

            #inference
            pred_mean, pred_std = sklearn_head.predict(scaled_art_embeddings, return_std=True)
            #print(np.mean(pred_std, axis=0))
            
            #loss_fun = ProbDistribCrossEntropyLoss()
            #print(loss_fun(torch.tensor(pred_mean), batch['targets']))
    else:
        raise ValueError("embedding_regressor must be either a nn.Module or a sklearn.base.BaseEstimator")

    

def validate_end2end_model(model, val_loader, val_epochs, loss_fun, device, target_type):
    #compute_metrics = get_compute_metrics_func(target_mode)
    
    if target_type == "mean_score":
        running_metrics = {"mse_loss": 0.0, "mae_loss": 0.0}
    elif target_type == "score_distribution":
        running_metrics = {"kl_loss": 0.0, "js_loss": 0.0}

    for epoch in range(val_epochs):
        with torch.no_grad():
            for (b_i, batch) in enumerate(val_loader,0):
                
                targets = batch['targets'].to(device)
                outputs = model(input_ids = batch['input_ids'].to(device),
                                attention_mask = batch['attention_mask'].to(device),
                                )
                
                loss_output = loss_fun(outputs, targets)
                if target_type == "mean_score":
                    metrics_output = compute_simple_regression_metrics((outputs, targets))
                elif target_type == "score_distribution":
                    metrics_output = compute_distribution_metrics((outputs, targets))
                
                running_metrics = {k: running_metrics[k] + metrics_output[k] / (len(val_loader)*val_epochs) for k in running_metrics}

                if b_i%100==0:
                    print(f'Completed {b_i} batches out of {len(val_loader)}')

    print("Metrics: ", running_metrics)
    return running_metrics


def evaluate_end2end_model(model, eval_loader, error_fun, preds_output_filepath, device, target_type):

    if target_type == "mean_score":
        running_metrics = {"mse_loss": 0.0,
                            "mae_loss": 0.0,
                            }
    elif target_type == "score_distribution":
        running_metrics = {"kl_loss": 0.0,
                           "js_loss": 0.0,
                           }
    
    prediction_buffer = []

    with torch.no_grad():
        for (b_i, batch) in enumerate(eval_loader,0):
            batch_size = len(batch['article_ids'])
            
            targets = batch['targets'].to(device)
            
            result = model(input_ids = batch['input_ids'].to(device),
                            attention_mask = batch['attention_mask'].to(device),
                            #return_attention_weights=True,
                            )
            
            print("Result: ", result)
            
            if isinstance(result, tuple):
                outputs, attention_weights = result
            else:
                outputs, attention_weights = result, None
            
            errors = error_fun(outputs, targets, keep_batch_dim=True).cpu().numpy()
            
            print("Errors: ", errors)
            
            if target_type == "mean_score":
                metrics_output = compute_simple_regression_metrics((outputs, targets))
            elif target_type == "score_distribution":
                metrics_output = compute_distribution_metrics((outputs, targets))
            
            running_metrics = {k: running_metrics[k] + metrics_output[k] / (len(eval_loader)*batch_size) for k in running_metrics.keys()}

            outputs = outputs.cpu().numpy()
            if attention_weights is not None:
                attention_weights = attention_weights.cpu().numpy()
            targets = targets.cpu().numpy()
            
            print("Targets: ", targets)
            print("Outputs: ", outputs)
            print("Attention Weights: ", attention_weights)
            print(metrics_output)
            
            if len(outputs.shape)==1:
                outputs = outputs[np.newaxis, :]
            if len(targets.shape)==1:
                targets = targets[np.newaxis, :]
            if attention_weights is not None and len(attention_weights.shape)==1:
                attention_weights = attention_weights[np.newaxis, :]
            
            
            batch_out_dicts = [{"article_id": batch["article_ids"][sample_idx],
                        "prediction": outputs[sample_idx, :].tolist(),
                        "target": targets[sample_idx,:].tolist(),
                        "error": errors[sample_idx].tolist(),
                        } for sample_idx in range(batch_size)]

            prediction_buffer.extend(batch_out_dicts)
            
            if b_i%100==0:
                print(f'Completed evaluation on {b_i} batches out of {len(eval_loader)}')

    print("Total Metrics: ", running_metrics)
    
    export_eval_df = pd.DataFrame(prediction_buffer)

    wandb.log({"Prediction_output_table": wandb.Table(dataframe=export_eval_df)})

    export_eval_df.to_csv(preds_output_filepath, index=False, sep=";")
    return export_eval_df

def flatten(l): #For python lists
    r =  [item.item() for sublist in l for item in sublist]
    print(r)
    return r
def load_embedder_model(pretrain_text_embedder, subtexts_aggregation_strategy, base_model_hf_id , train_dataset, art_df, cmt_df,device):
    # "SimCE" # Is "SimCE" or "TSDAE" or None or CosSim or SimCE-Head-Body or SimCE-Head-Comment
    
    assert pretrain_text_embedder in ["SimCE", "TSDAE", "CosSim", "SimCE-Head-Body", "SimCE-Head-Comment", None]

    if pretrain_text_embedder in ["SimCE", "TSDAE", "CosSim", "SimCE-Head-Body", "SimCE-Head-Comment"]:
        print("pretrain_text_embedder: ", pretrain_text_embedder)
        print("subtexts_aggregation_strategy: ", subtexts_aggregation_strategy)
        print("base_model_hf_id: ", base_model_hf_id)
    
        base_model_hf_id = 'sentence-transformers/all-distilroberta-v1'
        #base_model_hf_id = 'roberta-base'
        
        #addendum_label = "senti_reverseTrained_"
        #addendum_label = "reverseTrained_"
        addendum_label = ""

        if  pretrain_text_embedder == "TSDAE":
            model_output_dir = "./models/{}distilroberta-TSDAE-sentence-embedder/".format(addendum_label) #"./models/distilroberta-SimCE-sentence-embedder/"
        elif pretrain_text_embedder == "SimCE":
            model_output_dir = "./models/{}distilroberta-SimCE-sentence-embedder/".format(addendum_label)
        elif pretrain_text_embedder == "CosSim":
            model_output_dir = "./models/{}distilroberta-CosSim-sentence-embedder/".format(addendum_label)
        elif pretrain_text_embedder == "SimCE-Head-Body":
            model_output_dir = "./models/{}distilroberta-SimCE-Head-Body-sentence-embedder/".format(addendum_label)
        elif pretrain_text_embedder == "SimCE-Head-Comment":
            model_output_dir = "./models/{}distilroberta-SimCE-Head-Comment-sentence-embedder/".format(addendum_label)

        try:
            print("Trying to load pretrained model from {} ...")
            raise Exception("Always recompute")
            text_embedder = AutoModel.from_pretrained(model_output_dir)    
        except:
        
            text_sentence_embedder = SentenceTransformer(base_model_hf_id, device=device) #'all-mpnet-base-v2')
            
            train_article_ids = [sample['article_ids'] for sample in train_dataset]
            train_sentence_collection = [sample['input_texts'] for sample in train_dataset]
            n_sentences = len(train_sentence_collection)
            print("We have {} train sentences".format(len(train_sentence_collection)))
            
            #Truncate article bodies to max. 20 sentences
            train_sentence_bodies = []
            train_sentence_comment_bodies = []
            for art_id in train_article_ids:
                c_comment_bodies = cmt_df[cmt_df["article_id"] == art_id]["body"].values
                train_sentence_comment_bodies.append(c_comment_bodies)

                c_art_body = art_df[art_df["article_id"] == art_id]["body"].values[0]
                c_trunc_art_body = ".".join([subtxt for subtxt in c_art_body.split(".")[:min(25, len(c_art_body.split(".")))]])
                
                ### c_trunc_art_body = art_df[art_df["article_id"] == art_id]["headline"].values[0]
                
                train_sentence_bodies.append(c_trunc_art_body)         
            
            if pretrain_text_embedder == "SimCE":
                train_examples = [InputExample(texts=[c_text, c_text]) for c_text in train_sentence_collection] # Convert train sentences to sentence pairs compatible with MultipleNegativesRankingLoss
                train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
                simce_loss = losses.MultipleNegativesRankingLoss(text_sentence_embedder)
                train_loss = simce_loss
            elif pretrain_text_embedder == "TSDAE":
                train_denoising_set = datasets.DenoisingAutoEncoderDataset(train_sentence_collection)
                train_dataloader = DataLoader(train_denoising_set, shuffle=True, batch_size=8)
                denoising_loss = losses.DenoisingAutoEncoderLoss(text_sentence_embedder, decoder_name_or_path=base_model_hf_id, tie_encoder_decoder=True)
                train_loss = denoising_loss
            elif pretrain_text_embedder == "CosSim":
                train_sentence_scores = [sample['targets'] for sample in train_dataset]

                scaled_labels = [[np.abs(train_sentence_scores[i] - train_sentence_scores[j]) for j in range(i+1, n_sentences)] for i in range(n_sentences)]
                max_score_diff_scale = 2.0 / np.max(np.ravel(np.array(flatten(scaled_labels))))
                train_examples = [InputExample(texts=[train_sentence_collection[i], train_sentence_collection[j]], label=1 - (scaled_labels[i][j-i - 1] * max_score_diff_scale)) for i in range(n_sentences) for j in range(i+1, n_sentences)]
                label_collection = np.array([example.label.item() for example in train_examples])
                #Save label_collection to npy:
                np.save("./embed_checkpoints/label_collection.npy", label_collection)
                print("label colelct ", label_collection)
                #Plot histgoram of labels
                import matplotlib.pyplot as plt
                plt.hist(label_collection,density=True, bins=70)
                plt.show()

                train_examples = np.array(list(np.random.choice(train_examples, size=1000, replace=False)))
                reduced_label_collection = [example.label.item() for example in train_examples]
                print("reduced label colelct ", reduced_label_collection)
                plt.hist(reduced_label_collection,density=True,bins=70)
                plt.show()

                train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
                cos_sim_loss = losses.CosineSimilarityLoss(text_sentence_embedder)
                train_loss = cos_sim_loss
                #Batch size 16,  30 epochs (i.e. 50 epochs), lr 2e-5
            elif pretrain_text_embedder == "SimCE-Head-Body":
                train_examples = [InputExample(texts=[c_text, c_body]) for c_text, c_body in zip(train_sentence_collection, train_sentence_bodies)]
                train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
                simce_loss = losses.MultipleNegativesRankingLoss(text_sentence_embedder)
                train_loss = simce_loss
                #Batch size 8,  50 epochs (i.e. 200 epochs), lr 1e-5
            elif pretrain_text_embedder == "SimCE-Head-Comment":
                train_examples = [InputExample(texts=[train_sentence_collection[i], c_body]) for i in range(n_sentences) for c_body in train_sentence_comment_bodies[i]]
                print("We have {} train examples".format(len(train_examples)))
                train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
                simce_loss = losses.MultipleNegativesRankingLoss(text_sentence_embedder)
                train_loss = simce_loss

            text_sentence_embedder.fit([(train_dataloader, train_loss)],# train_objectives=[(train_dataloader, simce_loss)],   #
                                        epochs=50,
                                        weight_decay=0,
                                        #scheduler='constantlr',
                                        optimizer_params={'lr': 2e-5}, 
                                        show_progress_bar=True, 
                                        )
            
                                    
            text_embedder = text_sentence_embedder._first_module().auto_model
            text_embedder.save_pretrained(model_output_dir)
            
        finally:
            wrapped_text_embedder = CustomModelForArticleEmbedding(text_embedder_model_or_hf_id=text_embedder,
                                                                    device=device)
    else:
        wrapped_text_embedder = CustomModelForArticleEmbedding(text_embedder_model_or_hf_id=base_model_hf_id,
                                                                device=device)
    return wrapped_text_embedder

#-----------------------------------------------------------------------
# Tests and experiments - not core routines
#-----------------------------------------------------------------------

def capture_embedding_similarity(text_embedder, tokenized_dataset,data_output_dir, checkpoint_filename, device):
    
    eval_batch_size = 16
    data_loader = DataLoader(tokenized_dataset, batch_size=eval_batch_size ,shuffle=False)
    
    embeddings_buffer = np.zeros((len(tokenized_dataset), text_embedder.get_text_embedding_size()))
    article_id_list = []
    targets_list = []

    with torch.no_grad():
        for (b_i, batch) in enumerate(data_loader,0):
            batch_text_embeddings = text_embedder(input_ids = batch['input_ids'].to(device),
                                                    attention_mask = batch['attention_mask'].to(device),
                                                    )
            
            
            embeddings_buffer[b_i*eval_batch_size:(b_i+1)*eval_batch_size,:] = batch_text_embeddings.cpu().numpy()
            
            article_id_list.extend(batch['article_ids'])
            targets_list.append(batch['targets'].numpy())

            if b_i % 20 == 0:
                print("Completed embedding computation for {} batches out of {}".format(b_i, len(data_loader)))
    
    with h5py.File("./embed_checkpoints/"+checkpoint_filename, 'w') as hf:
                hf.create_dataset("article_ids", data=article_id_list)
                hf.create_dataset("targets", data=np.concatenate(targets_list, axis=0))
                hf.create_dataset("text_embeddings", data=embeddings_buffer)


    # Compute cosine similarity matrix between all pairs of embeddings:
    cosine_sim_mat = cosine_similarity(embeddings_buffer, embeddings_buffer)
    
    # Save cosine similarity matrix to csv file:
    np.savetxt(os.path.join(data_output_dir, "cosine_sim_mat.csv"), cosine_sim_mat, delimiter=";")
    



def main(model_output_dir, data_dirpath, HF_MODEL_ID, subtexts_aggregation_strategy, pretrain_text_embedder, target_type, deepspeed):

    # Set up non-variable parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #torch.backends.cuda.matmul.allow_tf32 = True
    #device = torch.device('cpu')
    
    print("We are using the following device:  ", device)
    MAX_SOURCE_LEN = 512
    feature_label = "sentiment" #"sentiment" #"engagement

    if target_type == "mean_score":
        num_reg_targets = 1
    elif target_type == "score_distribution":
        num_reg_targets = 5 if feature_label == "engagement"  else 9 if feature_label == "sentiment" else None
        
        
    global_seed = 22


    # Set random seeds and deterministic pytorch for reproducibility
    random.seed(global_seed)
    torch.manual_seed(global_seed) 
    np.random.seed(global_seed) 
    torch.backends.cudnn.deterministic = True

    #Prepare the dataset
    art_filepath =  data_dirpath + "r_art_stratified_annotated.csv"
    art_df = pd.read_csv(art_filepath, sep=';')
    art_df = art_df[['article_id', 'date', 'headline', 'body', 'comments_ids', 'engagement_scores', 'sentiment_scores', 'avg_engagement_score', 'avg_sentiment_score', 'std_engagement_score', 'std_sentiment_score']]
    art_df["body"] = preprocess_article_text(art_df.loc[:,"body"])

    cmt_filepath = data_dirpath + "r_cmt_stratified_annotated.csv"
    cmt_df = pd.read_csv(cmt_filepath, sep=';').sort_values(by=['article_id', 'level', 'date'], ascending=[True, True, True])
    cmt_df = cmt_df[['comment_id','article_id', 'body', 'level']]

    
    tokenizer = RobertaTokenizer.from_pretrained(HF_MODEL_ID) if "roberta" in HF_MODEL_ID else AutoTokenizer.from_pretrained(HF_MODEL_ID)
    dataset_inst = CustomArticleSentencesDataset(art_df, feature_label, tokenizer, MAX_SOURCE_LEN, device=device, target_type=target_type, comment_df=cmt_df)
    train_set, test_set = dataset_inst.train_test_split(train_frac=0.8, seed=global_seed)
    
    print(" \n Split data in to train set of size {} and test set of size {}.".format(len(train_set), len(test_set)))
    print(" \n After data initialization and split, the GPU memory usage is: ", print_gpu_utilization())

    # ------ For deepspeed, we need to load all TrainingArguments in the script before instantiating any model. ------

    '''
    contrastive_training_args = TrainingArguments(output_dir='models',
                                                #report_to="wandb",
                                                #logging_steps=5,
                                                #run_name="...", # A descriptor for the run. Typically used for wandb logging.
                                                label_names= ["targets"],
                                                remove_unused_columns=False,
                                                per_device_train_batch_size=2,
                                                #per_device_eval_batch_size=8,
                                                #gradient_accumulation_steps=1,
                                                #gradient_checkpointing=False,
                                                optim="adamw_torch",
                                                learning_rate=5e-4,
                                                num_train_epochs=8,
                                                #max_steps = 1,
                                                save_strategy="epoch",
                                                save_steps = 100,
                                                #do_eval=True,
                                                #evaluation_strategy="steps", 
                                                #eval_steps=20, #10,
                                                seed=global_seed,
                                                dataloader_pin_memory=False,
                                                deepspeed=deepspeed,
                                                )
    '''
    
    
    predict_finetune_training_args = TrainingArguments(output_dir=model_output_dir,
                                        #report_to="wandb",
                                        #logging_steps=5,
                                        #run_name="...", # A descriptor for the run. Typically used for wandb logging.
                                        label_names= ["targets"],
                                        remove_unused_columns=False,
                                        per_device_train_batch_size=16,
                                        per_device_eval_batch_size=16,
                                        #dataloader_drop_last=True, 
                                        #gradient_accumulation_steps=2,
                                        #gradient_checkpointing=True,
                                        #fp16=True,#,tf32=True,
                                        optim="adamw_torch",
                                        learning_rate=1e-5,
                                        weight_decay=0.01,
                                        num_train_epochs=120,
                                        #max_steps = 25,
                                        save_strategy="epoch",
                                        save_steps = 2,
                                        #do_eval=True,
                                        evaluation_strategy="epoch", 
                                        #eval_steps=20, #10,
                                        seed=global_seed,
                                        dataloader_pin_memory=False,
                                        deepspeed=deepspeed,
                                        )
    

    #-------Defining & Fine-tuning the subtext embedding model:-------
    
    text_embedder = load_embedder_model(pretrain_text_embedder=pretrain_text_embedder,
                                        subtexts_aggregation_strategy=subtexts_aggregation_strategy,
                                        base_model_hf_id=HF_MODEL_ID,
                                        train_dataset=train_set,
                                        art_df = art_df,
                                        cmt_df = cmt_df,
                                        device=device,
                                        ) 
    
    if target_type == "mean_score":
        embedding_regressor = CustomModelForSimpleRegression(text_embedder.get_text_embedding_size(),
                                                            device=device)
    
    elif target_type == "score_distribution":
        embedding_regressor = CustomModelForMultinomialRegression(text_embedder.get_text_embedding_size(), 
                                                                num_reg_targets, 
                                                                device=device)
        
    text_embedder = text_embedder.to(device)
    embedding_regressor = embedding_regressor.to(device)
    
    print("\n After initialization of model components (embedder and head), the GPU memory usage is: ", print_gpu_utilization())
    
    ''' Test for the embedding model
    #temp_loader = dataset_inst.get_dataloader(batch_size=4)
    temp_loader = DataLoader(train_set, batch_size=4, collate_fn=dataset_inst.dynamic_batch_padding,shuffle=True, drop_last=True)
    
    for temp_batch in tqdm(temp_loader):
        #print(temp_batch['tokenized_input_subtexts'].shape)
        #print(embedding_model(**temp_batch))
        
        tokenized_input_subtexts = temp_batch['tokenized_input_subtexts'].to(device)
        subtexts_attention_masks = temp_batch['subtexts_attention_masks'].to(device)
        
        tokenized_input_subtexts_shape = tokenized_input_subtexts.shape
        train_subtext_embeds = mean_pooling(text_embedder.embedder_base(input_ids=tokenized_input_subtexts.reshape(-1, tokenized_input_subtexts_shape[-1]), 
                                                             attention_mask=subtexts_attention_masks.reshape(-1, tokenized_input_subtexts_shape[-1]))[0], 
                                          subtexts_attention_masks.reshape(-1, tokenized_input_subtexts_shape[-1])) #.last_hidden_state is equivalent to [0]
        
        
        np.savetxt(os.path.join("./output_and_results/", "train_subtext_embeddings.csv"), train_subtext_embeds.cpu().detach().numpy() , delimiter=";")
        train_subtext_embed_cosine_sim_mat = cosine_similarity(train_subtext_embeds.cpu().detach().numpy(), train_subtext_embeds.cpu().detach().numpy())
        np.savetxt(os.path.join("./output_and_results/", "train_subtext_embeddings_cosine_sim_mat.csv"), train_subtext_embed_cosine_sim_mat , delimiter=";")

        break
    
    '''
    
    addendum_label = pretrain_text_embedder if pretrain_text_embedder is not None else "zero"
    #addendum_label = "body_" + addendum_label
    
    if feature_label == "engagement":
        capture_embedding_similarity(text_embedder, train_set, "./embed_checkpoints/", "train_distilroberta_{}_embeddings.hdf5".format(addendum_label), device)
        capture_embedding_similarity(text_embedder, test_set, "./embed_checkpoints/", "test_distilroberta_{}_embeddings.hdf5".format(addendum_label), device)
    elif feature_label == "sentiment":
        capture_embedding_similarity(text_embedder, train_set, "./embed_checkpoints/", "train_senti_distilroberta_{}_embeddings.hdf5".format(addendum_label), device)
        capture_embedding_similarity(text_embedder, test_set, "./embed_checkpoints/", "test_senti_distilroberta_{}_embeddings.hdf5".format(addendum_label), device)    

    #capture_embedding_similarity(text_embedder, train_set, "./embed_checkpoints/", "train_roberta_zero_embeddings.hdf5", device)
    #capture_embedding_similarity(text_embedder, test_set, "./embed_checkpoints/", "test_roberta_zero_embeddings.hdf5", device)
    

    ''' Test for the end2end model	
    
    end2end_reg_model = CustomModelForEnd2EndRegression(text_embedder,
                                                                    embedding_regressor,
                                                                    num_reg_targets,
                                                                    )

    temp_loader = dataset_inst.get_dataloader(batch_size=4)
    for temp_batch in tqdm(temp_loader):
        #print(temp_batch['tokenized_input_subtexts'].shape)
        print(end2end_reg_model(**temp_batch))
    '''
    
    print("Finished preparing the embedding model.")
    
    
    # ----------------- Contrastive fine-tining of the embedding model -----------------
    '''
    pair_batching_mode = "pairs_per_batch" # Must be "all_pairs_in_dataset" or "pairs_per_batch""
    
    model_args = {"text_embedder": text_embedder,
                "embedding_regressor": embedding_regressor,
                "num_reg_targets": num_reg_targets,
                "loss_func": ProbDistribCrossEntropyLoss(), #JSDivLoss(),#NonLogKLDivLoss() 
                }

    
    contrastive_finetune_end2end_model(dataset_inst.sort(train_set, 'subtexts_counts', reverse=True),
                                               dataset_inst.sort(test_set, 'subtexts_counts', reverse=True),
                                               pair_batching_mode,
                                                model_args,
                                                contrastive_training_args, 
                                                device)
                                      

    print("Finished contrastive fine-tuning of the embedding model.")
    time.sleep(10000)
    '''
    
    #------------------Training the End2End Multinomial Regression model -------------------------
    
    print("Training the End-2-End Multinomial Regression model...")
    
    #for name, param in text_embedder.named_parameters():
    #    #print("\n" ,name, param.requires_grad)
    #    if name.startswith("embedder_base"):
    #        param.requires_grad = False


    #'''####
    dispersion_coeff = 0.001
    if target_type == "mean_score":
        model_args = {"text_embedder": text_embedder,
                    "embedding_regressor": embedding_regressor,
                    "num_reg_targets": 1,
                    "loss_func": CustomMSELossWithDispersion(dispersion_coeff), #JSDivLoss(), #ProbDistribCrossEntropyLoss(), ,
                    }
    
    elif target_type == "score_distribution":
        
        model_args = {"text_embedder": text_embedder,
                    "embedding_regressor": embedding_regressor,
                        "num_reg_targets": num_reg_targets,
                        "loss_func": NonLogKLDivLoss(), #JSDivLoss(), #ProbDistribCrossEntropyLoss(), ,
                    }
    
    
    train_end2end_regression_model(train_set,
                                    test_set, 
                                    None,
                                    model_args,
                                    predict_finetune_training_args, 
                                    device,
                                    dataset_inst)
    
    #'''
    
    print("Successfully finished and closing the session now. Goodbye!")

    
if __name__ == '__main__':
    
    #import deepspeed
    
    script_kwargs = {}
    
    c_arg = None
    for arg in sys.argv[1:]: # skip the script name
        if "--" in arg:
            c_arg = arg.replace("--", "")
        elif c_arg:
            script_kwargs[c_arg] = arg
            c_arg = None
        else:
            raise ValueError("Invalid passing of arguments")
    
   # print("Running script with the following arguments: ", script_kwargs)
    
    try:
        del script_kwargs["deepspeed"]
    except:
        pass
    
    #Default script kwargs:
    default_win_kwargs = {"model_output_dir": "models",
                        "data_dirpath": "S:\\Sync\\University\\2023_MRP_1\\MRP1_WorkDir\\data\\annotated\\",  
                        "HF_MODEL_ID": 'sentence-transformers/all-distilroberta-v1' , #'roberta-base', #  , # 'sentence-transformers/all-distilroberta-v1', #'sentence-transformers/all-mpnet-base-v2' ,
                        #"HF_MODEL_ID": 'roberta-base',
                        "subtexts_aggregation_strategy": "mean",  
                        "pretrain_text_embedder": "CosSim", #"CosSim", #None, #"SimCE","TSDAE", "CosSim", "SimCE-Head-Body", 
                        "target_type": "mean_score", # "mean_score", "score_distribution
                        "deepspeed":None,#"/home/s1930443/.config/deepspeed/ds_config_zero3.json", #None, #"/home/s1930443/.config/deepspeed/ds_config_zero3.json",
                        }
    
    default_alice_kwargs = {"model_output_dir": "/data1/s1930443/hf_models/",
                        "data_dirpath": "/home/s1930443/MRP1_pred/data/",  
                        "HF_MODEL_ID": 'sentence-transformers/all-distilroberta-v1' , #'roberta-base', #  , # 'sentence-transformers/all-distilroberta-v1', #'sentence-transformers/all-mpnet-base-v2' ,
                        #"HF_MODEL_ID": 'roberta-base',
                        "subtexts_aggregation_strategy": "mean",  
                        "pretrain_text_embedder":"CosSim", #"CosSim", #None, #"SimCE","TSDAE", "CosSim", "SimCE-Head-Body", 
                        "target_type": "mean_score", # "mean_score", "score_distribution
                        "deepspeed":None,#"/home/s1930443/.config/deepspeed/ds_config_zero3.json", #None, #"/home/s1930443/.config/deepspeed/ds_config_zero3.json",
                        }   
    
    #Need to convert the known non-string arguments to the correct type
    
    script_kwargs = default_win_kwargs
    main(**script_kwargs)

