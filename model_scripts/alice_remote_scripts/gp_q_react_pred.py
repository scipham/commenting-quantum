
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

# Importing the ML and transformers libraries
import torch
from torch import cuda
from torch import nn
from torchmetrics import MeanSquaredError
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler

from sentence_transformers import SentenceTransformer, models as sentence_models, losses, evaluation, datasets, InputExample

#from datasets import Dataset
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification, RobertaConfig, Trainer, TrainingArguments, AutoModel, AutoTokenizer
#require sentencepiece

from sklearn.metrics.pairwise import cosine_similarity
import wandb

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


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

  
class CustomArticleSentencesDataset(Dataset):
    def __init__(self, article_df, feature_label, tokenizer, max_source_len, pad_tokenized_subtexts, device, target_type="mean_score"):
        assert feature_label in ["engagement", "sentiment"], "Feature label should be 'engagement' or 'sentiment'"
        
        self.device = device
        self.feature_label = feature_label
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len

        self.article_ids = np.unique(article_df["article_id"])
        self.num_articles = len(self.article_ids)
        
        self.tokenizer = tokenizer
        self.input_texts = article_df["body"].values
        self.input_texts = article_df["headline"].values
        
        self.feature_scores = np.array([parse_liststr_to_floatarray(liststr) for liststr in article_df[feature_label + "_scores"]])
        self.input_subtexts = [[tokenizer.cls_token + " " + subtxt + " " + tokenizer.sep_token for subtxt in txt.split(".")] for txt in self.input_texts]
        
        ### Truncate number of subtexts to 50:
        max_num_subtexts = 80
        self.input_subtexts = [subtexts if len(subtexts) <= max_num_subtexts else subtexts[:max_num_subtexts] for subtexts in self.input_subtexts]

      
        self.input_subtexts_counts = [len(art_subtexts) for art_subtexts in self.input_subtexts]
        
        tokenized_input_subtexts = []
        subtexts_attention_masks = []
        
        for art_subtexts in self.input_subtexts:
            art_tokenized_subtexts = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=art_subtexts, 
                                                                    max_length=self.max_source_len, 
                                                                    padding='longest' if pad_tokenized_subtexts else 'do_not_pad',
                                                                    truncation=True,
                                                                    return_tensors='pt' if pad_tokenized_subtexts else None,
                                                                    )
            
            
            #if pad_tokenized_subtexts:
            #    print("Was padded to ", art_tokenized_subtexts["input_ids"].shape[-1], " The lengths of the original sequences ", [self.tokenizer.encode(subtxt, return_tensors='pt').shape[-1] for subtxt in art_subtexts])
            #else:
            #    print("Was not padded. Output size ", len(art_tokenized_subtexts["input_ids"]))
            

            tokenized_input_subtexts.append(art_tokenized_subtexts["input_ids"])
            subtexts_attention_masks.append(art_tokenized_subtexts["attention_mask"])
            
        self.tokenized_input_subtexts = tokenized_input_subtexts
        self.subtexts_attention_masks = subtexts_attention_masks

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
        item = {'input_subtexts': self.input_subtexts[index],
                'tokenized_input_subtexts': self.tokenized_input_subtexts[index],
                "subtexts_attention_masks": self.subtexts_attention_masks[index],
                'targets': self.targets[index, :],
                'feature_scores': self.feature_scores[index, :],
                'article_ids': self.article_ids[index], 
                'subtexts_counts': self.input_subtexts_counts[index],
                }
        return item
    
    def select(self, indices):
        selected_subset = torch.utils.data.Subset(self, indices)
        selected_subset.dynamic_batch_padding = self.dynamic_batch_padding
        selected_subset.device = self.device
        return selected_subset
    
    def sort(self, key, reverse=False):
        sorted_subset = torch.utils.data.Subset(self, sorted(range(len(self)), key=lambda x: self[x]['subtexts_counts'], reverse=reverse))
        sorted_subset.dynamic_batch_padding = self.dynamic_batch_padding
        sorted_subset.device = self.device
        return sorted_subset  
    
    @staticmethod
    def sort(dataset_or_subset, key, reverse=False):
        sorted_subset = torch.utils.data.Subset(dataset_or_subset, sorted(range(len(dataset_or_subset)), key=lambda x: dataset_or_subset[x]['subtexts_counts'], reverse=reverse))
        sorted_subset.dynamic_batch_padding = dataset_or_subset.dynamic_batch_padding
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


    #@staticmethod
    def dynamic_batch_padding(self, batch):
        
        tokenized_input_subtexts = [item['tokenized_input_subtexts'] for item in batch]
        subtexts_attention_masks = [item['subtexts_attention_masks'] for item in batch]
        subtexts_counts_in_batch = [item['subtexts_counts'] for item in batch]
        #subtexts_counts_in_batch = [len(['tokenized_input_subtexts']) for item in batch]
        
        if not monotonic(subtexts_counts_in_batch):
            warnings.warn("Subtexts counts in batch are not monotonic. This makes padding rather inefficient. It is recommended to sort the dataset before creating the DataLoader.")

        max_num_of_subtexts_in_batch = max(subtexts_counts_in_batch)
        max_source_len = tokenized_input_subtexts[0].shape[-1]
        #print(max_num_of_subtexts_in_batch, subtexts_counts_in_batch)
        
        padded_tokenized_input_subtexts = torch.empty((0, max_num_of_subtexts_in_batch, max_source_len), dtype=torch.long) #First dimension is the number of articles in the batch
        padded_subtexts_attention_masks = torch.empty((0, max_num_of_subtexts_in_batch, max_source_len), dtype=int) 
        
        for b_i in range(len(batch)):
            art_tokenized_subtexts, art_subtexts_attention_masks = tokenized_input_subtexts[b_i], subtexts_attention_masks[b_i]
        
            padded_art_tokenized_subtexts = nn.functional.pad(art_tokenized_subtexts, (0,max_source_len - art_tokenized_subtexts.shape[1],0, max_num_of_subtexts_in_batch - art_tokenized_subtexts.shape[0]), "constant", 0)
            padded_art_subtexts_attention_masks = nn.functional.pad(art_subtexts_attention_masks, (0, max_source_len - art_subtexts_attention_masks.shape[1],0, max_num_of_subtexts_in_batch - art_subtexts_attention_masks.shape[0]), "constant", 0)
            padded_tokenized_input_subtexts = torch.concatenate((padded_tokenized_input_subtexts, torch.unsqueeze(padded_art_tokenized_subtexts, dim=0)), dim=0)
            padded_subtexts_attention_masks = torch.concatenate((padded_subtexts_attention_masks, torch.unsqueeze(padded_art_subtexts_attention_masks, dim=0)), dim=0)
        
        #print("\n", len(batch), padded_tokenized_input_subtexts.shape, padded_subtexts_attention_masks.shape)

        return {'input_subtexts': [item['input_subtexts'] + ["[PAD]", ]* (max_num_of_subtexts_in_batch - len(item['input_subtexts'])) for item in batch],
            'tokenized_input_subtexts': padded_tokenized_input_subtexts.to(self.device),
            'subtexts_attention_masks': padded_subtexts_attention_masks.to(self.device),
            'targets': torch.concatenate([item['targets'].reshape(1,-1) for item in batch], dim=0).to(torch.float32).to(self.device),
            'feature_scores': [item['feature_scores'] for item in batch],
            'article_ids': [item['article_ids'] for item in batch], 
            'subtexts_counts': [item['subtexts_counts'] for item in batch],
            }
    
    def get_dataloader(self, batch_size):
        return DataLoader(self.sort('subtexts_counts', reverse=True), batch_size=batch_size, collate_fn=self.dynamic_batch_padding, shuffle=False)
        


class WeightedAverageAttention(nn.Module):
    def __init__(self, input_dim, device):
        super().__init__()
        ''' Similar to Dot-Product Attention, but the value vectors are not parameterized. We just weight the value vectors themselves by the attention scores.
        Note: The output embedding must in this case always be equal to the input embedding. So no option to change the embedding size.'''

        self.embed_dim = input_dim
        self.head_dim = input_dim

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qk_proj = nn.Linear(input_dim, 2 * self.embed_dim, device=device)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qk_proj.weight)
        self.qk_proj.bias.data.fill_(0)
    
    
    @staticmethod
    def scaled_dot_product(q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / np.sqrt(d_k)
        if mask is not None:
            #mask = mask.unsqueeze(1) if len(mask.shape) != len(attn_logits.shape) else mask
            #print(mask.shape, attn_logits.shape)
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        #print("Attn logtis: ", attn_logits.shape)
        attention = nn.functional.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        #print("Attention and v: ", attention.shape, v.shape)
        #print("Values: ", values.shape	)
        
        return values, attention

    
    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qk = self.qk_proj(x) #Computes Q and K in one go

        # Separate Q, K from linear output
        qk = qk.reshape(batch_size, seq_length, 2 * self.head_dim) 
        q, k = qk.chunk(2, dim=-1)
        v = x

        #print(self.head_dim, batch_size, seq_length, embed_dim)
        
        # Determine value outputs
        values, attention = self.scaled_dot_product(q, k, v, mask=mask)
        values = values.reshape(batch_size, seq_length, embed_dim)

        if return_attention:
            return torch.sum(values, dim=1), torch.sum(attention, dim=1) / torch.sum(torch.sum(attention, dim=1),dim=-1).unsqueeze(-1)
        else:
            return torch.sum(values, dim=1)

class CustomModelForArticleEmbedding(nn.Module):
    def __init__(self, subtext_embedder_model_or_hf_id, subtext_pooling_mode, device):
        super(CustomModelForArticleEmbedding, self).__init__()
        
        #Subtext embedding
        if isinstance(subtext_embedder_model_or_hf_id, nn.Module):
            self.embedder_base = subtext_embedder_model_or_hf_id.to(device)
        elif isinstance(subtext_embedder_model_or_hf_id, str):    
            if "roberta" in subtext_embedder_model_or_hf_id:
                self.embedder_base = RobertaModel.from_pretrained(subtext_embedder_model_or_hf_id) #all-mpnet-base-v2, all-distilroberta-v1, all-MiniLM-L12-v2
                self.embedder_base = self.embedder_base.to(device)
            else:
                self.embedder_base = AutoModel.from_pretrained(subtext_embedder_model_or_hf_id) #all-mpnet-base-v2, all-distilroberta-v1, all-MiniLM-L12-v2
                self.embedder_base = self.embedder_base.to(device)
        else:
            raise ValueError("Subtext embedder model must be HF_model_id or nn.Module model.")
        
        
        #Article embedding by subtext embedding aggregation
        self.subtext_pooling_mode = subtext_pooling_mode
        if subtext_pooling_mode == "mean":
            self.subtext_pooler = None #Needs kernel_size, so we need to define it in the forward pass
        elif subtext_pooling_mode == "attention":
            self.subtext_pooler = WeightedAverageAttention(input_dim=self.embedder_base.config.hidden_size, device=device)
        elif subtext_pooling_mode == "gated":
            self.subtext_pooler = nn.Linear(self.embedder_base.config.hidden_size, 1, device=device) #Applied to 3D tensor of shape (batch_size, num_subtexts, embedding_length), but is only applied to the last dimension
        else:
            raise ValueError("Invalid subtext pooling mode.")
        
        self.text_embedding_size = self.embedder_base.config.hidden_size
        self.device = device

    def get_text_embedding_size(self):
        return self.text_embedding_size
    
    def forward(self, tokenized_input_subtexts, subtexts_attention_masks,targets=None, input_subtexts=None, return_attention_weights=False,**kwargs):
        
        assert len(tokenized_input_subtexts.shape) == 3
        assert tokenized_input_subtexts.shape == subtexts_attention_masks.shape
        
        #Reshape the input tensors to compute the embeddings over all texts and subtexts at once
        tokenized_input_subtexts_shape = tokenized_input_subtexts.shape
        print("\n tish: ", tokenized_input_subtexts_shape)
        
        #####
        #np.savetxt(os.path.join("./output_and_results/", "tokenized_subtexts.csv"), tokenized_input_subtexts.reshape(-1, tokenized_input_subtexts_shape[-1]).cpu().detach().numpy() , delimiter=";")
        #tokenized_subtext_cosine_sim_mat = cosine_similarity(tokenized_input_subtexts.reshape(-1, tokenized_input_subtexts_shape[-1]).cpu().detach().numpy(), tokenized_input_subtexts.reshape(-1, tokenized_input_subtexts_shape[-1]).cpu().detach().numpy())
        #sim_mat_sidelength = subtexts_attention_masks.shape[0]*subtexts_attention_masks.shape[1]
        #print(sim_mat_sidelength, subtexts_attention_masks.shape)
        #sim_mat_mask = torch.logical_or(torch.any(subtexts_attention_masks, dim=-1).int().reshape(-1).unsqueeze(-1).expand(sim_mat_sidelength, sim_mat_sidelength).int(), torch.any(subtexts_attention_masks, dim=-1).int().reshape(-1).unsqueeze(0).expand(sim_mat_sidelength, sim_mat_sidelength).int())
        #tokenized_subtext_cosine_sim_mat[sim_mat_mask.cpu().numpy()] = np.nan
        #np.savetxt(os.path.join("./output_and_results/", "tokenized_subtext_cosine_sim_mat.csv"), tokenized_subtext_cosine_sim_mat , delimiter=";")
        #####
        
        
        #print(tokenized_input_subtexts.device, subtexts_attention_masks.device, subtexts_attention_masks.reshape(-1, tokenized_input_subtexts_shape[-1]).device)
        #Note: the following pooling has nothing to do with the subtexts aggregation. Rather it is still part of embedding the subtexts themselves.

        self.embedder_base.to(self.device)
        subtext_embeddings = mean_pooling(self.embedder_base(input_ids=tokenized_input_subtexts.reshape(-1, tokenized_input_subtexts_shape[-1]), 
                                                             attention_mask=subtexts_attention_masks.reshape(-1, tokenized_input_subtexts_shape[-1]))[0], 
                                          subtexts_attention_masks.reshape(-1, tokenized_input_subtexts_shape[-1])) #.last_hidden_state is equivalent to [0]
        
        text_subtext_embeddings = subtext_embeddings.reshape(*tokenized_input_subtexts_shape[:-1], self.embedder_base.config.hidden_size) #Shape: (batch_size, max_num_of_sentences, embedding_length)
        subtext_embeddings_attention_masks = torch.any(subtexts_attention_masks, dim=-1).int() #Shape: (batch_size, max_num_of_subtexts_in_batch)
        
        
        #print("\n ||| ", tokenized_input_subtexts.shape, " ", subtext_embeddings_attention_masks.shape)
        np.savetxt(os.path.join("./output_and_results/", "attn_mask_tokenized_input_subtexts.csv"), subtexts_attention_masks.reshape(-1, tokenized_input_subtexts_shape[-1]).cpu().detach().numpy() , delimiter=";")
        #for t_i in range(tokenized_input_subtexts_shape[-1]):
        #    np.savetxt(os.path.join("./output_and_results/embed_out/", "subtext_embeddings_tok"+str(t_i)+".csv"), 
        #                self.embedder_base(input_ids=tokenized_input_subtexts.reshape(-1, tokenized_input_subtexts_shape[-1]), 
        #                                attention_mask=subtexts_attention_masks.reshape(-1, tokenized_input_subtexts_shape[-1]),
        #                                )[0][:, t_i, :].cpu().detach().numpy(),
        #                delimiter=";")
        
        np.savetxt(os.path.join("./output_and_results/", "subtext_embeddings.csv"), subtext_embeddings.cpu().detach().numpy() , delimiter=";")
        subtext_embed_cosine_sim_mat = cosine_similarity(subtext_embeddings.cpu().detach().numpy(), subtext_embeddings.cpu().detach().numpy())
        print("\n subtext_embeddings: ",  subtext_embeddings.shape, " ", subtext_embed_cosine_sim_mat.shape)
        
        sim_mat_sidelength = subtext_embeddings_attention_masks.shape[0]*subtext_embeddings_attention_masks.shape[1]
        np.savetxt(os.path.join("./output_and_results/", "subtext_embeds_cosine_sim_mat.csv"), subtext_embed_cosine_sim_mat , delimiter=";")
        subtext_embeddings.to(self.device)
        #####
        
        if self.subtext_pooling_mode == "mean":
            text_embeddings = mean_pooling(text_subtext_embeddings, subtext_embeddings_attention_masks)
            print("Text embed shape", text_embeddings.shape)
            #text_embeddings = torch.mean(text_subtext_embeddings, dim=1)
        
        elif self.subtext_pooling_mode == "attention":
            #Normalize text embeddings across subtexts
            text_subtext_embeddings = nn.functional.normalize(text_subtext_embeddings, p=2, dim=1)
            subtext_embeddings_attention_masks = subtext_embeddings_attention_masks.unsqueeze(1).expand(-1, subtext_embeddings_attention_masks.shape[-1], -1)
            text_embeddings, attention_weights = self.subtext_pooler(text_subtext_embeddings, subtext_embeddings_attention_masks, return_attention=True)
        
        elif self.subtext_pooling_mode == "gated":
            gate_weights = nn.functional.softmax(self.subtext_pooler(text_subtext_embeddings), dim=1) #Shape: (batch_size, num_subtexts, 1)
            gate_weights = gate_weights * subtext_embeddings_attention_masks.unsqueeze(-1) #Set gate weights to 0 for subtexts that are not present in the batch (i.e. padding subtexts)
            text_embeddings = torch.sum(text_subtext_embeddings * gate_weights, dim=1)
        
        
        if self.subtext_pooling_mode == "attention" and return_attention_weights:
            return (nn.functional.normalize(text_embeddings, p=2, dim=1), attention_weights)
        elif self.subtext_pooling_mode == "gated" and return_attention_weights:
            return (nn.functional.normalize(text_embeddings, p=2, dim=1), gate_weights)
        else:
            #print(self.subtext_pooling_mode, return_attention_weights)
            #print("Neither attention weights nor gate weights are returned.")
            #return nn.functional.normalize(text_embeddings, p=2, dim=1)
            return text_embeddings

    
    
#===================================================================================================
# From here on, everything is about prediction
#===================================================================================================

class CustomVariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        # Inducing points are the points that are used to approximate the function and are optimized during training. Given are only the initial values.
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0)) # The variational distribution is a distribution over the inducing points. It is initialized with the number of inducing points.
        #variational_distribution = gpytorch.variational.NaturalVariationalDistribution(inducing_points.size(0)) # The variational distribution is a distribution over the inducing points. It is initialized with the number of inducing points.
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(CustomVariationalGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class CustomModelForEnd2EndGPRegression(nn.Module):
    def __init__(self, text_embedder, embedding_regressor, device):
        super().__init__()
        self.text_embedder = text_embedder
        self.text_embedding_size = text_embedder.get_text_embedding_size()
        
        self.reg_head = embedding_regressor
        self.config = self.text_embedder.embedder_base.config
        
        self.device = device

    def forward(self, tokenized_input_subtexts, subtexts_attention_masks, targets=None, return_attention_weights=False, **kwargs):

        #First compute the embedding of each text (from their subtexts)
        if self.text_embedder.subtext_pooling_mode != "mean" and return_attention_weights:
            x, attention_weights = self.text_embedder(tokenized_input_subtexts, subtexts_attention_masks, return_attention_weights=return_attention_weights)
            #Next compute the regression head	
            x = self.reg_head(x)
            return x, attention_weights
        else:
            x = self.text_embedder(tokenized_input_subtexts, subtexts_attention_masks)
            #Next compute the regression head
            x = self.reg_head(x)
            return x


class CustomEnd2EndGPRegressionTrainer(Trainer):
    def __init__(self, *args, loss_func_or_wrapper,device,  **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.loss_func_or_wrapper = loss_func_or_wrapper
        
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # implement custom logic here
        outputs = model(tokenized_input_subtexts = inputs['tokenized_input_subtexts'].to(self.device),
                        subtexts_attention_masks = inputs['subtexts_attention_masks'].to(self.device),
                        return_attention_weights = False,
                        )
        #NOTE: Outputs is of type MultiVariateNormal
        #f_preds = model(test_x)
        #y_preds = likelihood(model(test_x))
        
        #f_mean = f_preds.mean
        #f_var = f_preds.variance
        #f_covar = f_preds.covariance_matrix
        #f_samples = f_preds.sample(sample_shape=torch.Size(1000,))
         
                
        ### inputs['targets'] = inputs['targets'].to(outputs.dtype)
        
        #For GP MLI loss must have a minus sign in front of it!
        loss = -self.loss_func_or_wrapper(outputs, inputs['targets'].squeeze(-1).to(self.device))
        
        print(outputs, " || ", inputs['targets'].squeeze(-1))
        print(outputs.mean.shape, " |-| ",  outputs.covariance_matrix.shape, " |-| ", loss.shape, " |-| ", inputs['targets'].squeeze(-1).shape )
        print(type(outputs), " || ", type(inputs['targets'].squeeze(-1)))
        print(loss, type(loss))
        return (loss, outputs) if return_outputs else loss
    
#===================================================================================================
#=============== Fine-tuning of End-2-End predicton model ====================================================

def train_end2end_regression_model(train_dataset, val_dataset, data_collator,model_args,  training_args, device, full_dataset=None):    

    required_model_args = ["text_embedder", "embedding_regressor", "num_reg_targets", "loss_func", "likelihood"]
    assert all([key in model_args.keys() for key in required_model_args]), f"Missing one or more of the required model arguments: {required_model_args}"
    
    # Make sure we are logged in to wandb
    os.environ["WANDB_API_KEY."]="925f0f3c8de42f022a0a5a390aab9845cb5c92cf"
    #wandb.login()
    # WandB – Initialize a new run
    wandb.init(project="article_score_logits_regression_test")
    # save the trained model checkpoint to wandb
    #os.environ["WANDB_LOG_MODEL"]="true"
    # turn off watch to log faster
    os.environ["WANDB_WATCH"]="false"
    #wandb.watch(model, log="all")
    
    # Store some key paramters to the WandB config for later reference
    config = wandb.config          # Initialize config
    
    config.TRAIN_BATCH_SIZE = training_args.per_device_train_batch_size  
    config.VALID_BATCH_SIZE = training_args.per_device_eval_batch_size   
    config.TRAIN_EPOCHS = training_args.num_train_epochs           
    #config.VAL_EPOCHS = training_args.num_eval_epochs      
    config.LEARNING_RATE = training_args.learning_rate    
    config.SEED = training_args.seed
    
    os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"

    if isinstance(model_args["embedding_regressor"], nn.Module):
        
        model_args["embedding_regressor"].train()
        model_args["likelihood"].train()
        
        end2end_model = CustomModelForEnd2EndGPRegression(text_embedder=model_args["text_embedder"],
                                                                embedding_regressor=model_args["embedding_regressor"],
                                                                device=device)
        
        


        
        val_loader = DataLoader(val_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=data_collator ,shuffle=False)
        #validate_end2end_model(end2end_model, val_loader, 1, model_args["loss_func"], device)
        
        
        # define the trainer and start training
        trainer = CustomEnd2EndGPRegressionTrainer(model=end2end_model,
                                                            args=training_args,
                                                            loss_func_or_wrapper=model_args["loss_func"],
                                                            device=device,
                                                            train_dataset=train_dataset,
                                                            eval_dataset=val_dataset,
                                                            data_collator=data_collator,
                                                            #compute_metrics=compute_metrics,
                                                        )
        trainer.train()
        
        evaluate_end2end_model(trainer.model, model_args["likelihood"],val_loader, CustomMSELoss(), "./output_and_results/temp_attention_predictions.csv", device, target_type="mean_score")
        
        #------------------Other experiments that should happen after training the end2end model:-------------------------
    
        #capture_embedding_similarity(end2end_model.text_embedder, full_dataset, "./output_and_results/", device)
    
        
        '''
        if training_args.deepspeed:
            checkpoint_dir = os.path.join(trainer.args.output_dir, "checkpoint-final")
            trainer.deepspeed.save_checkpoint(checkpoint_dir)
            end2end_model.cpu()
            trainer.save_model(os.path.join(trainer.args.output_dir, "model-final"))
        else:
            trainer.save_model(os.path.join(trainer.args.output_dir, "model-final"))
        '''
    else:
        raise ValueError("embedding_regressor must be a nn.Module")



def evaluate_end2end_model(model, likelihood, eval_loader, error_fun, preds_output_filepath, device, target_type):

    if target_type == "mean_score":
        running_metrics = {"mse_loss": 0.0,
                            }
    
    prediction_buffer = []

    with torch.no_grad():
        for (b_i, batch) in enumerate(eval_loader,0):
            batch_size = len(batch['article_ids'])
            
            targets = batch['targets'].to(device)
            
            result = model(tokenized_input_subtexts = batch['tokenized_input_subtexts'].to(device),
                        subtexts_attention_masks = batch['subtexts_attention_masks'].to(device),
                        return_attention_weights=True,
                        )
            
            if isinstance(result, tuple):
                outputs, attention_weights = result
            else:
                outputs, attention_weights = result, None
            
            if target_type == "mean_score":
                targets = targets.squeeze(-1)

            #NOTE: Outputs is of type MultiVariateNormal
            #f_preds = model(test_x)
            f_preds = outputs
            f_mean = f_preds.mean
            f_var = f_preds.variance
            f_covar = f_preds.covariance_matrix
            #f_samples = f_preds.sample(sample_shape=torch.Size(1000,))
            
            y_preds = likelihood(f_preds)
            y_mean = y_preds.mean
            y_var = y_preds.variance

            print("f Result: ", f_mean, targets)
            print("y Result: ", y_mean, targets, y_var)
            
            errors = error_fun(y_mean, targets, keep_batch_dim=True).cpu().numpy()
            
            print("Errors: ", errors)
            
            if target_type == "mean_score":
                metrics_output = compute_simple_regression_metrics((y_mean, targets))

            running_metrics = {k: running_metrics[k] + metrics_output[k] / (len(eval_loader)*batch_size) for k in running_metrics.keys()}

            y_mean = y_mean.cpu().numpy()
            y_var = y_var.cpu().numpy()
            f_mean = f_mean.cpu().numpy()

            if attention_weights is not None:
                attention_weights = attention_weights.cpu().numpy()
            targets = targets.cpu().numpy()
            
            
            subtexts = batch['input_subtexts']
            if type(subtexts)==torch.Tensor:
                subtexts = subtexts.tolist()
            elif type(subtexts)==list:
                if type(subtexts[0]) != list:
                    raise ValueError("subtexts must be nested list of strings")
            else:
                raise ValueError("subtexts must be either a torch.Tensor or a list")
            
            print("Number of subtexts: ", [len(art_subtexts) for art_subtexts in subtexts])
            print("Targets: ", targets)
            print("Outputs: ", y_mean)
            print("f Mean: ", f_mean)
            print("Attention Weights: ", attention_weights)
            print(metrics_output)
            
            if batch_size > 1: #Single target but multiple samples in batch
                if len(y_mean.shape)==1:
                    y_mean = y_mean[:, np.newaxis]
                if len(y_var.shape)==1:
                    y_var = y_var[:, np.newaxis]
                if len(f_mean.shape)==1:
                    f_mean = f_mean[:, np.newaxis]
                if len(targets.shape)==1:
                    targets = targets[:, np.newaxis]
                if attention_weights is not None and len(attention_weights.shape)==1:
                    attention_weights = attention_weights[:, np.newaxis]
            else: #Multiple targets but (possibly) single sample in batch
                if len(y_mean.shape)==1:
                    y_mean = y_mean[np.newaxis, :]
                if len(y_var.shape)==1:
                    y_var = y_var[np.newaxis, :]
                if len(f_mean.shape)==1:
                    f_mean = f_mean[np.newaxis, :]
                if len(targets.shape)==1:
                    targets = targets[np.newaxis, :]
                if attention_weights is not None and len(attention_weights.shape)==1:
                    attention_weights = attention_weights[np.newaxis, :]
            

            batch_out_dicts = [{"article_id": batch["article_ids"][sample_idx],
                        "subtexts": subtexts[sample_idx],
                        "f_mean": f_mean[sample_idx, :].tolist(),
                        "prediction": y_mean[sample_idx, :].tolist(),
                        "variability": y_var[sample_idx, :].tolist(),
                        "target": targets[sample_idx,:].tolist(),
                        "error": errors[sample_idx].tolist(),
                        "attention_weights": attention_weights[sample_idx, :].tolist() if attention_weights is not None else None,
                        } for sample_idx in range(batch_size)]

            prediction_buffer.extend(batch_out_dicts)
            
            if b_i%100==0:
                print(f'Completed evaluation on {b_i} batches out of {len(eval_loader)}')

    print("Total Metrics: ", running_metrics)
    
    export_eval_df = pd.DataFrame(prediction_buffer)
    export_eval_df.to_csv(preds_output_filepath, index=False, sep=";")
    return export_eval_df


def load_embedder_model(pretrain_subtext_embedder, subtexts_aggregation_strategy, base_model_hf_id , train_dataset, device):
    # "SimCE" # Is "SimCE" or "TSDAE" or None
    if pretrain_subtext_embedder in ["SimCE", "TSDAE"]:
        
        if  pretrain_subtext_embedder == "TSDAE":
            model_output_dir = "./models/distilroberta-TSDAE-sentence-embedder/" #"./models/mpnet-SimCE-sentence-embedder/"
            base_model_hf_id = 'all-distilroberta-v1' # 'all-mpnet-base-v2' #'all-distilroberta-v1' # 'all-mpnet-base-v2'
        elif pretrain_subtext_embedder == "SimCE":
            model_output_dir = "./models/mpnet-SimCE-sentence-embedder/"
            base_model_hf_id = 'all-distilroberta-v1'
        
        try:
            subtext_embedder = AutoModel.from_pretrained(model_output_dir)    
        except:
        
            subtext_sentence_embedder = SentenceTransformer(base_model_hf_id, device=device) #'all-mpnet-base-v2')
            
            train_sentence_collection = sum([sample['input_subtexts'] for sample in train_dataset], [])
            print("We have {} train sentences".format(len(train_sentence_collection)))
            if pretrain_subtext_embedder == "SimCE":
                train_examples = [InputExample(texts=[c_subtext, c_subtext]) for c_subtext in train_sentence_collection] # Convert train sentences to sentence pairs compatible with MultipleNegativesRankingLoss
                train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
                simce_loss = losses.MultipleNegativesRankingLoss(subtext_sentence_embedder)
                train_loss = simce_loss
            elif pretrain_subtext_embedder == "TSDAE":
                train_denoising_set = datasets.DenoisingAutoEncoderDataset(train_sentence_collection)
                train_dataloader = DataLoader(train_denoising_set, shuffle=True, batch_size=8)
                denoising_loss = losses.DenoisingAutoEncoderLoss(subtext_sentence_embedder, decoder_name_or_path=base_model_hf_id, tie_encoder_decoder=True)
                train_loss = denoising_loss
            #
            # Fine-tune the classification model on the reference comments:
            #torch.set_grad_enabled(True)
            
            subtext_sentence_embedder.fit([(train_dataloader, train_loss)],# train_objectives=[(train_dataloader, simce_loss)],   #
                                        epochs=50,
                                        weight_decay=0,
                                        scheduler='constantlr',
                                        optimizer_params={'lr': 4e-2}, 
                                        show_progress_bar=True,
                                        )
                                    
            subtext_embedder = subtext_sentence_embedder._first_module().auto_model
            subtext_embedder.save_pretrained(model_output_dir)
            
        finally:
            text_embedder = CustomModelForArticleEmbedding(subtext_embedder_model_or_hf_id=subtext_embedder,
                                                            subtext_pooling_mode=subtexts_aggregation_strategy, #'mean', 'attention', 'gated'
                                                            device=device)
    else:
        text_embedder = CustomModelForArticleEmbedding(subtext_embedder_model_or_hf_id=base_model_hf_id,
                                                        subtext_pooling_mode=subtexts_aggregation_strategy, #'mean', 'attention', 'gated'
                                                        device=device)
    return text_embedder

#-----------------------------------------------------------------------
# Tests and experiments - not core routines
#-----------------------------------------------------------------------

def capture_embedding_similarity(text_embedder, tokenized_dataset,data_output_dir, device):
    
    eval_batch_size = 4
    data_loader = DataLoader(tokenized_dataset.sort(tokenized_dataset, 'subtexts_counts', reverse=True), batch_size=eval_batch_size, collate_fn=tokenized_dataset.dynamic_batch_padding ,shuffle=False)
    
    embeddings_buffer = np.zeros((len(tokenized_dataset), text_embedder.get_text_embedding_size()))
    
    with torch.no_grad():
        for (b_i, batch) in enumerate(data_loader,0):
            batch_text_embeddings = text_embedder(tokenized_input_subtexts = batch['tokenized_input_subtexts'].to(device),
                                                    subtexts_attention_masks = batch['subtexts_attention_masks'].to(device),
                                                    )
            
            
            
            embeddings_buffer[b_i*eval_batch_size:(b_i+1)*eval_batch_size,:] = batch_text_embeddings.cpu().numpy()
            
            if b_i % 20 == 0:
                print("Completed embedding computation for {} batches out of {}".format(b_i, len(data_loader)))
    
    # Compute cosine similarity matrix between all pairs of embeddings:
    cosine_sim_mat = cosine_similarity(embeddings_buffer, embeddings_buffer)
    
    # Save cosine similarity matrix to csv file:
    #np.savetxt(os.path.join(data_output_dir, "cosine_sim_mat.csv"), cosine_sim_mat, delimiter=";")
    

def main(model_output_dir, data_dirpath, HF_MODEL_ID, subtexts_aggregation_strategy, pretrain_subtext_embedder, target_type, deepspeed):

    # Set up non-variable parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #torch.backends.cuda.matmul.allow_tf32 = True
    #device = torch.device('cpu')
    
    print("We are using the following device:  ", device)
    MAX_SOURCE_LEN = 512
    feature_label = "engagement" #"sentiment" #"engagement

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
    dataset_inst = CustomArticleSentencesDataset(art_df, feature_label, tokenizer, MAX_SOURCE_LEN, pad_tokenized_subtexts=True, device=device, target_type=target_type)
    train_set, test_set = dataset_inst.train_test_split(train_frac=0.8, seed=global_seed)
    

    # ------ For deepspeed, we need to load all TrainingArguments in the script before instantiating any model. ------

    
    predict_finetune_training_args = TrainingArguments(output_dir=model_output_dir,
                                        #report_to="wandb",
                                        #logging_steps=5,
                                        #run_name="...", # A descriptor for the run. Typically used for wandb logging.
                                        label_names= ["targets"],
                                        remove_unused_columns=False,
                                        per_device_train_batch_size=32,
                                        per_device_eval_batch_size=8,
                                        gradient_accumulation_steps=2,
                                        #gradient_checkpointing=True,
                                        #fp16=True,#,tf32=True,
                                        optim="adamw_torch",
                                        learning_rate=2e-2,
                                        num_train_epochs=500,
                                        # max_steps = 1,
                                        save_strategy="epoch",
                                        save_steps = 2,
                                        #do_eval=True,
                                        evaluation_strategy="steps", 
                                        eval_steps=20, #10,
                                        seed=global_seed,
                                        dataloader_pin_memory=False,
                                        deepspeed=deepspeed,
                                        )
    

    #-------Defining & Fine-tuning the subtext embedding model:-------
    
    text_embedder = load_embedder_model(pretrain_subtext_embedder=pretrain_subtext_embedder,
                                        subtexts_aggregation_strategy=subtexts_aggregation_strategy,
                                        base_model_hf_id=HF_MODEL_ID,
                                        train_dataset=train_set,
                                        device=device,
                                        ) 
        
    
    for name, param in text_embedder.named_parameters():
        #print("\n" ,name, param.requires_grad)
        if name.startswith("embedder_base"):
            param.requires_grad = False
    
    
    if target_type == "mean_score":
        
        num_inducing_points = len(dataset_inst) #Initialiye with 200 random points from the training set
        inducing_points = []
        for sample in dataset_inst:
            inducing_points.append(text_embedder(sample["tokenized_input_subtexts"].to(device).unsqueeze(0), sample["subtexts_attention_masks"].to(device).unsqueeze(0)).squeeze())
        inducing_points = torch.concatenate([ind_point.reshape(1, -1) for ind_point in inducing_points], axis=0)

        embedding_regressor = CustomVariationalGPModel(inducing_points=inducing_points,)
        embedding_regressor.to(device)
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.to(device)
        
        gp_loss = gpytorch.mlls.VariationalELBO(likelihood, embedding_regressor, num_data=len(train_set)) #num_data: Number of 

  
    text_embedder = text_embedder.to(device)
    embedding_regressor = embedding_regressor.to(device)
    

    ''' Test for the embedding model
    temp_loader = dataset_inst.get_dataloader(batch_size=4)
    for temp_batch in tqdm(temp_loader):
        #print(temp_batch['tokenized_input_subtexts'].shape)
        print(embedding_model(**temp_batch))
    '''
    
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
    
    
    
    #------------------Training the End2End Multinomial Regression model -------------------------
    
    print("Training the End-2-End Multinomial Regression model...")
    
    


    if target_type == "mean_score":
        model_args = {"text_embedder": text_embedder,
                    "embedding_regressor": embedding_regressor,
                    "num_reg_targets": 1,
                    "loss_func": gp_loss,
                    "likelihood": likelihood,
                    }
    
    #elif target_type == "score_distribution":
        
    #    model_args = {"text_embedder": text_embedder,
    #                "embedding_regressor": embedding_regressor,
    #                    "num_reg_targets": num_reg_targets,
    #                    "loss_func": NonLogKLDivLoss(), #JSDivLoss(), #ProbDistribCrossEntropyLoss(), ,
    #                }
    
    
    train_end2end_regression_model(dataset_inst.sort(train_set, 'subtexts_counts', reverse=True),
                                    dataset_inst.sort(test_set, 'subtexts_counts', reverse=True),
                                    dataset_inst.dynamic_batch_padding,
                                    model_args,
                                    predict_finetune_training_args, 
                                    device,
                                    dataset_inst)
    

    
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
                        "HF_MODEL_ID": 'sentence-transformers/all-mpnet-base-v2' ,
                        "subtexts_aggregation_strategy": "mean",  
                        "pretrain_subtext_embedder": None,
                        "target_type": "mean_score", # "mean_score", "score_distribution"
                        "deepspeed": None,
                          }
    
    default_alice_kwargs = {"model_output_dir": "/data1/s1930443/hf_models/",
                        "data_dirpath": "/home/s1930443/MRP1_pred/data/",  
                        "HF_MODEL_ID": 'sentence-transformers/all-mpnet-base-v2', # 'sentence-transformers/all-distilroberta-v1' , # 'sentence-transformers/all-distilroberta-v1', #'sentence-transformers/all-mpnet-base-v2' ,
                        "subtexts_aggregation_strategy": "mean",  
                        "pretrain_subtext_embedder": None, #"SimCE", #"SimCE",
                        "target_type": "mean_score", # "mean_score", "score_distribution
                        "deepspeed":None, # "/home/s1930443/.config/deepspeed/ds_config_zero3.json", #None, #"/home/s1930443/.config/deepspeed/ds_config_zero3.json",
                        }
    
    #Need to convert the known non-string arguments to the correct type
    
    script_kwargs = default_alice_kwargs
    main(**script_kwargs)

