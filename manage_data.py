# Importing stock libraries
import os
import random
import numpy as np
import pandas as pd
import re
import time


# Importing the ML and transformers libraries
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
#from datasets import Dataset
from transformers import T5Tokenizer
#require sentencepiece
import wandb
import deepspeed


class CustomDataset(Dataset):
    def __init__(self, article_df, comment_df, tokenizer, max_source_len, max_target_len):
        self.tokenizer = tokenizer

        #TODO: Parse article_dataframe and comment_dataframe to the encoder input and target text
        self.encoder_input_texts = []
        self.target_texts = []
        self.art_ids_map = []
        self.cmt_ids_map = []
        for art_id in article_df["article_id"]:
            art_cmts = [cmt["body"] for (_, cmt) in comment_df[comment_df["article_id"] == art_id].iterrows()]
            self.target_texts.extend(art_cmts) #Equals the decoder input texts
            self.encoder_input_texts.extend(len(art_cmts) * list(article_df[article_df["article_id"] == art_id]["body"]))
            self.art_ids_map.extend(len(art_cmts) * [art_id]) #Remember the article id for each comment
            self.cmt_ids_map.extend([cmt["comment_id"] for (_, cmt) in comment_df[comment_df["article_id"] == art_id].iterrows()])
        
        self.art_ids = np.unique(self.art_ids_map)
        self.num_articles = len(self.art_ids)
        
        print(len(self.encoder_input_texts), len(self.target_texts))
        # encoding the input and target text
        self.input_encodings = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=self.encoder_input_texts, 
                                                                max_length=max_source_len, 
                                                                padding='max_length',
                                                                truncation=True,
                                                                return_tensors='pt')
        self.target_encodings = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=self.target_texts,
                                                                max_length=max_target_len, 
                                                                padding='max_length',
                                                                truncation=True,
                                                                return_tensors='pt')
        
        self.input_ids, self.attention_mask = self.input_encodings['input_ids'], self.input_encodings['attention_mask']
        self.label_ids, self.decoder_attention_mask = self.target_encodings['input_ids'], self.target_encodings['attention_mask']
        #self.decoder_input_ids = label_ids[:, :-1].contiguous() #Decoder input: Shift left, last token falls off
        #self.decoder_attention_mask = self.decoder_attention_mask[:, :-1].contiguous() #Decoder output: Shift right to match decoder input sequence length
        #self.target_ids = label_ids[:, 1:].clone().detach() #Decoder output: Shift right to match decoder input sequence length
        #self.target_ids[label_ids[:, 1:] == tokenizer.pad_token_id] = -100 #Ignore pad tokens in loss calculation

        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        item = {'input_ids': self.input_ids[index], #Encoder input token ids
                'attention_mask': self.attention_mask[index], #Encoder input attention mask
                #'decoder_input_ids': self.label_ids[index], #Decoder input token ids
                #'decoder_attention_mask': self.decoder_attention_mask[index], #Decoder input attention mask
                'label_ids': self.label_ids[index], #Decoder output token ids (same as decoder input token ids in our task)
                'article_id': self.art_ids_map[index], #Article id associated to the comment
                'comment_id': self.cmt_ids_map[index], #Comment id of the comment
                }
        return item

    def select(self, indices):
        return torch.utils.data.Subset(self, indices)
    
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
    
    def train_test_split_on_articles(self, train_frac, seed):
        """Split the dataset on articles, and keep all comments per article in the same split."""	
        assert 0.0 < train_frac < 1.0
        
        train_art_num = int(train_frac*self.num_articles)
        test_art_num = self.num_articles - train_art_num
        train_art_ids = random.sample(self.art_ids, train_art_num)
        test_art_ids = [art_id for art_id in self.art_ids if art_id not in train_art_ids]
        
        train_dataset_ids, test_dataset_ids = [], []
        for (ds_idx, art_id) in enumerate(self.art_ids_map):
            if art_id in train_art_ids:
                train_dataset_ids.append(ds_idx)
            elif art_id in test_art_ids:
                test_dataset_ids.append(ds_idx)
        
        return self.select(train_dataset_ids).shuffle(seed=seed), self.select(test_dataset_ids).shuffle(seed=seed)
        
    
    def train_test_split_per_article(self, train_frac, seed):
        """Split the dataset per article; Keep train_frac of the comments of each article in the train set, and the rest in the test set.
            Thus all articles stay in both training and test set"""
        assert 0.0 < train_frac < 1.0
        
        train_dataset_ids = []
        test_dataset_ids = []
        for art_id in self.art_ids:
            art_dataset_ids = list(np.where(np.array(self.art_ids_map) == art_id)[0])
            
            num_art_cmts = len(art_dataset_ids)
            train_cmt_num = int(train_frac*num_art_cmts)
            test_cmt_num = num_art_cmts - train_cmt_num
            train_ds_ids = random.sample(art_dataset_ids, train_cmt_num)
            test_ds_ids = [ds_id for ds_id in art_dataset_ids if ds_id not in train_ds_ids]
        
            train_dataset_ids.extend(train_ds_ids)
            test_dataset_ids.extend(test_ds_ids)
        
        return self.select(train_dataset_ids).shuffle(seed=seed), self.select(test_dataset_ids).shuffle(seed=seed)
  
def preprocess_article_text(article_bodies):
    #Remove multiple newlines
    article_bodies = article_bodies.apply(lambda s: re.sub(r'(\n+\s*\n+)+' , r'\n', s))
    return article_bodies

def preprocess_comment_text(comment_bodies):
    return comment_bodies
