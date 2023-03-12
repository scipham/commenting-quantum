# Importing stock libraries
import os
import sys
import random
import numpy as np
import pandas as pd
import re
import time


# Importing the ML and transformers libraries
import torch
from torch import cuda

from manage_data import CustomDataset, preprocess_article_text, preprocess_comment_text
from torch.utils.data import DataLoader

#from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, Trainer, TrainingArguments

#require sentencepiece
import wandb
import deepspeed


def main(dataset_label, train_articles_filepath, train_comments_filepath, val_articles_filepath, val_comments_filepath,hf_model_id, train_batch_size, val_batch_size,  train_epochs, val_epochs, learning_rate, seed, max_source_len, max_target_len, wandb_project_name, deepspeed_config_filepath ):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Make sure we are logged in to wandb
    wandb.login()
    
    wandb.init(project=wandb_project_name,
               group="Group-" + str(time.time()),
               )
    
    # save the trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"]="true"

    # Store some key paramters to the WandB config for later reference
    config = wandb.config          # Initialize config
    config.HF_MODEL_ID = hf_model_id
    config.TRAIN_BATCH_SIZE = train_batch_size    # input batch size for training (default: 64)
    config.VAL_BATCH_SIZE = val_batch_size   # inpexut batch size for testing (default: 1000)
    config.TRAIN_EPOCHS = train_epochs        # number of epochs to train (default: 10)
    config.VAL_EPOCHS = val_epochs 
    config.LEARNING_RATE = learning_rate    # learning rate (default: 0.01)
    config.SEED = seed               # random seed (default: 42)
    config.MAX_SOURCE_LEN = max_source_len           # maximum length (num. of tokens) of the input text
    config.MAX_TARGET_LEN = max_target_len        # maximum length (num. of tokens) of the output text
    config.DATASET_LABEL = dataset_label

    # Set random seeds and deterministic pytorch for reproducibility
    random.seed(config.SEED)
    torch.manual_seed(config.SEED) 
    np.random.seed(config.SEED) 
    torch.backends.cudnn.deterministic = True

    # Importing and pre-processing the data
    task_prefix = "summarize: "
    
    train_art_df = pd.read_csv(train_articles_filepath, sep=";")
    train_art_df = train_art_df[['article_id', "headline",'body', "comments_ids"]]
    
    val_art_df = pd.read_csv(val_articles_filepath, sep=";")
    val_art_df = val_art_df[['article_id', "headline",'body', "comments_ids"]]
    
    train_cmt_df = pd.read_csv(train_comments_filepath, sep=";").sort_values(by=['article_id', 'level', 'date'], ascending=[True, True, True])
    train_tld_cmt_df = train_cmt_df[train_cmt_df['level'] == 0].reset_index()
    del train_cmt_df
    train_tld_cmt_df = train_tld_cmt_df[['comment_id','article_id', 'body', 'level']]
    
    val_cmt_df = pd.read_csv(val_comments_filepath, sep=";").sort_values(by=['article_id', 'level', 'date'], ascending=[True, True, True])
    val_tld_cmt_df = val_cmt_df[val_cmt_df['level'] == 0].reset_index()
    del val_cmt_df
    val_tld_cmt_df = val_tld_cmt_df[['comment_id','article_id', 'body', 'level']]
    
    # If you still need to preprocess data, do it here
    
    #Add the task prefix here to the article body text
    train_art_df["body"] = task_prefix + train_art_df["body"]
    val_art_df["body"] = task_prefix + val_art_df["body"]
    
    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(config.HF_MODEL_ID, model_max_length=512)

    # Creating the Training and Validation dataset for further creation of Dataloader
    train_dataset_inst = CustomDataset(train_art_df, train_tld_cmt_df, tokenizer, config.MAX_SOURCE_LEN, config.MAX_TARGET_LEN)
    val_dataset_inst = CustomDataset(val_art_df, val_tld_cmt_df, tokenizer, config.MAX_SOURCE_LEN, config.MAX_TARGET_LEN)

    custom_t5config = T5Config.from_pretrained(config.HF_MODEL_ID) 
    
    #For deepspeed we need to initialize the training arguments before loading the pretrained model
    training_args = TrainingArguments(
                    output_dir='/data1/s1930443/hf_models',
                    report_to="wandb",
                    logging_steps=5, 
                    per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
                    per_device_eval_batch_size=config.VAL_BATCH_SIZE,
                    learning_rate=config.LEARNING_RATE,
                    num_train_epochs=config.TRAIN_EPOCHS,
                    #max_steps = 100,
                    save_strategy="epoch",
                    #save_steps = 100,
                    do_eval=True,
                    evaluation_strategy="steps", 
                    eval_steps=20,
                    seed=config.SEED,
                    deepspeed=deepspeed_config_filepath,
                    )
    
    model = T5ForConditionalGeneration.from_pretrained(config.HF_MODEL_ID, config=custom_t5config)

    wandb.watch(model, log="all")
    
    print('Initiating Fine-Tuning for the model on the dataset')

    # define the trainer and start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        #compute_metrics=compute_metrics,
    )
    trainer.train()

    #To save the model after training with deepspeed, gather the parameters.
    checkpoint_dir = os.path.join(trainer.args.output_dir, "checkpoint-final")
    trainer.deepspeed.save_checkpoint(checkpoint_dir)
    model.cpu()
    trainer.save_model("/data1/s1930443/hf_models/")
    
    
    print("Successfully finished and closing now. Goodbye!")
    #Finish the wandb run; necessary in notebooks
    wandb.finish()
    
    
if __name__ == '__main__':
    
    script_kwargs = {}
    print("Arguments passed to the script: ", sys.argv)
    c_arg = None
    for arg in sys.argv[1:]: # skip the script name
        if "--" in arg:
            c_arg = arg.replace("--", "")
        elif c_arg:
            script_kwargs[c_arg] = arg
            c_arg = None
        else:
            raise ValueError("Invalid passing of arguments")
    
    print("Running script with the following arguments: ", script_kwargs)
    
    del script_kwargs["deepspeed"]
    #Need to convert the known non-string arguments to the correct type
    script_kwargs["train_batch_size"] = int(script_kwargs["train_batch_size"])
    script_kwargs["val_batch_size"] = int(script_kwargs["val_batch_size"])
    script_kwargs["train_epochs"] = int(script_kwargs["train_epochs"])
    script_kwargs["val_epochs"] = int(script_kwargs["val_epochs"])
    script_kwargs["learning_rate"] = float(script_kwargs["learning_rate"])
    script_kwargs["seed"] = int(script_kwargs["seed"])
    script_kwargs["max_source_len"] = int(script_kwargs["max_source_len"])
    script_kwargs["max_target_len"] = int(script_kwargs["max_target_len"])
    
    main(**script_kwargs)

"""
default_kwargs = {"hf_model_id": "t5-small",
"train_batch_size": 2,   
"val_batch_size": 2,   
"train_epochs": 1,       
"val_epochs": 1, 
"learning_rate": 1e-4,
"seed": 42,  
"max_source_len": 512,  
"max_target_len": 200,
}
"""