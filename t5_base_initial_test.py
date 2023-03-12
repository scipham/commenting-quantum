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

from manage_data import CustomDataset, preprocess_article_text, preprocess_comment_text
from torch.utils.data import DataLoader

#from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, Trainer, TrainingArguments
#require sentencepiece
import wandb


    
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return None

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Make sure we are logged in to wandb
    wandb.login()
    
    wandb.init(project="transformers_t5_alice_summarization_test")
    # save the trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"]="true"
    # turn off watch to log faster
    os.environ["WANDB_WATCH"]="false"
    
    
    art_filepath = "/home/s1930443/MRP1/data/r_all_deduplicated_postselected_articles.csv"
    cmt_filepath = "/home/s1930443/MRP1/data/r_all_deduplicated_comments.csv"
    
    # Store some key paramters to the WandB config for later reference
    config = wandb.config          # Initialize config
    config.HF_MODEL_ID = "t5-base"
    config.TRAIN_BATCH_SIZE = 2    # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 2    # inpexut batch size for testing (default: 1000)
    config.TRAIN_EPOCHS = 1        # number of epochs to train (default: 10)
    config.VAL_EPOCHS = 1 
    config.LEARNING_RATE = 1e-4    # learning rate (default: 0.01)
    config.SEED = 42               # random seed (default: 42)
    config.MAX_SOURCE_LEN = 512           # maximum length (num. of tokens) of the input text
    config.MAX_TARGET_LEN = 200        # maximum length (num. of tokens) of the output text
    

    # Set random seeds and deterministic pytorch for reproducibility
    random.seed(config.SEED)
    torch.manual_seed(config.SEED) 
    np.random.seed(config.SEED) 
    torch.backends.cudnn.deterministic = True

    # Importing and pre-processing the data
    task_prefix = "summarize: "
    art_df = pd.read_csv(art_filepath, sep=";")
    ps_art_df = art_df[art_df['selected'] == 1].reset_index()
    del art_df
    ps_art_df = ps_art_df[['article_id', "headline",'body', "comments_ids"]]
    
    cmt_df = pd.read_csv(cmt_filepath).sort_values(by=['article_id', 'level', 'date'], ascending=[True, True, True])
    tld_cmt_df = cmt_df[cmt_df['level'] == 0].reset_index()
    del cmt_df
    tld_cmt_df = tld_cmt_df[['comment_id','article_id', 'body', 'level']]

    # Clean and preprocess data before creating the dataset!
    ps_art_df["body"] = preprocess_article_text(ps_art_df.loc[:,"body"])
    tld_cmt_df["body"] = preprocess_comment_text(tld_cmt_df.loc[:,"body"])
    
    
    #Add the task prefix here to the article body text
    ps_art_df["body"] = task_prefix + ps_art_df["body"] 

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(config.HF_MODEL_ID, model_max_length=512)

    # Creating the Training and Validation dataset for further creation of Dataloader
    dataset_inst = CustomDataset(ps_art_df, tld_cmt_df, tokenizer, config.MAX_SOURCE_LEN, config.MAX_TARGET_LEN)
    #val_dataset = CustomDataset(val_set, tokenizer, config.MAX_LEN, )
    
    #Train validation split 
    trainset_ratio = 0.8
    start_time = time.time()
    print("Starting train test split...")
    train_dataset, val_dataset = dataset_inst.train_test_split(train_frac=trainset_ratio, seed=config.SEED)
    # train_dataset, val_dataset = dataset_inst.train_test_split_on_articles(train_frac=trainset_ratio, shuffle=True, seed=config.SEED)
    print("Train test split succesfull! Took {} seconds for splitting {} dataset entries into {} and {}".format(time.time() - start_time, len(dataset_inst), len(train_dataset), len(val_dataset)))
    
    #train_dataset=df.sample(frac=train_size,random_state = config.SEED)
    #train_dataset = train_dataset.reset_index(drop=True)
    #val_dataset=df.drop(train_dataset.index).reset_index(drop=True)

    
    #TODO: Added a Language model layer on top for generation of Summary?? 
    custom_t5config = T5Config.from_pretrained(config.HF_MODEL_ID) 
    model = T5ForConditionalGeneration.from_pretrained(config.HF_MODEL_ID, config=custom_t5config)

    wandb.watch(model, log="all")
    
    #model = model.to(device)
    device_map = {
    0: [0, 1, 2],
    1: [3, 4, 5],
    2: [ 6, 7, 8],
    3: [9, 10, 11]
    }
    model.parallelize(device_map)

    wandb.watch(model, log="all")
    
    should_finetune = True
    if should_finetune:
        print('Initiating Fine-Tuning for the model on our dataset')

        training_args = TrainingArguments(
                        output_dir='/data1/s1930443/hf_models',
                        report_to="wandb",
                        logging_steps=5, 
                        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
                        per_device_eval_batch_size=config.VALID_BATCH_SIZE,
                        learning_rate=config.LEARNING_RATE,
                        num_train_epochs=config.TRAIN_EPOCHS,
                        #max_steps = 100,
                        save_strategy="epoch",
                        #save_steps = 100,
                        do_eval=True,
                        evaluation_strategy="steps", 
                        eval_steps=20,
                        seed=config.SEED,
                        )
        
        # define the trainer and start training
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            #compute_metrics=compute_metrics,
        )
        trainer.train()


        #trainer.save_model("/data1/s1930443/hf_models/")

    
    print("Successfully finished and closing now. Goodbye!")
    #Finish the wandb run; necessary in notebooks
    wandb.finish()
    
    
if __name__ == '__main__':
    main()
