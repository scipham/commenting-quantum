# Importing stock libraries
import os
import random
import numpy as np
import pandas as pd
import re
import time
from tqdm import tqdm
from tqdm.auto import trange

# Importing the ML and transformers libraries
import torch
from torch import cuda
from manage_data import CustomDataset, preprocess_article_text, preprocess_comment_text
from torch.utils.data import DataLoader

#from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers.optimization import AdamW, get_constant_schedule
from accelerate import Accelerator, find_executable_batch_size

#require sentencepiece
import wandb

    
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return None


def validate_model(model, tokenizer, val_loader, val_epochs, val_batch_size):
    
    model.eval()
    
    tot_val_loss = 0

    for epoch in trange(val_epochs, desc="Validation epochs", leave=None):
        with torch.no_grad():
            for (b_i, batch) in enumerate(tqdm(val_loader, desc="Validation batches", leave=None),0):
                #Prepare decoder inputs and labels
                label_ids = batch['label_ids']
                dec_input_ids = label_ids[:, :-1].contiguous() #Decoder input: Shift left, last token falls off
                target_ids = label_ids[:, 1:].clone().detach() #Decoder output: Shift right to match decoder input sequence length
                target_ids[label_ids[:, 1:] == tokenizer.pad_token_id] = -100 #Ignore pad tokens in loss calculation
                
                #Prepare (encoder) inputs 
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                
                outputs = model(input_ids = input_ids,
                                attention_mask = attention_mask,
                                decoder_input_ids = dec_input_ids,
                                labels = target_ids)
                
                loss = outputs.loss #outputs[0]
                batch_loss = loss.item()
                tot_val_loss += batch_loss
                
                logits = outputs.logits #outputs[?]
        
    avg_val_loss = tot_val_loss / (len(val_loader) * val_epochs)
    return avg_val_loss
                
                                


def evaluate_model(accelerator, model, tokenizer, eval_dataset, eval_batch_size, max_target_len):
    
    eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)
    eval_loader = accelerator.prepare(eval_loader)
    
    prediction_buffer = []
    target_buffer = []
    input_buffer = []
    
    model.eval()

    with torch.no_grad():
        for (b_i, batch) in enumerate(eval_loader,0):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            generated_ids = model.generate(input_ids = input_ids,
                                            attention_mask = attention_mask,
                                            max_length=max_target_len,
                                            #repetition_penalty=2.5, 
                                            #length_penalty=1.0, 
                                            temperature = 0.1,
                                            #top_k=,
                                            #top_p=,
                                            #num_beams=4, 
                                            )
            
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            target = tokenizer.batch_decode(batch['label_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            all_preds, all_target = accelerator.gather_for_metrics((preds, target))
            
            prediction_buffer.extend(all_preds)
            target_buffer.extend(all_target)

            input_text = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            input_buffer.extend(input_text)
    
    return prediction_buffer, target_buffer, input_buffer        
    

class DistributedTrainer():
    def __init__(self, model_or_model_id, tokenizer, compute_metrics,
                        output_dir='models',
                        report_to_wandb=False,
                        logging_steps=5, 
                        tot_batch_size=4,
                        learning_rate=1e-4,
                        num_train_epochs=4,
                        max_steps = 100,
                        save_strategy="epoch",
                        save_steps = None,
                        do_eval=True,
                        evaluation_strategy="steps", 
                        eval_steps=100,
                        val_epochs=1,
                        seed=None):
        """Note: the training and validation batch sizes need to be equal! """
        
        self.model_or_model_id = model_or_model_id
        self.tokenizer = tokenizer #The tokenizer to be used
        self.compute_metrics = compute_metrics #The function to be used to compute metrics
        self.output_dir = output_dir #Where to save the model checkpoints
        self.report_to_wandb= report_to_wandb #Whether to report to log training to wandb
        self.logging_steps=logging_steps #How often to log training metrics
        self.tot_batch_size=tot_batch_size #Batch size for training
        self.learning_rate=learning_rate #Learning rate in optimizer
        self.num_train_epochs=num_train_epochs #Number of training epochs; overrides max_steps if set
        self.max_steps = max_steps #Maximum number of training steps; is ignored if num_train_epochs is set
        self.save_strategy=save_strategy #Whether to save the model after each epoch or after a set number of steps: "epoch" or "steps"
        self.save_steps = save_steps #How often to save the model; is ignored if save_strategy is set to "epoch"
        self.do_eval=do_eval    #Whether to evaluate the model during training (significantly affects performance)
        self.evaluation_strategy=evaluation_strategy #Whether to evaluate the model after each epoch or after a set number of steps: "epoch" or "steps"
        self.eval_steps = eval_steps #How often to evaluate the model; is ignored if evaluation_strategy is set to "epoch"
        self.val_epochs = val_epochs #Maximum number of epchs to average loss over in validation
        self.seed = seed #Random seed for reproducibility
        

    def train(self, train_dataset, eval_dataset=None):
        """Initiates and performs the training loop.

        Args:
            train_dataset PytorchDataset: The training dataset
            eval_dataset PytorchDataset: The dataset to evaluate on during training. Defaults to None for the case where no evaluation is desired.
        """
        assert self.do_eval and eval_dataset is not None, "Contradiction: Evaluation is not enabled or no evaluation dataset is provided"
        
        accelerator = Accelerator(log_with="wandb") #Accelerator(split_batches=True, log_with="wandb")
        accelerator.init_trackers("my_project", config={"learning_rate": 1e-4})
        
        #@find_executable_batch_size(starting_batch_size = self.tot_batch_size)
        def perform_training(batch_size):
            nonlocal accelerator
            #Model import and configuration
            custom_t5config = T5Config.from_pretrained(self.model_or_model_id) 
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_or_model_id, config=custom_t5config)

            # Log metrics with wandb
            wandb.watch(self.model, log="all")

            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
            optimizer = torch.optim.AdamW(params = self.model.parameters(), lr=self.learning_rate)
            scheduler = get_constant_schedule(optimizer=optimizer)
            
            val_loader = None
            if eval_dataset is not None:
                val_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
                self.model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(self.model, optimizer, train_loader, val_loader, scheduler)
            else:
                self.model, optimizer, train_loader, scheduler = accelerator.prepare(self.model, optimizer, train_loader, scheduler)

            
            self.model.train() #Put the model in training mode
                    
            for epoch in trange(self.num_train_epochs, desc="Training epochs"):  
                print('Running epoch: {}'.format(epoch))
                
                tot_train_loss = 0
                
                for b_i, batch in enumerate(tqdm(train_loader, desc="Training batches", leave=None), 0):
                    #Prepare decoder inputs and labels
                    label_ids = batch['label_ids']
                    dec_input_ids = label_ids[:, :-1].contiguous() #Decoder input: Shift left, last token falls off
                    target_ids = label_ids[:, 1:].clone().detach() #Decoder output: Shift right to match decoder input sequence length
                    target_ids[label_ids[:, 1:] == self.tokenizer.pad_token_id] = -100 #Ignore pad tokens in loss calculation
                    
                    #Prepare (encoder) inputs 
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    
                    optimizer.zero_grad() #Clear gradients
                    outputs = self.model(input_ids = input_ids, 
                                        attention_mask = attention_mask, 
                                        decoder_input_ids=dec_input_ids, 
                                        labels=target_ids,)
                    
                    loss = outputs.loss #outputs[0]
                    batch_loss = loss.item()
                    tot_train_loss += batch_loss
                    logits = outputs.logits #outputs[?]
                    
                    if self.compute_metrics is not None:
                        step_train_metric = self.compute_metrics((logits, target_ids))
                        accelerator.log({"Training Metric": step_train_metric})
                    
                    
                    if b_i%self.logging_steps == 0 and self.report_to_wandb:
                        accelerator.log({"Training Loss": batch_loss})

                    if self.do_eval and self.evaluation_strategy == "steps":
                        if b_i%self.eval_steps == 0:
                            batch_val_loss = validate_model(model=self.model, tokenizer=self.tokenizer, val_loader=val_loader, val_epochs=self.val_epochs ,val_batch_size=batch_size)
                            self.model.train() #Put the model back in training mode
                            accelerator.log({"Validation Loss": batch_val_loss})
                            if self.compute_metrics is not None:
                                batch_val_metric = self.compute_metrics((logits.cpu(), target_ids.cpu()))
                                accelerator.log({"Validation Metric": batch_val_metric})
                    
                            
                    if self.save_strategy == "steps":
                        if b_i%self.save_steps == 0:
                            raise NotImplementedError("Saving model after each step is not implemented yet")
                    
                    if b_i%len(train_loader)==0:
                        print(f'Epoch: {epoch}, Loss:  {batch_loss:.3f}')
                    

                    accelerator.backward(loss) #Backpropagate loss
                    optimizer.step() #Update weights
                    scheduler.step() #Update learning rate
                    
                avg_train_loss = tot_train_loss/len(train_loader)        
                
                if self.save_strategy == "epoch":
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(self.model)
                    unwrapped_model.save_pretrained(self.output_dir)
                    print(f'Checkpoint in {self.output_dir}')
                
                print(f'Finished training epoch: {epoch}, Total training Loss:  {avg_train_loss:.3f}')
                
                if self.do_eval and self.evaluation_strategy == "epoch":
                    epoch_val_loss = validate_model(model=self.model, tokenizer=self.tokenizer, val_loadert=val_loader, val_epochs=self.val_epochs ,val_batch_size=batch_size)
                    self.model.train() #Put the model back in training mode
                    accelerator.log({"Validation Loss": epoch_val_loss})
                    if self.compute_metrics is not None:
                        epoch_val_metric = self.compute_metrics((logits.cpu(), target_ids.cpu()))
                        accelerator.log({"Validation Metric": epoch_val_metric})
        
        perform_training(self.tot_batch_size)
        
        unwrapped_model = accelerator.unwrap_model(self.model)
        accelerator.end_training()
        return unwrapped_model
                
            
    
def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Make sure we are logged in to wandb
    #os.environ["WANDB_API_KEY."]="925f0f3c8de42f022a0a5a390aab9845cb5c92cf"
    wandb.login()
    # WandB – Initialize a new run
    wandb.init(project="transformers_t5_alice_accelerate")
    
    # save the trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"]="true"
    # turn off watch to log faster
    os.environ["WANDB_WATCH"]="false"
    
    art_filepath = "/home/s1930443/MRP1/data/r_all_deduplicated_postselected_articles.csv"
    cmt_filepath = "/home/s1930443/MRP1/data/r_all_deduplicated_comments.csv"
    

    # Store some key paramters to the WandB config for later reference
    config = wandb.config          # Initialize config
    config.HF_MODEL_ID = "t5-base"
    config.TRAIN_BATCH_SIZE = 1    # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 1    # inpexut batch size for testing (default: 1000)
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

    
    print('Initiating Fine-Tuning for the model on our dataset')

    training_args = {"output_dir": '/data1/s1930443/hf_models',
                    "report_to_wandb": True,
                    "logging_steps": 5, 
                    "tot_batch_size": config.TRAIN_BATCH_SIZE,
                    "learning_rate": config.LEARNING_RATE,
                    "num_train_epochs": config.TRAIN_EPOCHS,
                    #"max_steps": 100,
                    "save_strategy": "epoch",
                    #"save_steps": 100,
                    "do_eval": True,
                    "evaluation_strategy": "steps", 
                    "eval_steps": 100, #10,
                    "val_epochs": config.VAL_EPOCHS,
                    "seed": config.SEED,
                    }
                
    
    # define the trainer and start training
    trainer = DistributedTrainer(model_or_model_id=config.HF_MODEL_ID, 
                                tokenizer=tokenizer,
                                compute_metrics=None,#compute_metrics,
                                **training_args
                                ) 
    
    model = trainer.train(train_dataset=train_dataset, eval_dataset=val_dataset)

    
    # Validation loop and saving the resulting file with predictions and acutals in a dataframe.
    # TODO: Replace validation via trainer OR generate method of the model
    print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
    

    prediction_buffer, target_buffer, input_buffer = evaluate_model(model=model, tokenizer=tokenizer, val_dataset=val_dataset, eval_batch_size=config.VALID_BATCH_SIZE, max_target_len=config.MAX_TARGET_LEN)
    final_df = pd.DataFrame({'Generated Text':prediction_buffer,'Input Text':input_buffer})
    final_df.to_csv('init_generated_comments.csv', index=False)

    
    print("Successfully finished and closing now. Goodbye!")
    #Finish the wandb run; necessary in notebooks
    
    #wandb.finish()
    
    
if __name__ == '__main__':
    main()