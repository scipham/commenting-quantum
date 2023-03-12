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
from transformers.trainer_utils import get_last_checkpoint
#require sentencepiece
import wandb
import deepspeed
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

    # distributed setup
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    loc_device = local_rank
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()

    print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
    
    hf_model_id = "t5-base"
    model_checkpoint_path = "/data1/s1930443/HF_MODELS/"
    ds_config_path = "/home/s1930443/.config/deepspeed/...."
    
    dschf = HfDeepSpeedConfig(ds_config_path)  # keep this object alive
    # now a model can be loaded.
    tokenizer = T5Tokenizer.from_pretrained(hf_model_id, model_max_length=512)

    model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir)
    model = load_state_dict_from_zero_checkpoint(model, checkpoint_dir) # Re-load fp32 version of model
    
    # initialise Deepspeed ZeRO and store only the engine object
    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()

    eval_dataset = Dataset.from_disk(model_checkpoint_path+"val_dataset")
    eval_loader = DataLoader(eval_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False)

    prediction_buffer = []
    target_buffer = []
    input_buffer = []

    with torch.no_grad():
        for (b_i, batch) in enumerate(eval_loader,0):
            print(len(batch['input_ids']), len(batch['label_ids']))
            print(len(batch['input_ids'].flatten()), len(batch['label_ids'].flatten()))
            generated_ids = ds_engine.module.generate(input_ids = batch['input_ids'].to(loc_device),
                                                    attention_mask = batch['attention_mask'].to(loc_device),
                                                    max_length=config.MAX_TARGET_LEN,
                                                    repetition_penalty=2.5, 
                                                    length_penalty=1.0, 
                                                    temperature = 0.0,
                                                    #top_k=,
                                                    #top_p=,
                                                    #num_beams=4, 
                                                    )
            
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            target = tokenizer.batch_decode(batch['label_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            #preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            #target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in val_dataset.label_ids]
            if b_i%100==0:
                print('Completed {} batches out of {}'.format(b_i, len(eval_loader)))
            
            input_text = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            input_buffer.extend(input_text)
            prediction_buffer.extend(preds)
            target_buffer.extend(target)

    final_df = pd.DataFrame({'Generated Text':prediction_buffer,'Input Text':input_buffer})
    final_df.to_csv('init_generated_comments.csv', index=False)


if __name__ == '__main__':
    main()
