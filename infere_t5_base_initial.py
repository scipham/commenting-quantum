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
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, Trainer, TrainingArguments
#require sentencepiece

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    hf_model_id = "t5-base"
    model_checkpoint_path = "/data1/s1930443/HF_MODELS/"
    
    #art_filepath = "S:\\Sync\\University\\2023_MRP_1\\MRP1_WorkDir\\data\\per_subreddit\\r_all_deduplicated_postselected_articles.csv"
    #cmt_filepath = "S:\\Sync\\University\\2023_MRP_1\\MRP1_WorkDir\\data\\per_subreddit\\r_all_deduplicated_comments.csv"
    
    MODEL_CHECKPOINT_ID = "t5-small"# "models/checkpoint-3498"
    HF_MODEL_ID = "t5-small"
    SEED = 42
    MAX_SOURCE_LEN = 512
    MAX_TARGET_LEN = 200
    EVAL_BATCH_SIZE = 4
    
    # Set random seeds and deterministic pytorch for reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED) 
    np.random.seed(SEED) 
    torch.backends.cudnn.deterministic = True

    # Load input data
    tokenizer = T5Tokenizer.from_pretrained(HF_MODEL_ID, model_max_length=512)
    
    custom_t5config = T5Config.from_pretrained(HF_MODEL_ID) 
    model = T5ForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT_ID, config=custom_t5config)
    device_map = {
                    0: [0, 1, 2],
                    1: [3, 4, 5],
                    2: [ 6, 7, 8],
                    3: [9, 10, 11]
                    }
    model.parallelize(device_map)
    
    # ----- Load data and perform inference ------
    
    #First on a simple example:
    encoder_input_texts = ["""summarize: IBM has taken its first step towards selling computers that are millions of times faster than the one you're reading this on.
The company has set up a new division, IBM Q, that is intended to make quantum computers and sell them commercially.
Until now, quantum computers have mostly been a much hyped but long away dream. But IBM believes they are close enough to reality to start work on getting software ready for when they become commercially available.
Quantum computers rely on quantum mechanics and the bizarre behaviour of quantum bits to do calculations far faster than any computer yet known.
Most researchers agree that the theoretical machines are still years away from actually being made, despite advances towards creating a working quantum computer. But IBM says that its work has shown enough promise to get working on programming the machines.
So far, IBM has demonstrated systems that use quantum effects in small-scale demonstrations. They have taken advantage of effects like superpositioning, which means that electrons can exist in two states at the same time – behaviour that could in the future be harnessed to allow them to work in far more complex ways than the 1s and 0s that are used in today's computers.
That work has shown enough promise that IBM executives believe such computers could soon be made commercially available, they said. As such, the software and the technologies that will be required to run on those computers must be prepared now, they said.
As such it will create APIs that allow developers and programmers to start working on machines that will be able to send programmes between quantum computers and traditional ones. Later this year it will also release a simulator that can mimic the kind of circuits that will be inside quantum computers, allowing people to get to work making software for them.
Eventually those technologies could be put to work in industries like drug and materials discovery or as yet unimagined artificial intelligence.
“Classical computers are extraordinarily powerful and will continue to advance and underpin everything we do in business and society. But there are many problems that will never be penetrated by a classical computer. To create knowledge from much greater depths of complexity, we need a quantum computer,” said Tom Rosamilia, senior vice president of IBM Systems. “We envision IBM Q systems working in concert with our portfolio of classical high-performance systems to address problems that are currently unsolvable, but hold tremendous untapped value.”
"""]
    
    input_encodings = tokenizer.batch_encode_plus(batch_text_or_text_pairs=encoder_input_texts, 
                                                                max_length=MAX_SOURCE_LEN, 
                                                                padding='max_length',
                                                                truncation=True,
                                                                return_tensors='pt')

    input_ids, attention_mask = input_encodings['input_ids'], input_encodings['attention_mask']
    
    model.eval()
    
    results = model.generate(input_ids=input_ids, 
                             attention_mask=attention_mask, 
                             max_length=MAX_TARGET_LEN)
    print(results)
    print(tokenizer.decode(results[0], skip_special_tokens=True))
    
    
    #Then on a dataset:
    eval_dataset = Dataset.from_disk(model_checkpoint_path+"val_dataset")
    model.eval()

    eval_loader = DataLoader(eval_dataset, 
                             batch_size=EVAL_BATCH_SIZE, 
                             shuffle=False)

    prediction_buffer = []
    target_buffer = []
    input_buffer = []

    with torch.no_grad():
        for (b_i, batch) in enumerate(eval_loader,0):
            print(len(batch['input_ids']), len(batch['label_ids']))
            print(len(batch['input_ids'].flatten()), len(batch['label_ids'].flatten()))
            generated_ids = model.generate(input_ids = batch['input_ids'].to(device),
                                            attention_mask = batch['attention_mask'].to(device),
                                            max_length=MAX_TARGET_LEN,
                                            #repetition_penalty=2.5, 
                                            #length_penalty=1.0, 
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