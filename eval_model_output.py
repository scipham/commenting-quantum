import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC, SVC

import torch
from torch import cuda
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchmetrics import Accuracy
from sentence_transformers import SentenceTransformer, models, losses, evaluation, InputExample

from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_metric
from torchmetrics.functional.text.rouge import rouge_score
from rouge import Rouge
from scipy.stats import hmean


def t5_online_metrics(eval_preds): #To be used only during training / fine-tuning of the pytorch model
    logits, labels = eval_preds
    
    cosine_similarity()
    rouge_metric = load_metric("rouge")
    
    predictions = np.argmax(logits, axis=-1)
    cos_sim = cosine_similarity(predictions, labels)
    rouge = rouge_metric.compute(predictions=predictions, references=labels)["recall"]
    
    return {"cosine_similarity": cos_sim, "rouge": rouge}

def transform_text_corpus_to_embeddings(gen_corpus, ref_corpus, encod_model_name):
    
    # model_name = "tf-idf", "sbert-base-uncased"
    if encod_model_name == "tf-idf":
        txt_prep_pipeline = Pipeline([
                    ('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ])
        
        #Note: Fit on both reference and generated comments to get a common vector encoding length
        #so make sure to pass both the reference and generated comments to this function
        txt_prep_pipeline.fit(np.concatenate((gen_corpus, ref_corpus)))
        return txt_prep_pipeline.transform(gen_corpus), txt_prep_pipeline.transform(ref_corpus)

    elif encod_model_name == "sbert":
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        encod_model = SentenceTransformer("all-mpnet-base-v2", device='cuda')
        #encod_model = encod_model.to(device)
        
        gen_corpus_uncased = [gen_cmt.lower() for gen_cmt in gen_corpus]
        ref_corpus_uncased = [ref_cmt.lower() for ref_cmt in ref_corpus]
        
        """
        gen_corpus_tokenized = tokenizer.batch_encode_plus(gen_corpus_uncased, return_tensors="pt", padding=True, truncation=True, max_length=512)
        ref_corpus_tokenized =  tokenizer.batch_encode_plus(ref_corpus_uncased, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        gen_corpus_dataset = TensorDataset(gen_corpus_tokenized["input_ids"], gen_corpus_tokenized["attention_mask"])
        ref_corpus_dataset = TensorDataset(ref_corpus_tokenized["input_ids"], ref_corpus_tokenized["attention_mask"])
        
        gen_corpus_dataloader = DataLoader(gen_corpus_dataset, batch_size=32, shuffle=False)
        ref_corpus_dataloader = DataLoader(ref_corpus_dataset, batch_size=32, shuffle=False)
        
        corpus_embeddings = {"gen_corpus": [], 
                             "ref_corpus": []}
        
        with torch.no_grad():
            for (dl_key, dl_inst) in zip(["gen_corpus", "ref_corpus"], [gen_corpus_dataloader, ref_corpus_dataloader]):
                for batch in dl_inst:
                
                    corpus_outputs = encod_model(batch["input_ids"].to(device),
                                                attention_mask=batch["attention_mask"].to(device),
                                                )
                    
                    corpus_mean_embeddings = torch.mean(corpus_outputs.last_hidden_state, dim=1).cpu() #.numpy()
                    
                    corpus_embeddings[dl_key].extend(corpus_mean_embeddings.tolist())
        #### /////// #####
        word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
        """
        
        corpus_embeddings = {"gen_corpus": encod_model.encode(gen_corpus_uncased, batch_size=32, show_progress_bar=True, device=device), 
                             "ref_corpus": encod_model.encode(ref_corpus_uncased, batch_size=32, show_progress_bar=True, device=device),
                             }
        # THe encode function already includes mean pooling of the last hidden state of the transformer model & tokenization with padding and truncation to (default) 128 tokens
        
        return corpus_embeddings["gen_corpus"], corpus_embeddings["ref_corpus"]
            
def cosine_similarity_score(generated_comments_df,reference_comments_df, relevant_article_ids=None):
    
    if relevant_article_ids is None:
        relevant_article_ids =  list(generated_comments_df["article_id"].unique())

    mean_max_cosine_scores = []
    for art_id in relevant_article_ids:
        art_ref_cmt_df = reference_comments_df[reference_comments_df["article_id"] == art_id]
        art_gen_cmt_df = generated_comments_df[generated_comments_df["article_id"] == art_id]
       
        gen_cmts_encod, ref_cmts_encod= transform_text_corpus_to_embeddings(art_gen_cmt_df["body"].values, 
                                                                            art_ref_cmt_df["body"].values, 
                                                                            encod_model_name="sbert",
                                                                            )
         
        cosine_sim_scores = cosine_similarity(gen_cmts_encod, ref_cmts_encod)
        
        mean_max_art_cosine_score = np.mean(np.max(cosine_sim_scores, axis=1))
        mean_max_cosine_scores.append(mean_max_art_cosine_score)

    return (relevant_article_ids, mean_max_cosine_scores)


def rouge_similarity_score( generated_comments_df,reference_comments_df, relevant_article_ids=None):
    
    txt_prep_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ])
    
    if relevant_article_ids is None:
        relevant_article_ids =  list(generated_comments_df["article_id"].unique())
    
    hmean_rouge1_scores = []
    hmean_rouge2_scores = []
    for art_id in relevant_article_ids:
        art_ref_cmt_df = reference_comments_df[reference_comments_df["article_id"] == art_id]
        art_gen_cmt_df = generated_comments_df[generated_comments_df["article_id"] == art_id]
        
        #rouge_scores = rouge_score(art_gen_cmt_df["body"].values[:, np.newaxis], art_ref_cmt_df["body"].values[np.newaxis, :])
        rouge_metric = Rouge()
        ref_cmts = list(art_ref_cmt_df["body"].values)
        gen_cmts = list(art_gen_cmt_df["body"].values)
        
        rouge1_max_fscores = []
        rouge2_max_fscores = []
        
        for (h_i, hypothesis) in enumerate(gen_cmts):
            scores = rouge_metric.get_scores([hypothesis,]*len(ref_cmts), ref_cmts)
            max_score = max(scores, key=lambda s: hmean([s['rouge-1']['f'], s['rouge-2']['f']])) # Want the best hypothesis-reference match. The f score is the harmonic mean of precision and recall, i.e. it tends to the lower value, which makes it a conservative measure.
            rouge1_max_fscores.append(max_score["rouge-1"]["f"])
            rouge2_max_fscores.append(max_score["rouge-2"]["f"])

        
        print(np.mean(rouge1_max_fscores), np.mean(rouge2_max_fscores))
        
        hmean_rouge1_scores.append(np.mean(rouge1_max_fscores))
        hmean_rouge2_scores.append(np.mean(rouge2_max_fscores))
    
    return (relevant_article_ids, hmean_rouge1_scores, hmean_rouge2_scores)


def in_out_svm_classif_correlation_score(generated_comments_df,reference_comments_df):
    """
    Calculates the correlation between the input and output of the model as dataframes.
    This is a measure of how much the model is learning, in what extent it actually uses 
    the input to generate the output and whether the output is characterizing the input.
    """
    
    
    assert all([gen_art_id in reference_comments_df["article_id"].values for gen_art_id in generated_comments_df["article_id"].values]), "Article ids of generated comments do not match with article ids in reference comments"
    
    art_id_label_map = {art_id: i for i, art_id in enumerate(np.unique(reference_comments_df["article_id"].values))}

    gen_comments_txt = generated_comments_df["body"].values
    ref_comments_txt = reference_comments_df["body"].values
    gen_comments_labels = np.array([art_id_label_map[art_id_val] for art_id_val in generated_comments_df["article_id"].values])
    ref_comments_labels = np.array([art_id_label_map[art_id_val] for art_id_val in reference_comments_df["article_id"].values])
    print("Found {} article ids in generated comments and {} article ids in reference comments".format(len(np.unique(gen_comments_labels)), len(np.unique(ref_comments_labels))))
    
    tot_comments_txt = np.concatenate([gen_comments_txt, ref_comments_txt]) 
    
    
    #txt_prep_pipeline.fit(tot_comments_txt) #Vocab should be derived from both hypotheses and references
    #gen_comments_encod, ref_comments_encod = txt_prep_pipeline.transform(gen_comments_txt), txt_prep_pipeline.transform(ref_comments_txt)
    gen_comments_encod, ref_comments_encod = transform_text_corpus_to_embeddings(gen_comments_txt, 
                                                                                 ref_comments_txt, 
                                                                                 encod_model_name="sbert")
    
    #Initialize the classifier model:

    predictions = None
    
    grid_search = False
    if grid_search:
            
        #clf = OneVsOneClassifier(LinearSVC(random_state=0))
        clf = OneVsOneClassifier(SVC(C=1.0) )
        
        parameters = {
        'estimator__kernel': ('poly', 'rbf', 'linear'), #Linear already tested
        #'estimator__tol': (1e-2, 1e-3, 1e-4),
        }
    
        gs_clf = GridSearchCV(clf, parameters, cv=2, n_jobs=-1, verbose=10)
    
        gs_clf.fit(ref_comments_encod, ref_comments_labels)

        predictions = gs_clf.predict(gen_comments_encod)

        print("===== Grid search results =====")
        print(gs_clf.cv_results_)
        print(gs_clf.best_score_)
        print(gs_clf.best_params_)
        
        print("=====================================")
        
    else:
        clf = OneVsOneClassifier(LinearSVC(C=1.0, random_state=0))
        #clf = OneVsOneClassifier(SVC(C=1.0, kernel="linear") )
        
        clf.fit(ref_comments_encod, ref_comments_labels)
        predictions = clf.predict(gen_comments_encod)
        
    predict_accuracy = np.mean(predictions == gen_comments_labels)

    print(predict_accuracy)
    print(metrics.classification_report(gen_comments_labels, predictions)) # labels=art_id_label_map.values() ,target_names=art_id_label_map.keys()

    print(metrics.confusion_matrix(gen_comments_labels, predictions))
    #[{"f1":, "recall":, "...":  }  for c_i in range(len(category_labels))]
    
    return predict_accuracy

class CustomSentenceDataset(Dataset):
    def __init__(self, input_sentences, labels, sentence_embedder):
        self.input_sentences = input_sentences
        self.sentence_embedder = sentence_embedder
        self.input_embeddings = self.sentence_embedder.encode(input_sentences)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #return self.input_sentences[idx], self.labels[idx]
        item = {'input_sentence': self.input_sentences[idx],
                'input_embedding': self.input_embeddings[idx],
                'label': self.labels[idx],
                }
        
        return item	
        

class SentenceClassifier(nn.Module):
    def __init__(self,sentence_embedding_length, num_labels, device):
        super(SentenceClassifier, self).__init__()
        
        self.num_labels = num_labels
        self.fc1_layer = nn.Linear(in_features=sentence_embedding_length,
                                   out_features=int(sentence_embedding_length/2),
                                   device=device)
        self.fc_out_layer = nn.Linear(in_features=int(sentence_embedding_length/2), #sentence_embedding_length,#
                                    out_features=num_labels, 
                                    device=device)
        self.dropout_layer = nn.Dropout(0.1)             

    def forward(self, x):
        x = self.dropout_layer(x)
        x = nn.functional.tanh(self.fc1_layer(x))
        #x = nn.functional.softmax(self.fc_out_layer(x), dim=1) #dim=1 is the node dimension
        x = self.fc_out_layer(x) #dim=1 is the node dimension
        return x


def train_classifier(model, train_dataloader, val_dataloader, num_train_epochs, device):
    # put model in train mode
    model.train()
    torch.set_grad_enabled(True)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    
    running_loss = 0.0
    
    for epoch in trange(num_train_epochs, desc="Training epochs"):  
        print('Running epoch: {}'.format(epoch+1))
        
        loss_buffer = []
        for b_i, batch in enumerate(tqdm(train_dataloader, desc="Training batches", leave=None), 0):
            
            inputs = batch['input_embedding'].to(device)
            labels = batch['label'].to(torch.int64) #.type(torch.LongTensor)
            labels = labels.to(device)
            
            # clear gradients
            optimizer.zero_grad()
            
            # Make predictions for this batch
            outputs = model(inputs)
            
            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            
            loss.backward()

            # update parameters
            optimizer.step()

            loss_buffer.append(loss.item())
            
        print("Loss for this epoch: {}".format(np.mean(loss_buffer)))
    
    return model
  
def evaluate_classifier(model, eval_dataloader, num_labels, device):
    model.eval()
    torch.set_grad_enabled(False)
    
    loss_fn = nn.CrossEntropyLoss()
    error_fn = Accuracy(task="multiclass", num_classes=num_labels)
    
    label_buffer = []
    prediction_buffer = []
    running_loss = 0.0
    
    for b_i, batch in enumerate(tqdm(eval_dataloader, desc="Evaluation batches", leave=None), 0):
        
        inputs = batch['input_embedding'].to(device)
        label_buffer.extend(batch['label'])
        labels = batch['label'].to(torch.int64)
        labels = labels.to(device)
        
        outputs = model(inputs)
        class_predict = torch.argmax(outputs, dim=1)
        prediction_buffer.extend(class_predict)
        
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()
    
    av_loss = running_loss / len(eval_dataloader)
    print("Average evaluation loss: {}".format(av_loss))
    
    classif_accuracy = error_fn(torch.tensor(prediction_buffer), torch.tensor(label_buffer))
    print("Classification accuracy: {}".format(classif_accuracy))
    
    return label_buffer, prediction_buffer, classif_accuracy
        
        
        
        
        
def in_out_sbert_classif_correlation_score(generated_comments_df,reference_comments_df):
    """
    Note: Although performance might be a bit better, we do not fine-tune the model on sentence similarity in our data.
    Instead we directly perform classification. Usually we would first pre-train on sentence similarity and then fine-tune on classification.
    Here we make convenient use of the model being already pre-trained on sentence similarity on a large (but different) corpus.
    Especially, if we were to use normal BERT or RoBERTa, we would have to pre-train on sentence similarity first. 
    """
    
    assert all([gen_art_id in reference_comments_df["article_id"].values for gen_art_id in generated_comments_df["article_id"].values]), "Article ids of generated comments do not match with article ids in reference comments"
    
    art_id_label_map = {art_id: i for i, art_id in enumerate(np.unique(reference_comments_df["article_id"].values))}

    
    gen_comments_txt = generated_comments_df["body"].values
    ref_comments_txt = reference_comments_df["body"].values
    gen_comments_labels = np.array([art_id_label_map[art_id_val] for art_id_val in generated_comments_df["article_id"].values])
    ref_comments_labels = np.array([art_id_label_map[art_id_val] for art_id_val in reference_comments_df["article_id"].values])
    print("Found {} article ids in generated comments and {} article ids in reference comments".format(len(np.unique(gen_comments_labels)), len(np.unique(ref_comments_labels))))
    
    # Prepare the training data and dataloader:
    gen_comments_uncased = [gen_cmt.lower() for gen_cmt in gen_comments_txt]
    ref_comments_uncased = [ref_cmt.lower() for ref_cmt in ref_comments_txt]
    
    #for gen_cmt, cmt_art_label in zip(gen_comments_uncased, gen_comments_labels):
    #    if len(gen_cmt) != 1:
    #        print(type(gen_cmt))
    
    ref_train_examples = [InputExample(texts=[ref_cmt], label=cmt_art_label) for ref_cmt, cmt_art_label in zip(ref_comments_uncased, ref_comments_labels)]
    gen_val_examples = [InputExample(texts=[gen_cmt], label=cmt_art_label) for gen_cmt, cmt_art_label in zip(gen_comments_uncased, gen_comments_labels)]
    
    ref_train_dataloader = DataLoader(ref_train_examples, shuffle=True, batch_size=4)
    gen_val_dataloader = DataLoader(gen_val_examples, shuffle=False, batch_size=4)

    # Prepare the classification model:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #encod_model = SentenceTransformer("all-mpnet-base-v2", device='cuda')
    #encod_model = encod_model.to(device)

    #sentence_embedding_model = models.SentenceTransformer('sentence-transformers/all-distilroberta-v1', max_seq_length=512)
    #word_embedding_model = models.Transformer('roberta-base-uncased', max_seq_length=512)
    #pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    #dense_model = models.Dense(in_features=sentence_embedding_model.get_sentence_embedding_dimension(), #pooling_model.get_sentence_embedding_dimension(),  #Note: get_sentence_embedding_dimension() and get_word_embedding_dimension() are only available for specific models each. These are SentenceTransformer specific functions!!!
    #                           out_features=len(art_id_label_map), 
    #                           activation_function=nn.Softmax()) #nn.Tanh() or nn.Softmax()

    #classif_model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
    #classif_model = SentenceTransformer(modules=[sentence_embedding_model, dense_model])
    embedding_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1', device=device)
    print("Sentence embedding length>", embedding_model.get_sentence_embedding_dimension())	
    
    # Define the objective/loss for the task:
    embedding_loss_function = losses.BatchAllTripletLoss(model=embedding_model, 
                                                        #sentence_embedding_dimension=classif_model.get_sentence_embedding_dimension(), 
                                                        #num_labels=len(art_id_label_map),
                                                        )
                
    # Fine-tune the classification model on the reference comments:
    torch.set_grad_enabled(True)
    
    embedding_model.fit(train_objectives=[(ref_train_dataloader, embedding_loss_function)],
                    epochs=4,
                    steps_per_epoch=None,
                    show_progress_bar=True,
                    )
    
    ref_train_examples = CustomSentenceDataset(ref_comments_uncased, ref_comments_labels, embedding_model)
    gen_val_examples = CustomSentenceDataset(gen_comments_uncased, gen_comments_labels, embedding_model)
    
    ref_train_dataloader = DataLoader(ref_train_examples, shuffle=True, batch_size=16)
    gen_val_dataloader = DataLoader(gen_val_examples, shuffle=False, batch_size=16)

    device = torch.device('cpu')
    #Train a classification model:
    classif_model = SentenceClassifier(embedding_model.get_sentence_embedding_dimension(),
                                       num_labels=len(art_id_label_map),
                                       device=device)
    
    classif_model =  train_classifier(classif_model, 
                                      ref_train_dataloader, 
                                      gen_val_dataloader,
                                      num_train_epochs=12,
                                      device=device)

    # Evaluate the classification model on the generated comments:

    val_labels, val_predictions, classif_accuracy = evaluate_classifier(classif_model, 
                                                                        gen_val_dataloader, 
                                                                        num_labels=len(art_id_label_map),
                                                                        device=device)
    print(val_labels, val_predictions, classif_accuracy)
    results = classif_accuracy
    #print(results)           
    #print(predict_accuracy)
    #print(metrics.classification_report(gen_comments_labels, predictions)) # labels=art_id_label_map.values() ,target_names=art_id_label_map.keys()

    #print(metrics.confusion_matrix(gen_comments_labels, predictions))
    
    clf = OneVsOneClassifier(LinearSVC(C=1.0, random_state=0))
    #ref_comments_encod = [batch["input_embedding"].re for b_i, batch in enumerate(ref_train_dataloader)]
    #gen_comments_encod = [batch["input_embedding"].reshape(-1) for b_i, batch in enumerate(gen_val_dataloader)]
    ref_comments_encod = embedding_model.encode(ref_comments_uncased, batch_size=32, show_progress_bar=True, device=device)
    gen_comments_encod = embedding_model.encode(gen_comments_uncased, batch_size=32, show_progress_bar=True, device=device)
    clf.fit(ref_comments_encod, ref_comments_labels)
    svm_predictions = clf.predict(gen_comments_encod)
    svm_predict_accuracy = np.mean(svm_predictions == gen_comments_labels)
    print("SVM accuracy: ", svm_predict_accuracy)
    
    return results

    

def model_collect_scores(gen_cmt_df, ref_cmt_df):
    model_score_dict = {}
    
    sbert_class_accuracy = in_out_sbert_classif_correlation_score(gen_cmt_df, ref_cmt_df)
    model_score_dict["sbert_class_accuracy"] = sbert_class_accuracy
    
    print("======= DONE WITH SBERT CLASSIFICATION ======== ")
    # Cosine similarity score
    relevant_art_ids, mean_max_cosine_scores = cosine_similarity_score(gen_cmt_df, ref_cmt_df)                                                             

    print("Mean cosine similarity score over all articles: ", np.mean(mean_max_cosine_scores))

    model_score_dict["cosine_similarity"] = np.mean(mean_max_cosine_scores)

    
    # Rouge similarity score
    relevant_art_ids, rouge1_scores, rouge2_scores = rouge_similarity_score(gen_cmt_df, ref_cmt_df )

    print("Mean Rouge-1 score over all articles: ", np.mean(rouge1_scores))
    print("Worst Rouge-1 score over all articles: ", np.min(rouge1_scores))
    print("Mean Rouge-2 score over all articles: ", np.mean(rouge2_scores))
    print("Worst Rouge-2 score over all articles: ", np.min(rouge2_scores))

    model_score_dict["rouge1_score"] = np.mean(rouge1_scores)
    model_score_dict["rouge2_score"] = np.mean(rouge2_scores)

    # In-out correlation score by SVM classification
    svm_class_accuracy = in_out_svm_classif_correlation_score(gen_cmt_df, ref_cmt_df)
    model_score_dict["svm_class_accuracy"] = svm_class_accuracy
    
    return model_score_dict 

def modelset_collect_scores(modelset):
    #Buffers for final scores (to be printed in a table)
    score_buffer = [] #Append a dictionary per model. Dictionary should contain: "model_name", "cosine_score", "rouge1_score", "rouge2_score" and "in_out_correlation_score"
    
    for model in modelset:    
        print("Collecting scores for model: ", model["label"], " ...")
        score_buffer.append(model_collect_scores(model["path_to_generated_comments"], model["path_to_reference_comments"]))


def load_complete_split_reddit_data(split_type, return_eval=False):
    assert split_type in ["artsplit", "cmtsplit", "simplesplit"], "Invalid split type"
    # ------- Data import -------
    train_art_filepath = "S:\\Sync\\University\\2023_MRP_1\\MRP1_WorkDir\\data\\destilled\\{}\\r_art_train_{}.csv".format(split_type,split_type)
    train_art_df = pd.read_csv(train_art_filepath, sep=";")

    val_art_filepath = "S:\\Sync\\University\\2023_MRP_1\\MRP1_WorkDir\\data\\destilled\\{}\\r_art_val_{}.csv".format(split_type,split_type)
    val_art_df = pd.read_csv(val_art_filepath, sep=";")

    eval_art_df = None
    if return_eval:
        eval_art_filepath = "S:\\Sync\\University\\2023_MRP_1\\MRP1_WorkDir\\data\\destilled\\{}\\r_art_eval_{}.csv".format(split_type,split_type)
        eval_art_df = pd.read_csv(eval_art_filepath, sep=";")
    
    #print(len(ps_art_df[ps_art_df["num_tld_comments"] > 100]))

    train_cmt_filepath = "S:\\Sync\\University\\2023_MRP_1\\MRP1_WorkDir\\data\\destilled\\{}\\r_cmt_train_{}.csv".format(split_type,split_type)
    train_cmt_df = pd.read_csv(train_cmt_filepath, sep=";").sort_values(by=['article_id', 'level', 'date'], ascending=[True, True, True])
    train_tld_cmt_df = train_cmt_df[train_cmt_df['level'] == 0]

    val_cmt_filepath = "S:\\Sync\\University\\2023_MRP_1\\MRP1_WorkDir\\data\\destilled\\{}\\r_cmt_val_{}.csv".format(split_type,split_type)
    val_cmt_df = pd.read_csv(val_cmt_filepath, sep=";").sort_values(by=['article_id', 'level', 'date'], ascending=[True, True, True])
    val_tld_cmt_df = val_cmt_df[val_cmt_df['level'] == 0]

    eval_tld_cmt_df = None
    if return_eval:
        eval_cmt_filepath = "S:\\Sync\\University\\2023_MRP_1\\MRP1_WorkDir\\data\\destilled\\{}\\r_cmt_eval_{}.csv".format(split_type,split_type)
        eval_cmt_df = pd.read_csv(eval_cmt_filepath, sep=";").sort_values(by=['article_id', 'level', 'date'], ascending=[True, True, True])
        eval_tld_cmt_df = eval_cmt_df[eval_cmt_df['level'] == 0]
    
    if return_eval:
        return train_art_df, val_art_df, eval_art_df, train_tld_cmt_df, val_tld_cmt_df, eval_tld_cmt_df
    else:
        return train_art_df, val_art_df, train_tld_cmt_df, val_tld_cmt_df

def load_limited_split_data(split_type):
    assert split_type in ["artsplit", "cmtsplit", "simplesplit"], "Invalid split type"
    # ------- Data import -------
    train_art_filepath = "S:\\Sync\\University\\2023_MRP_1\\MRP1_WorkDir\\data\\destilled\\{}\\r_limited_art_train_{}.csv".format(split_type,split_type)
    train_art_df = pd.read_csv(train_art_filepath, sep=";")

    val_art_filepath = "S:\\Sync\\University\\2023_MRP_1\\MRP1_WorkDir\\data\\destilled\\{}\\r_limited_art_val_{}.csv".format(split_type,split_type)
    val_art_df = pd.read_csv(val_art_filepath, sep=";")

    #print(len(ps_art_df[ps_art_df["num_tld_comments"] > 100]))

    train_cmt_filepath = "S:\\Sync\\University\\2023_MRP_1\\MRP1_WorkDir\\data\\destilled\\{}\\r_limited_cmt_train_{}.csv".format(split_type,split_type)
    train_cmt_df = pd.read_csv(train_cmt_filepath, sep=";").sort_values(by=['article_id', 'level', 'date'], ascending=[True, True, True])
    train_tld_cmt_df = train_cmt_df[train_cmt_df['level'] == 0]

    val_cmt_filepath = "S:\\Sync\\University\\2023_MRP_1\\MRP1_WorkDir\\data\\destilled\\{}\\r_limited_cmt_val_{}.csv".format(split_type,split_type)
    val_cmt_df = pd.read_csv(val_cmt_filepath, sep=";").sort_values(by=['article_id', 'level', 'date'], ascending=[True, True, True])
    val_tld_cmt_df = val_cmt_df[val_cmt_df['level'] == 0]

    return train_art_df, val_art_df, train_tld_cmt_df, val_tld_cmt_df
    # ------------------------

#if __name__ == "__main__":
