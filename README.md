# commenting-quantum

This repo provides partial access to the datasets and code mentioned in the thesis "Exploring sentiment and engagement of comments on news articles about quantum science & technology using pre-trained transformers", available at {THEDIS PDF HABDLE, coming soon...]

Main contributions in this repo:
- The full Reddit dataset (without annotations):
/data/destilled/r_all_articles_complete_preprocessed & /data/destilled/r_all_comments_complete_preprocessed

- The full Shtetl Optimized [https://scottaaronson.blog/] dataset (without annotations):
/data/destilled/shtetl_optim_articles.csv & /data/destilled/shtetl_optim_comments.csv

- The stratified and annotated Reddit dataset (sentiment towards QS&T and engagement towards QS&T):
/data/annotated/r_art_stratified_annotated.csv & /data/annotated/r_cmt_stratified_annotated.csv

- Code for the scraper and data-structure that was used to retrieve & export the dataset (2 versions: Python & Julia)
data_collect_and_prep_scripts/data_struct.py or data_struct.jl

- Code for performing annotation of comments automatically via the OpenAI API:
/data_collect_and_prep_scripts/comment_annotation.ipynb

- Code for SVM predictions on pre-computed embeddings ("embedding checkpoints"):
/model_scripts/svm_predict_checkpointed_embeds.ipynb

- Original and separated image files for all figures used in the thesis:
/thesis_assets/

