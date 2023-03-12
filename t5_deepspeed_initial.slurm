#!/bin/bash -l
#SBATCH --job-name=t5_deepspeed_initial_test
#SBATCH --output=%x_%j.out
#SBATCH --error==%x_%j.err
#SBATCH --mail-user="rothe@physics.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gpus=4


# load modules (assuming you start from the default environment)
module purge
module load CUDA/11.7.0
module load Python/3.10.4-GCCcore-11.3.0
module load Miniconda2/4.7.10

CWD_DIR=$(pwd)
echo "## Current dircectory $CWD_DIR"

echo "## Number of available CUDA devices: $CUDA_VISIBLE_DEVICES"

echo "## Checking status of CUDA device with nvidia-smi"
nvidia-smi

# Source the Python Conda virtual environment
# source /home/s1930443/.bashrc
conda activate T5-test-env
# mkdir -p /data1/s1930443/conda_pkgs/
# export CONDA_PKGS_DIRS=/data1/s1930443/conda_pkgs/
# conda env update --file t5_conda_environment.yml --prune

echo "[$SHELL] #### Starting Initial T5 Training Test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
export WANDB_API_KEY="925f0f3c8de42f022a0a5a390aab9845cb5c92cf"
WANDB_LOGGING_DIR=/data1/s1930443/wandb/
mkdir -p /data1/s1930443/.wandb_cache/
mkdir -p $WANDB_LOGGING_DIR
export WANDB_CACHE_DIR=/data1/s1930443/.wandb_cache/
export WANDB_DIR=$WANDB_LOGGING_DIR

mkdir -p /data1/s1930443/.hf_cache/
export TRANSFORMERS_CACHE=/data1/s1930443/.hf_cache/

mkdir -p /data1/s1930443/hf_models/
deepspeed_stage=3
deepspeed_config_filepath="/home/s1930443/.config/deepspeed/ds_config_zero$deepspeed_stage.json"
wandb_project_name="t5-deepspeed-alice-initial-test"
data_split_type="artsplit" # "artsplit" or "cmtsplit" or "simplesplit"
train_articles_filepath="/home/s1930443/MRP1/data/$data_split_type/r_art_train_$data_split_type.csv"
train_comments_filepath="/home/s1930443/MRP1/data/$data_split_type/r_cmt_train_$data_split_type.csv"
val_articles_filepath="/home/s1930443/MRP1/data/$data_split_type/r_art_val_$data_split_type.csv"
val_comments_filepath="/home/s1930443/MRP1/data/$data_split_type/r_cmt_val_$data_split_type.csv"
hf_model_id="t5-small"
train_batch_size=2   
val_batch_size=2  
train_epochs=1      
val_epochs=1
learning_rate=0.0001
seed=42  
max_source_len=512  
max_target_len=200

deepspeed t5_deepspeed_initial.py --deepspeed "/home/s1930443/.config/deepspeed/ds_config_zero$deepspeed_stage.json" \
--dataset_label $data_split_type \
--train_articles_filepath $train_articles_filepath \
--train_comments_filepath $train_comments_filepath \
--val_articles_filepath $val_articles_filepath \
--val_comments_filepath $val_comments_filepath \
--hf_model_id $hf_model_id \
--train_batch_size $train_batch_size \
--val_batch_size $val_batch_size \
--train_epochs $train_epochs \
--val_epochs $val_epochs \
--learning_rate $learning_rate \
--seed $seed \
--max_source_len $max_source_len \
--max_target_len $max_target_len \
--wandb_project_name $wandb_project_name \
--deepspeed_config_filepath $deepspeed_config_filepath \

echo "[$SHELL] ## Script finished"

conda deactivate
echo "[$SHELL] ## Deactivating environment and close job."
