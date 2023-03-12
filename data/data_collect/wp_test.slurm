#!/bin/bash
#SBATCH --job-name=test_python_wp
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="rothe@physics.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=400M
#SBATCH --time=00:15:00
#SBATCH --partition=testing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# load modules (assuming you start from the default environment)
module load Python/3.10.4-GCCcore-11.3.0

# Source the Python virtual environment
source $HOME/MRP1/python_test_venv/bin/activate

echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
python3 mrp1_data_module.py
echo "[$SHELL] ## Script finished"

# Move output files to home data:
# mv /scratchdata/s1930443/$SLURM_JOB_ID/_articles.csv $HOME/MRP1/data/_articles.csv
# mv /scratchdata/s1930443/$SLURM_JOB_ID/_comments.csv $HOME/MRP1/data/_comments.csv

echo "[$SHELL] ## Data succesfully moved"