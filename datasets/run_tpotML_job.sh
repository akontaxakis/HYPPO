#!/bin/bash
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --mem=32G                     # Total memory for the job
#SBATCH --time=24:00:00              # Time limit (hh:mm:ss)


#Load necessary modules
module load releases/2022b
module load Python/3.10.8-GCCcore-12.2.0

# Load Python from your local directory
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH

# Run your Python script
python3 /home/users/a/k/akontaxa/tpotML.py ${DATASET_ID} ${ID} ${K} ${N}