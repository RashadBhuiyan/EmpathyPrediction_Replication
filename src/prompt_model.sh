#!/bin/bash
#SBATCH --account=def-cfwelch
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1:59:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mail-user=bhuiyr2@mcmaster.ca
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=logs/slurm/model_results.out

module load python/3.11
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load cudnn/9.2.1.18
module load arrow/18.1.0

source ../env/bin/activate
srun python LLMPrompt.py