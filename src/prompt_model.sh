#!/bin/bash
#SBATCH --account=def-cfwelch
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=3:59:00
#SBATCH --gres=gpu:h100:2
#SBATCH --mail-user=bhuiyr2@mcmaster.ca
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=logs/slurm/llm_results.out

module load python/3.11
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load cudnn/9.2.1.18
module load arrow/18.1.0

export HF_HOME=/scratch/rashadb/huggingface
export TRANSFORMERS_CACHE=/scratch/rashadb/huggingface
export HF_DATASETS_CACHE=/scratch/rashadb/huggingface
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ../env/bin/activate
python LLMPrompt.py