#!/bin/bash
#SBATCH --job-name=all_models
#SBATCH --account=def-cfwelch
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:h100:2
#SBATCH --array=1-3
#SBATCH --mail-user=bhuiyr2@mcmaster.ca
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=logs/slurm/model_results_%A_%a.out

module load python/3.11
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load cudnn/9.2.1.18
module load arrow/18.1.0

source ../env/bin/activate

echo "Starting run $SLURM_ARRAY_TASK_ID"

echo "Running all models with pooling mechanism: CLS"
srun python PPEPModel.py -p CLS
srun python CosineSimilarityModel.py -p CLS
srun python storyBModel.py -p CLS
srun python ContextModel.py -p CLS
srun python MultiDatasetTest.py -p CLS

echo "Running all models with pooling mechanism: MEAN"
srun python PPEPModel.py -p MEAN
srun python CosineSimilarityModel.py -p MEAN
srun python storyBModel.py -p MEAN
srun python ContextModel.py -p MEAN
srun python MultiDatasetTest.py -p MEAN