#!/bin/bash
#SBATCH --job-name=video-classification-small
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --output=.slurm/video-classification-small.out.txt
#SBATCH --error=.slurm/video-classification-small.err.txt

source /ikerlariak/osainz006/venvs/collie/bin/activate

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export WANDB_PROJECT=TEDD1104

CONFIGS_FOLDER="configs/video-classification"

torchrun --standalone --master_port 37223 --nproc_per_node=4 train_TEDD1104.py ${CONFIGS_FOLDER}/TEDD1104-small.yaml
