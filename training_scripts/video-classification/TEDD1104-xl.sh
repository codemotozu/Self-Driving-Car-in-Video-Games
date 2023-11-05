#!/bin/bash
#SBATCH --job-name=video-classification-xl
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --output=.slurm/video-classification-xl.out.txt
#SBATCH --error=.slurm/video-classification-xl.err.txt

source /ikerlariak/igarcia945/envs/pytorch2/bin/activate

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export WANDB_PROJECT=TEDD1104_vmae
export OMP_NUM_THREADS=16

CONFIGS_FOLDER="configs/video-classification"

torchrun --standalone --master_port 37223 --nproc_per_node=4 train_TEDD1104.py ${CONFIGS_FOLDER}/TEDD1104-xl.yaml
