#!/bin/bash -l
#SBATCH --job-name=iqa
#SBATCH --partition=rtx3080
#SBATCH --clusters=tinygpu
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --mail-user=arashmousavi193@gmail.com
#SBATCH --mail-type=ALL

. ~/.bashrc
source activate vlm-iqa-26

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

export HF_HOME=/home/woody/iwi5/iwi5255h/.cache/huggingface
export TRANSFORMERS_CACHE=/home/woody/iwi5/iwi5255h/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/home/woody/iwi5/iwi5255h/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$HF_HOME"
cd /home/woody/iwi5/iwi5255h/ldct_iqa_mos_based


python -m scripts.evaluate --config config/eval.json














. ~/.bashrc
source activate vlm-iqa-26

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

export HF_HOME=/home/woody/iwi5/iwi5255h/.cache/huggingface
export TRANSFORMERS_CACHE=/home/woody/iwi5/iwi5255h/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/home/woody/iwi5/iwi5255h/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$HF_HOME"
cd /home/woody/iwi5/iwi5255h/ldct_iqa_mos_based

python -m scripts.evaluate --config config/eval.json
