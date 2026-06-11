#!/bin/bash -l
#SBATCH --job-name=iqa-a100
#SBATCH --partition=a100
#SBATCH --clusters=tinygpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=22:00:00
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --mail-user=arashmousavi193@gmail.com
#SBATCH --mail-type=ALL

set -euo pipefail

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

python -m scripts.sweep_sft --config config/sft_sweep.json --resume
python -m scripts.sweep_grpo --config config/grpo_sweep.json --resume

# To run evaluation on A100 instead, replace the command above with:
# python -m scripts.evaluate --config config/eval.json
