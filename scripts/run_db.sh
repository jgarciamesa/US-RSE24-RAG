#!/bin/bash
#SBATCH -p general
#SBATCH -q public
#SBATCH --time=0-00:15:00
#SBATCH --gres=gpu:a100:1

#set up the environment
module load mamba/latest
source activate genai

python scripts/create_db.py example/llm_papers example/llm_db scripts/config.yaml
