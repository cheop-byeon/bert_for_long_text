#!/bin/bash
#SBATCH --job-name=cls
#SBATCH --account=12345678
#SBATCH --time=00-20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=accel 
#SBATCH --gpus=1

export WANDB_MODE=offline

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module use -a /cluster/shared/nlpl/software/eb/etc/all/
module swap nlpl-huggingface-hub/0.11.0-gomkl-2021a-Python-3.9.5 nlpl-pytorch/1.11.0-gomkl-2021a-cuda-11.3.1-Python-3.9.5
module load nlpl-transformers/4.24.0-gomkl-2021a-Python-3.9.5
module load nlpl-nlptools/2022.01-gomkl-2021a-Python-3.9.5
module load nlpl-wandb/0.13.1-gomkl-2021a-Python-3.9.5

# short
# python3 infer.py --model_path "./AB/short/bmp.pt" --model_type "AB"
# python3 infer.py --model_path "./AplusB/short/bmp.pt" --model_type "AplusB21"

# long
# python3 infer.py --model_path "./AB/long/bmp.pt" --model_type "AB"
# python3 infer.py --model_path "./AplusB/long/bmp.pt"  --model_type "AplusB"
