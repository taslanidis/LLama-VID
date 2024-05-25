#!/bin/bash

#SBATCH --gpus=1
#SBATCH --job-name=InstallLaMA3
#SBATCH --ntasks=1
#SBATCH --gres=gpumem:50g
#SBATCH --cpus-per-task=18
#SBATCH --time=00:40:00
#SBATCH --output=install_lamma3-instruct-%A.out

#exit if an error occurs
set -e 

module purge
module load StdEnv
module load eth_proxy
module load cuda/11.7.0


source /cluster/scratch/kfanaras/miniconda3/etc/profile.d/conda.sh

conda create -n llama_3_instruct python=3.10 -y
source activate llama_3_instruct

pip install --upgrade pip
pip install -r requirements_llama_three.txt

pip install openpyxl
pip install transformers==4.40.0
conda install -c conda-forge code-server
