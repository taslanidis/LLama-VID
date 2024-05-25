#!/bin/bash

#SBATCH --gpus=1
#SBATCH --job-name=InstallEnvironment
#SBATCH --ntasks=1
#SBATCH --gres=gpumem:50g
#SBATCH --cpus-per-task=18
#SBATCH --time=00:15:00
#SBATCH --output=install_environment-%A.out

#exit if an error occurs
set -e 

module purge
module load StdEnv
module load eth_proxy
module load cuda/11.7.0


source /cluster/scratch/kfanaras/miniconda3/etc/profile.d/conda.sh
conda create -n llamavid python=3.10 -y
conda activate llamavid

pip install --upgrade pip
pip install -e .

pip install openpyxl
pip install ninja
pip install flash-attn==2.1.2.post3 --no-build-isolation
pip install openai==0.28.0
pip install -U openai-whisper
conda install -c conda-forge code-server
