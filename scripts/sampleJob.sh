#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=100g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --partition=compchemq
#SBATCH --qos=compchem
#SBATCH --nodelist=compchem002
#SBATCH --reservation=scych2_9
module load anaconda-uon/3
cd $SLURM_SUBMIT_DIR
source activate ckfyp
python python_file_to_execute.py
conda deactivate
