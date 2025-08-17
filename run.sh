#!/bin/bash

#SBATCH --job-name=test_job              
#SBATCH --output=output_%j.log           
#SBATCH --error=error_%j.log             
#SBATCH --partition=nvidia               
#SBATCH --nodes=1                        
#SBATCH --ntasks=1                       
SBATCH --cpus-per-task=16               
SBATCH --gres=gpu:nvidia:4              
#SBATCH --mem=256G                       
#SBATCH --time=00:05:00        

srun make clean
srun make USE_CUDA=ON
srun --gres=gpu:nvidia:2 make test-cpp