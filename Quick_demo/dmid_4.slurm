#!/bin/bash
#SBATCH --job-name=radfm_test
#SBATCH --output=radfm_test_%j.out
#SBATCH --error=radfm_test_%j.err
#SBATCH --partition=normal                 # partition (queue)
#SBATCH --gres=gpu:nvidia_h100_pcie:1      # request 1 H100 GPU
#SBATCH --cpus-per-task=8                  # CPU cores
#SBATCH --mem=64G                          # memory
#SBATCH --time=01:00:00                    # max walltime (hh:mm:ss)

# ----------------------------
# Environment setup
# ----------------------------
module load anaconda
source activate radfm   # or: conda activate radfm

# ----------------------------
# Go to project directory
# ----------------------------
cd /lustre/fs1/home/ab575577/projects_fall_2025/mammochat/RadFM/Quick_demo

# ----------------------------
# Run Python script
# ----------------------------
python test_exp4_dmid.py
