#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=1024
#SBATCH --mail-type=BEGIN,END

python train.py -dataroot ""/cluster/home/rvecchiarell/pix2pix_ffd/U_Scaled/"" -name U_Lmass --model pix2pix -direction BtoA -display_id 0 -use_wandb -wandb_project_name SemesterProject -n_epochs 1 -n_epochs_decay 1