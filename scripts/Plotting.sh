#!/bin/sh
#SBATCH -J MakePlots
#SBATCH -o hxiong1.o%j
#SBATCH -e hxiong1.e%j
#SBATCH -p skx
#SBATCH -N 1                    # Number of nodes
#SBATCH -n 1                    # Number of tasks (just 1 for OpenMP)
#SBATCH -c 1                   # CPUs per task (16 threads)
#SBATCH -t 05:00:00             # Walltime

module load python

source ~/RyanKick_env/bin/activate
python ~/sailfish/scripts/make_plot.py --indir /scratch/10824/hxiong1/Runs/LargeDomainLossKick\
  --outdir /scratch/10824/hxiong1/Plot-LargeDomain \
  --follow=True --radius 20000 \