#!/bin/bash
#SBATCH --job-name=beng2800_fixes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

cd /home/rag88/projects/tests/2800
python3 scripts/30_hpc_issue_fixes.py
