#!/bin/bash
#SBATCH --job-name=d3d_eval
#SBATCH --account=beqw-tgirails
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/beqw/ahmadmoo/decide3d_runs/logs/eval.%j.out

# Temperatures on val-dev, report on val-test -> report.json, final.pt: sbatch slurm/evaluate.slurm
set -euo pipefail
export PYTHONNOUSERSITE=1
cd $HOME/decide3d
$HOME/.conda/envs/decide3d/bin/python -m decide3d.evaluate configs/train.yaml ${1:-/projects/beqw/ahmadmoo/decide3d_runs/main/B/last.pt}
