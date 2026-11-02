#!/bin/bash
#SBATCH --job-name=d3d_frames
#SBATCH --account=beqw-tgirails
#SBATCH --partition=cpu_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/beqw/ahmadmoo/cavqa/logs/frames.%j.out

# Records -> frames. sbatch slurm/build_frames.slurm
set -euo pipefail
export PYTHONNOUSERSITE=1
WORK=/projects/beqw/ahmadmoo/cavqa
$HOME/.conda/envs/decide3d/bin/python $HOME/decide3d/preprocessing/build_frames.py $WORK/out $WORK/frames
