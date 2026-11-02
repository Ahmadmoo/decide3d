#!/bin/bash
#SBATCH --job-name=d3d_moge
#SBATCH --account=beqw-tgirails
#SBATCH --partition=gpu_a100
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/projects/beqw/ahmadmoo/cavqa/logs/moge.%A_%a.out

# Frames -> MoGe-2 geometry, one GPU per part. A stopped part continues: sbatch slurm/moge.slurm
set -euo pipefail
export PYTHONNOUSERSITE=1
WORK=/projects/beqw/ahmadmoo/cavqa
$HOME/.conda/envs/decide3d/bin/python $HOME/decide3d/preprocessing/moge_geometry.py $WORK/frames $WORK/geom --part=$SLURM_ARRAY_TASK_ID/${SLURM_ARRAY_TASK_COUNT}
