#!/bin/bash
#SBATCH --job-name=d3d_train
#SBATCH --account=beqw-tgirails
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --mem=320G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --output=/projects/beqw/ahmadmoo/decide3d_runs/logs/train.%j.out

# sbatch slurm/train.slurm A, then sbatch slurm/train.slurm B. A stopped job resumes from <out_dir>/<stage>/last.pt
# Both stages in one job, keeps the GPUs for B: sbatch slurm/train.slurm "A B"
# Other config: sbatch slurm/train.slurm A configs/other.yaml
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
cd $HOME/decide3d
for stage in ${1:?stage}; do
  $HOME/.conda/envs/decide3d/bin/torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) -m decide3d.train ${2:-configs/train.yaml} $stage
done
