#!/bin/bash
#SBATCH -p gen-a100.p
#SBATCH --gres=gpu:a100:1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -J r019_features
#SBATCH -o /path/to/workspace/logs/r019_features.%j.out
#SBATCH -e /path/to/workspace/logs/r019_features.%j.err
# R019 E-a input: frozen RAD-DINO tokens of TAIX-Ray train + val (test only after the freeze).
ulimit -u 65536
source /path/to/home/myconda/etc/profile.d/conda.sh
conda activate spectra-gpu
# NumPy madvises large arrays for huge pages; with THP defrag=madvise every fault can stall in compaction
export NUMPY_MADVISE_HUGEPAGE=0
export HF_HOME=/path/to/home/cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
mkdir -p /tmp/spectra-no-mps-$(id -u)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/spectra-no-mps-$(id -u) CUDA_MPS_LOG_DIRECTORY=/tmp/spectra-no-mps-$(id -u) TMPDIR=/tmp
cd /path/to/spectra
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/r019_features.py --config config/taix.yaml --out ../runs/R019/features/taix --splits train \
       --batch 256 --workers 28
