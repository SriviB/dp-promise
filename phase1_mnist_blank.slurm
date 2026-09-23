#!/bin/bash
#SBATCH -J phase1_blank_audit
#SBATCH -o status/phase1_blank_%j.o
#SBATCH -e status/phase1_blank_%j.e
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:30:00
#SBATCH -A ASC25102

module load cuda/12.4

eval "$(conda shell.bash hook)"
conda activate dp_promise
cd $WORK/dp-promise

mkdir -p status results/test_run_slurm

echo "=== Running SLURM Sanity Check ==="
python -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

python parallel_audit_model.py \
    --config configs/dp_promise/mnist_28/eps10.0/config.yaml \
    --data_name mnist \
    --model_name unet \
    --n_reps 2 \
    --n_epochs 1 \
    --n_df 32 \
    --batch_size 32 \
    --eval_noise_samples 2 \
    --target_type blank \
    --seed 0 \
    --holdout_audit \
    --store_all_losses \
    --out results/test_run_slurm

echo "=== Sanity Check Completed ==="
