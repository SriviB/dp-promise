#!/bin/bash
#SBATCH -J phase1_blank_audit
#SBATCH -o status/phase1_blank_%j.o
#SBATCH -e status/phase1_blank_%j.e
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH -A ASC25102

module load cuda/12.4

eval "$(conda shell.bash hook)"
conda activate dp_promise
cd $WORK/dp-promise

mkdir -p status results/phase1_mnist_blank_400reps

echo "=== Running DP-PROMISE Phase 1 Blank Canary Audit (400 Reps) ==="
python -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

python parallel_audit_model.py \
    --config configs/dp_promise/mnist_28/eps10.0/config.yaml \
    --data_name mnist \
    --model_name unet \
    --n_reps 400 \
    --target_type blank \
    --seed 0 \
    --holdout_audit \
    --store_all_losses \
    --out results/phase1_mnist_blank_400reps

echo "=== Audit Completed ==="
