"""Auditing DP-SGD in black-box setting - parallel / distributed entry-point."""
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import argparse
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from omegaconf import OmegaConf
import dill

from utils.audit import compute_eps_lower_from_mia, compute_eps_lower_from_mia_given_t

from src.utils import get_unet_model, load_dataset_from_config
from src.trainers import DPPromiseTrainer
import torch.nn.functional as F

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def save_checkpoint(
    out_folder: str,
    outputs: dict,
    losses: dict,
    all_losses: dict,
    train_set_accs: list,
    test_set_accs: list,
    fit_world_only,
    rank: int = 0,
) -> None:
    """Persist current run state to disk. Each rank writes its own file."""
    os.makedirs(out_folder, exist_ok=True)
    suffix = f'_rank{rank}' if rank > 0 else ''

    random_state = {
        'np': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
    }
    with open(f'{out_folder}/random_state{suffix}.dill', 'wb') as f:
        dill.dump(random_state, f)

    if fit_world_only:
        w = fit_world_only
        np.save(f'{out_folder}/outputs_{w}{suffix}.npy', outputs[w])
        np.save(f'{out_folder}/losses_{w}{suffix}.npy', losses[w])
        if all_losses is not None:
            np.save(f'{out_folder}/all_losses_{w}{suffix}.npy', all_losses[w])
        if w == 'out':
            np.save(f'{out_folder}/train_set_accs{suffix}.npy', train_set_accs)
            np.save(f'{out_folder}/test_set_accs{suffix}.npy', test_set_accs)
    else:
        np.save(f'{out_folder}/outputs_in{suffix}.npy', outputs['in'])
        np.save(f'{out_folder}/outputs_out{suffix}.npy', outputs['out'])
        np.save(f'{out_folder}/losses_in{suffix}.npy', losses['in'])
        np.save(f'{out_folder}/losses_out{suffix}.npy', losses['out'])
        np.save(f'{out_folder}/train_set_accs{suffix}.npy', train_set_accs)
        np.save(f'{out_folder}/test_set_accs{suffix}.npy', test_set_accs)
        if all_losses is not None:
            np.save(f'{out_folder}/all_losses_in{suffix}.npy', all_losses['in'])
            np.save(f'{out_folder}/all_losses_out{suffix}.npy', all_losses['out'])


def init_run_state(out_folder: str, fit_world_only, rank: int = 0):
    """Initialise fresh run state and write an initial checkpoint."""
    outputs = {'out': [], 'in': []}
    losses = {'out': [], 'in': []}
    all_losses = {'in': [], 'out': []}
    train_set_accs = []
    test_set_accs = []

    os.makedirs(out_folder, exist_ok=True)
    save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, fit_world_only, rank)

    return outputs, losses, all_losses, train_set_accs, test_set_accs


def build_parser() -> argparse.ArgumentParser:
    """Return a fully configured ArgumentParser for parallel audit runs."""
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # ------------------------------------------------------------------
    # Distributed / system
    # ------------------------------------------------------------------
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for torchrun distributed training')

    # ------------------------------------------------------------------
    # Data and model
    # ------------------------------------------------------------------
    parser.add_argument('--data_name', type=str, default='mnist',
                        help='Dataset: mnist, cifar10, cifar100, purchase, tiny_shakespeare')
    parser.add_argument('--model_name', type=str, default='unet',
                        help='Model architecture')
    parser.add_argument('--n_df', type=int, default=0,
                        help='Dataset size |D| (0 = full dataset)')

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    parser.add_argument('--n_reps', type=int, default=100,
                        help='Total shadow models to train across in/out (e.g. 400 = 200 in + 200 out)')
    parser.add_argument('--n_epochs', type=int, default=None,
                        help='Number of training epochs per model (None = config default)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (None = config default)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (None = config default)')
    parser.add_argument('--block_size', type=int, default=None,
                        help='Process batch in sub-blocks to save GPU memory')
    parser.add_argument('--aug_mult', type=int, default=1,
                        help='Augmentation multiplicity')
    parser.add_argument('--sampling', type=str, default='poisson',
                        choices=['poisson', 'shuffle'],
                        help='Minibatch sampling strategy')

    # ------------------------------------------------------------------
    # Privacy
    # ------------------------------------------------------------------
    parser.add_argument('--epsilon', type=float, default=None,
                        help='DP ε budget (None = non-private)')
    parser.add_argument('--delta', type=float, default=1e-5,
                        help='DP δ budget')
    parser.add_argument('--max_grad_norm', type=float, default=1,
                        help='Per-sample gradient clipping norm')

    # ------------------------------------------------------------------
    # Canary / target sample
    # ------------------------------------------------------------------
    parser.add_argument('--target_type', type=str, default='blank',
                        help='Canary type: blank')
    parser.add_argument('--canary_pt', type=str, default=None,
                        help='Path to a .pt canary file; overrides --target_type')
    parser.add_argument('--blank_alpha', type=float, default=0.0,
                        help='Blank canary interpolation: 0 = all-zeros, 1 = label-9 image')

    # ------------------------------------------------------------------
    # Audit configuration
    # ------------------------------------------------------------------
    parser.add_argument('--seed', type=int, default=0,
                        help='Global random seed')
    parser.add_argument('--fixed_init', type=str, nargs='?', default=None, const='',
                        help='Fix model init across reps (optionally supply a weight path)')
    parser.add_argument('--fit_world_only', type=str, default=None,
                        choices=['in', 'out'],
                        help='Train only "in" or only "out" models')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level for empirical ε estimation')
    parser.add_argument('--holdout_audit', action='store_true',
                        help='Hold out half the reps for threshold selection')
    parser.add_argument('--store_all_losses', action='store_true',
                        help='Save per-sample training losses for every rep')
    parser.add_argument('--out', type=str, default='exp_data/',
                        help='Output directory')

    # ------------------------------------------------------------------
    # DP-PROMISE / Diffusion specific flags
    # ------------------------------------------------------------------
    parser.add_argument('--config', type=str, default='configs/dp_promise/mnist_28/eps10.0/config.yaml',
                        help='Path to experiment config YAML')
    parser.add_argument('--eval_noise_samples', type=int, default=32,
                        help='Number of noise samples to evaluate canary loss')

    return parser


def distribute_reps(n_reps, world_size):
    """Distribute repetitions across GPUs."""
    reps_per_gpu = [[] for _ in range(world_size)]
    for i in range(n_reps):
        reps_per_gpu[i % world_size].append(i)
    return reps_per_gpu


def compute_per_sample_losses(model, trainer, X, y, config, device, batch_size=256, num_eval_samples=4, eval_seed=999999):
    """Return per-sample diffusion denoising losses as a numpy array."""
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    losses = []
    model.eval()
    eval_gen = torch.Generator(device='cpu').manual_seed(eval_seed)
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            dummy_y = torch.zeros((batch_X.shape[0],), dtype=torch.long, device=device)
            batch_mse = torch.zeros(batch_X.shape[0], device=device)
            for _ in range(num_eval_samples):
                t = torch.randint(
                    config.dp.S,
                    config.diffusion.timesteps,
                    size=(batch_X.shape[0],),
                    generator=eval_gen,
                ).to(device)
                noise = torch.randn(batch_X.shape, generator=eval_gen).to(device)
                x_noisy = trainer.q_sample(batch_X, t, noise)
                pred_noise = model(x_noisy, t, dummy_y)
                per_sample = F.mse_loss(noise, pred_noise, reduction='none').mean(dim=[1, 2, 3])
                batch_mse += per_sample
            batch_mse /= num_eval_samples
            losses.append(batch_mse.cpu().numpy())
    return np.concatenate(losses)


def train_model(model_name, X, y, X_target, y_target, epsilon, delta, max_grad_norm,
               n_epochs, lr, block_size, batch_size, init_model=None, out_dim=10, aug_mult=1,
               gradient_space_audit=False, crafted_gradient=None, defense=False, defense_k: int = 5,
               defense_apply_ascent=False, defense_filter_every: int = 1, device='cuda:0',
               generator=None, dl_generator=None, rank=0, world_size=None,
               defense_score_norm='linf', defense_score_fn='grad_norm',
               loss_volatility_k: int = 5, grad_norm_percentile_k: int = 20,
               grad_dir_volatility_k: int = 5, grad_dir_proj_dim: int = 64,
               grad_dir_proj_seed: int = 0, rand_proj_var_m: int = 10,
               rand_proj_var_seed: int = 0, maxmin_proj_k: int = 10,
               maxmin_proj_seed: int = 0, grad_rank_mode: str = 'effdim',
               grad_rank_eps: float = 1e-12, grad_accel_proj_dim: int = 64,
               grad_accel_proj_seed: int = 0, grad_jerk_proj_dim: int = 64,
               grad_jerk_proj_seed: int = 0, dir_unique_k: int = 5,
               alignment_proj_k: int = 10, alignment_proj_seed: int = 0,
               grad_scatter_k: int = 5, num_workers: int = 4,
               persistent_workers: bool = True, return_defense_state: bool = False,
               sampling: str = 'poisson', config=None):
    """
    Train a single model on a single GPU (no DDP).
    """

    # Move everything to the specified device
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    
    if init_model is None:
        model = get_unet_model(config).to(device)
    else:
        model = copy.deepcopy(init_model).to(device)

    model.train()
    trainer = DPPromiseTrainer(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=dl_generator,
        drop_last=True
    )

    # Phase 1 training loop
    for epoch in range(n_epochs):
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            # Unconditional / class-free diffusion training for Phase 1
            y_batch = torch.zeros_like(y_batch, dtype=torch.long).to(device)

            loss = trainer(model, X_batch, y_batch, phase="1")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model, trainer


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    
    # Check if running under torchrun (distributed mode)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )
        
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(device)
            print(f'[Rank {rank}] Using device: {torch.cuda.get_device_name(local_rank)}')
        else:
            device = torch.device('cpu')
            print(f'[Rank {rank}] CUDA not available, using CPU')
    else:
        # Single GPU mode (no distributed training)
        local_rank = 0
        rank = 0
        world_size = 1
        
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)
            print(f'Single GPU mode - Using device: {torch.cuda.get_device_name(0)}')
        else:
            device = torch.device('cpu')
            print(f'Single GPU mode - Using CPU')
    
    # Parse arguments — base parser is shared across all entry-points
    parser = build_parser()
    args = parser.parse_args()
    if args.epsilon == -1:
        args.epsilon = None
    if args.max_grad_norm == -1:
        args.max_grad_norm = None

    # Load configuration
    config = OmegaConf.load(args.config)
    if args.lr is None:
        args.lr = float(config.train.lr1)
    if args.batch_size is None:
        args.batch_size = int(config.train.batch_size1)
    if args.n_epochs is None:
        args.n_epochs = int(config.train.epochs1)
    if args.delta is None:
        args.delta = float(config.dp.delta)

    out_folder = f'{args.out}/{args.data_name}_{args.model_name}_eps{args.epsilon}'
    os.makedirs(out_folder, exist_ok=True)

    if rank == 0:
        print('Loading data')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x * 2.0 - 1.0,
    ])
    base_dataset = load_dataset_from_config(config, transform, train=True)

    all_imgs = []
    all_labels = []
    temp_loader = DataLoader(base_dataset, batch_size=1024, shuffle=False)
    for imgs, labels in temp_loader:
        all_imgs.append(imgs)
        all_labels.append(labels)
    X_out = torch.cat(all_imgs, dim=0)
    y_out = torch.cat(all_labels, dim=0)
    if args.n_df > 0:
        X_out = X_out[:args.n_df]
        y_out = y_out[:args.n_df]

    # Initialize model with SAME seed across all GPUs for fixed_init
    if rank == 0:
        print('Initializing model')
    init_model = None
    if args.fixed_init is not None:
        # Use same seed for all GPUs to ensure identical initialization
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        init_model = get_unet_model(config)
    
    # NOW set per-rank seeds for everything else (after init_model is created)
    # This ensures data loading and other operations are still independent per GPU
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    # Craft target
    if rank == 0:
        print('Crafting target data point')

    if args.target_type == 'blank':
        blank_img = torch.full_like(X_out[[0]], -1.0)
        if args.blank_alpha > 0:
            label_9_indices = (y_out == 9).nonzero(as_tuple=True)[0]
            if len(label_9_indices) == 0:
                raise ValueError("No label 9 samples found in dataset")
            label_9_img = X_out[label_9_indices[0]].unsqueeze(0)
            target_X = args.blank_alpha * label_9_img + (1 - args.blank_alpha) * blank_img
        else:
            target_X = blank_img
        target_y = torch.tensor([0], dtype=torch.long)
    else:
        raise Exception(f'Target {args.target_type} not supported for Phase 1 audit')

    # Define datasets
    X_in, y_in = torch.vstack((X_out[:-1], target_X)), torch.cat((y_out[:-1], target_y))

    if rank == 0:
        print('Training models')
    
    # Initialize run state (no resume support)
    worlds = [args.fit_world_only] if args.fit_world_only else ['in', 'out']
    
    outputs, losses, all_losses, train_set_accs, test_set_accs = init_run_state(
        out_folder, args.fit_world_only, rank)

    # Distribute repetitions across GPUs
    reps_per_gpu = distribute_reps(args.n_reps // 2, world_size)
    my_reps = reps_per_gpu[rank]
    
    if rank == 0:
        print(f"Rep distribution: {[len(r) for r in reps_per_gpu]}")
    
    for world in worlds:
        curr_X, curr_y = (X_out, y_out) if world == 'out' else (X_in, y_in)
        
        # Each rank trains its assigned models
        for rep_idx, rep in enumerate(my_reps):
            print(f"[Rank {rank}] Training rep {rep_idx+1}/{len(my_reps)} (global rep {rep})")
            
            # Create unique generators for each repetition
            # Use rep (global repetition number) to ensure uniqueness across all GPUs
            generator = torch.Generator().manual_seed(args.seed + rep * 2)
            dl_generator = torch.Generator().manual_seed(args.seed + rep * 2 + 1)
            
            model, trainer = train_model(
                args.model_name, 
                curr_X, 
                curr_y, 
                target_X, 
                target_y, 
                args.epsilon, 
                args.delta,
                args.max_grad_norm, 
                args.n_epochs, 
                args.lr, 
                args.block_size, 
                args.batch_size,
                init_model=init_model,
                device=device,
                generator=generator,
                dl_generator=dl_generator,
                rank=rank,
                world_size=world_size,
                config=config,
            )
            
            # Compute outputs and losses
            model.eval()
            with torch.no_grad():
                target_X_device = target_X.to(device)
                target_y_device = target_y.to(device)
                
                # Evaluate diffusion denoising reconstruction loss on canary
                eval_gen = torch.Generator(device='cpu').manual_seed(999999)
                total_mse = 0.0
                for _ in range(args.eval_noise_samples):
                    t = torch.randint(
                        config.dp.S,
                        config.diffusion.timesteps,
                        size=(1,),
                        generator=eval_gen,
                    ).to(device)
                    noise = torch.randn(
                        target_X_device.shape,
                        generator=eval_gen,
                    ).to(device)
                    x_noisy = trainer.q_sample(target_X_device, t, noise)
                    pred_noise = model(x_noisy, t, target_y_device)
                    total_mse += F.mse_loss(noise, pred_noise).item()
                
                loss = -(total_mse / args.eval_noise_samples)
                output = torch.tensor([loss])
                
                # Store locally - no gathering needed
                outputs[world].append(output.cpu().numpy())
                losses[world].append(loss)

            if args.store_all_losses:
                all_losses[world].append(compute_per_sample_losses(model, trainer, curr_X, curr_y, config, device=device))

            save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, args.fit_world_only, rank)

            del model, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Convert to numpy arrays
    for world in worlds:
        outputs[world] = np.array(outputs[world])
        losses[world] = np.array(losses[world])

    # Save final rank-specific results
    save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, args.fit_world_only, rank)

    # Wait for all processes to finish before combining results
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.barrier()

    # Rank 0 combines all results
    if rank == 0:
        print("\nCombining results from all GPUs...")
        combined_outputs = {'in': [], 'out': []}
        combined_losses = {'in': [], 'out': []}
        combined_all_losses = {'in': [], 'out': []}
        combined_train_accs = []
        combined_test_accs = []
        
        for r in range(world_size):
            suffix = f'_rank{r}' if r > 0 else ''
            try:
                if not args.fit_world_only:
                    combined_outputs['in'].extend(np.load(f'{out_folder}/outputs_in{suffix}.npy'))
                    combined_outputs['out'].extend(np.load(f'{out_folder}/outputs_out{suffix}.npy'))
                    combined_losses['in'].extend(np.load(f'{out_folder}/losses_in{suffix}.npy'))
                    combined_losses['out'].extend(np.load(f'{out_folder}/losses_out{suffix}.npy'))
                    if args.store_all_losses and os.path.exists(f'{out_folder}/all_losses_in{suffix}.npy'):
                        combined_all_losses['in'].extend(np.load(f'{out_folder}/all_losses_in{suffix}.npy', allow_pickle=True))
                    if args.store_all_losses and os.path.exists(f'{out_folder}/all_losses_out{suffix}.npy'):
                        combined_all_losses['out'].extend(np.load(f'{out_folder}/all_losses_out{suffix}.npy', allow_pickle=True))
                else:
                    combined_outputs[args.fit_world_only].extend(np.load(f'{out_folder}/outputs_{args.fit_world_only}{suffix}.npy'))
                    combined_losses[args.fit_world_only].extend(np.load(f'{out_folder}/losses_{args.fit_world_only}{suffix}.npy'))
            except FileNotFoundError:
                print(f"Warning: Could not find results for rank {r}")
        
        # Save combined results
        if not args.fit_world_only:
            np.save(f'{out_folder}/outputs_in.npy', combined_outputs['in'])
            np.save(f'{out_folder}/outputs_out.npy', combined_outputs['out'])
            np.save(f'{out_folder}/losses_in.npy', combined_losses['in'])
            np.save(f'{out_folder}/losses_out.npy', combined_losses['out'])
            if args.store_all_losses:
                np.save(f'{out_folder}/all_losses_in.npy', np.array(combined_all_losses['in'], dtype=object))
                np.save(f'{out_folder}/all_losses_out.npy', np.array(combined_all_losses['out'], dtype=object))
        else:
            np.save(f'{out_folder}/outputs_{args.fit_world_only}.npy', combined_outputs[args.fit_world_only])
            np.save(f'{out_folder}/losses_{args.fit_world_only}.npy', combined_losses[args.fit_world_only])
        
        if not args.fit_world_only:
            def audit_canary(losses, args):
                # Convert to numpy arrays for indexing
                losses_in = np.array(losses['in'])
                losses_out = np.array(losses['out'])
                n = len(losses_in)
                t_losses = {'in': None, 'out': None}
                holdout_losses = {'in': None, 'out': None}

                if args.holdout_audit:
                    # Use random sampling for holdout split to avoid ordering effects
                    np.random.seed(args.seed)  # Use same seed for reproducibility
                    indices = np.random.permutation(n)
                    threshold_indices = indices[:n // 2]
                    holdout_indices = indices[n // 2:]
                    
                    t_losses['in'] = losses_in[threshold_indices]
                    t_losses['out'] = losses_out[threshold_indices]
                    holdout_losses['in'] = losses_in[holdout_indices]
                    holdout_losses['out'] = losses_out[holdout_indices]
                else:
                    # No holdout - use all data for threshold selection
                    t_losses['in'] = losses_in
                    t_losses['out'] = losses_out

                # Calculate empirical epsilon using GDP
                mia_scores = np.concatenate([t_losses['in'], t_losses['out']])
                mia_labels = np.concatenate([np.ones_like(t_losses['in']), np.zeros_like(t_losses['out'])])

                max_t, emp_eps_loss, _ = compute_eps_lower_from_mia(mia_scores, mia_labels, args.alpha, args.delta, 'GDP', n_procs=1)

                if args.holdout_audit:
                    emp_eps_loss, _ = compute_eps_lower_from_mia_given_t(np.concatenate(
                        [holdout_losses['in'], holdout_losses['out']]), 
                        np.concatenate([np.ones_like(holdout_losses['in']), np.zeros_like(holdout_losses['out'])]), 
                        args.alpha, 
                        args.delta, 
                        max_t, 
                        'GDP')
                
                return emp_eps_loss, mia_scores, mia_labels
            
            emp_eps_loss, mia_scores, mia_labels = audit_canary(combined_losses, args)

            np.save(f'{out_folder}/emp_eps_loss.npy', [emp_eps_loss])
            np.save(f'{out_folder}/mia_scores.npy', mia_scores)
            np.save(f'{out_folder}/mia_labels.npy', mia_labels)
        
            print(f'Theoretical eps: {args.epsilon}')
            print(f'Empirical eps: {emp_eps_loss}')

    print(f"[Rank {rank}] Finished!")

    # Only destroy process group if we initialized it (distributed mode)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    except Exception as e:
        print(f'\nError in main: {str(e)}')
        import traceback
        traceback.print_exc()
