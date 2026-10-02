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
from torch.utils.data import TensorDataset, DataLoader

from models import Models
from utils.data import load_data
from utils.dpsgd import clip_and_accum_grads, DefenseConfig
from utils.audit import compute_eps_lower_from_mia, compute_eps_lower_from_mia_given_t
from utils.accounting import get_noise_multiplier
from utils.canaries import craft_clipbkd, craft_gradient, fgsm_attack, choose_worstcase_label
from utils.training import (
    AugmentationFunction, IndexedTensorDataset,
    xavier_init_model, init_wideresnet,
    test_model, compute_per_sample_losses,
)
from utils.checkpoint import save_checkpoint, init_run_state
from utils.args import build_parser

import torch.nn.functional as F

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

try:
    from opacus.grad_sample import GradSampleModule as _GradSampleModule
    _OPACUS_AVAILABLE = True
except ImportError:
    _GradSampleModule = None
    _OPACUS_AVAILABLE = False


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
               sampling: str = 'poisson'):
    """
    Train a single model on a single GPU (no DDP).
    """

    # Move everything to the specified device
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    
    if init_model is None:
        if model_name == 'lstm':
            vocab_size = out_dim
            model = Models[model_name](vocab_size=vocab_size, out_dim=out_dim).to(device)
        else:
            model = Models[model_name](X.shape, out_dim=out_dim).to(device)
            if model_name == 'cnn':
                xavier_init_model(model)
            elif model_name == 'wideresnet':
                init_wideresnet(model)
            else:
                xavier_init_model(model)
    else:
        model = copy.deepcopy(init_model).to(device)

    if model_name == "lstm":
        if not _OPACUS_AVAILABLE:
            raise RuntimeError("LSTM training requires opacus: pip install opacus")
        if not isinstance(model, _GradSampleModule):
            model = _GradSampleModule(model)

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Set DP noise
    if epsilon is not None:
        sample_rate = batch_size / len(X)
        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=n_epochs,
        )
        if rank == 0:
            print(f"DP config: eps={epsilon}, delta={delta}, sample_rate={sample_rate:.6f}, epochs={n_epochs}, noise_multiplier={noise_multiplier}")
    else:
        noise_multiplier = 0

    block_size = min(block_size, batch_size)

    if len(X.shape) > 2:
        aug_fn = AugmentationFunction(X.shape[2], X.shape[1])
    else:
        aug_fn = None

    # Create dataset
    dataset = IndexedTensorDataset(X, y)
    scores = np.zeros(len(dataset), dtype=np.float32)
    # 0 = active, 1 = apply gradient ascent, 2 = inactive (dropped)
    # Must be integer (NOT bool) because we rely on the 3-state semantics downstream.
    drop_mask = np.zeros(len(dataset), dtype=np.int8)

    # Create global index to gradient mapping for gradient space canary (single canary at last index)
    global_idx_to_grad = None
    if gradient_space_audit and crafted_gradient is not None:
        canary_idx = len(dataset) - 1
        global_idx_to_grad = {canary_idx: crafted_gradient}

    # Set up batch iteration based on sampling strategy.
    # 'poisson': each sample independently included with prob q = batch_size / n (standard DP analysis).
    # 'shuffle': standard random-sampler DataLoader (every sample once per epoch).
    n_samples = len(dataset)
    if sampling == 'poisson':
        _poisson_q = batch_size / n_samples
        _poisson_n_batches = (n_samples + batch_size - 1) // batch_size
        loader = None  # not used for Poisson
    else:
        _poisson_q = None
        _poisson_n_batches = None
        sampler = torch.utils.data.RandomSampler(
            dataset, replacement=False, num_samples=None, generator=generator
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=int(num_workers),
            persistent_workers=bool(persistent_workers) if int(num_workers) > 0 else False,
            drop_last=False,
            generator=dl_generator,
        )

    prev_params = None
    prev_delta_theta = None
    theta0_params = None
    prev_losses = None
    loss_hist = None
    loss_hist_pos = None
    grad_norm_hist = None
    grad_norm_hist_pos = None
    grad_dir_hist = None
    grad_dir_hist_pos = None
    grad_dir_proj = None
    rand_proj_mat = None
    maxmin_proj_mat = None
    grad_accel_hist = None
    grad_accel_hist_pos = None
    grad_accel_proj = None
    grad_jerk_hist = None
    grad_jerk_hist_pos = None
    grad_jerk_proj = None
    dir_unique_hist = None
    dir_unique_hist_pos = None
    alignment_proj_mat = None
    canary_dropped_epoch = None
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        optimizer.zero_grad()
        print(f"Epoch: {epoch} (Active samples: {int((drop_mask == 0).sum())}/{len(drop_mask)})", end='', flush=True)

        # Build a unified batch iterator regardless of sampling strategy.
        if sampling == 'poisson':
            def _poisson_iter():
                for _ in range(_poisson_n_batches):
                    mask = torch.rand(n_samples, generator=generator) < _poisson_q
                    idx = torch.where(mask)[0]
                    if len(idx) == 0:
                        continue
                    yield X[idx], y[idx], idx
            batch_iter = enumerate(_poisson_iter())
        else:
            batch_iter = enumerate(loader)

        for batch_idx, (curr_X, curr_y, global_indices) in batch_iter:
            curr_X, curr_y = curr_X.to(device, non_blocking=True), curr_y.to(device, non_blocking=True)
            global_indices = global_indices.to(device, non_blocking=True)

            if defense_score_fn == 'loss_momentum' and prev_losses is None:
                prev_losses = np.full((len(dataset),), np.nan, dtype=np.float32)

            if defense_score_fn == 'loss_volatility' and loss_hist is None:
                k = int(loss_volatility_k)
                if k <= 0:
                    raise ValueError(f"loss_volatility_k must be > 0, got {k}")
                loss_hist = np.full((len(dataset), k), np.nan, dtype=np.float32)
                loss_hist_pos = np.zeros((len(dataset),), dtype=np.int64)

            if defense_score_fn == 'grad_norm_percentile' and grad_norm_hist is None:
                k = int(grad_norm_percentile_k)
                if k <= 0:
                    raise ValueError(f"grad_norm_percentile_k must be > 0, got {k}")
                grad_norm_hist = np.full((len(dataset), k), np.nan, dtype=np.float32)
                grad_norm_hist_pos = np.zeros((len(dataset),), dtype=np.int64)

            if defense_score_fn in ('grad_dir_volatility', 'grad_dir_volatility_unclipped') and grad_dir_hist is None:
                k = int(grad_dir_volatility_k)
                if k <= 0:
                    raise ValueError(f"grad_dir_volatility_k must be > 0, got {k}")

                # Note: grad_dir_proj will be created lazily on first batch when we know the actual gradient dimensions
                grad_dir_hist = np.full((len(dataset), k, int(grad_dir_proj_dim)), np.nan, dtype=np.float32)
                grad_dir_hist_pos = np.zeros((len(dataset),), dtype=np.int64)

            if defense_score_fn in ('norm_x_dir_uniqueness', 'norm_x_dir_uniqueness_unclipped') and dir_unique_hist is None:
                k = int(dir_unique_k)
                if k <= 0:
                    raise ValueError(f"dir_unique_k must be > 0, got {k}")

                # Note: grad_dir_proj will be created lazily on first batch
                dir_unique_hist = np.full((len(dataset), k, int(grad_dir_proj_dim)), np.nan, dtype=np.float32)
                dir_unique_hist_pos = np.zeros((len(dataset),), dtype=np.int64)

            if defense_score_fn == 'rand_proj_var' and rand_proj_mat is None:
                pass  # Will be created in dpsgd.py

            if defense_score_fn == 'maxmin_proj_ratio' and maxmin_proj_mat is None:
                pass  # Will be created in dpsgd.py

            if defense_score_fn == 'alignment_with_rand_proj' and alignment_proj_mat is None:
                pass  # Will be created in dpsgd.py

            if defense_score_fn in ('grad_accel', 'grad_accel_unclipped') and grad_accel_hist is None:
                # Keep a 3-step history for discrete second difference.
                # Note: grad_accel_proj will be created lazily on first batch
                grad_accel_hist = np.full((len(dataset), 3, int(grad_accel_proj_dim)), np.nan, dtype=np.float32)
                grad_accel_hist_pos = np.zeros((len(dataset),), dtype=np.int64)

            if defense_score_fn in ('grad_jerk', 'grad_jerk_unclipped') and grad_jerk_hist is None:
                # Keep a 4-step history for discrete third difference.
                # Note: grad_jerk_proj will be created lazily on first batch
                grad_jerk_hist = np.full((len(dataset), 4, int(grad_jerk_proj_dim)), np.nan, dtype=np.float32)
                grad_jerk_hist_pos = np.zeros((len(dataset),), dtype=np.int64)

            if defense_score_fn == 'cos_update' and prev_params is None:
                prev_params = {n: p.detach().clone() for n, p in model.named_parameters()}

            if defense_score_fn == 'cos_theta0' and theta0_params is None:
                theta0_params = {n: p.detach().clone() for n, p in model.named_parameters()}

            if defense_score_fn == 'norm_x_trajectory_orth' and theta0_params is None:
                theta0_params = {n: p.detach().clone() for n, p in model.named_parameters()}

            curr_params = {n: p.detach() for n, p in model.named_parameters()}
            if prev_params is not None:
                prev_delta_theta = {n: curr_params[n] - prev_params[n] for n in prev_params.keys()}
            else:
                prev_delta_theta = None

            if theta0_params is not None:
                theta_t_minus_theta0 = {n: curr_params[n] - theta0_params[n] for n in theta0_params.keys()}
            else:
                theta_t_minus_theta0 = None

            defense_cfg = DefenseConfig(
                score_fn=defense_score_fn,
                score_norm=defense_score_norm,
                delta_theta=prev_delta_theta,
                theta_t_minus_theta0=theta_t_minus_theta0,
                grad_norm_hist=grad_norm_hist,
                grad_norm_hist_pos=grad_norm_hist_pos,
                grad_norm_percentile_k=int(grad_norm_percentile_k),
                grad_dir_hist=grad_dir_hist,
                grad_dir_hist_pos=grad_dir_hist_pos,
                grad_dir_volatility_k=int(grad_dir_volatility_k),
                grad_dir_proj=grad_dir_proj,
                rand_proj_mat=rand_proj_mat,
                rand_proj_var_m=int(rand_proj_var_m),
                maxmin_proj_mat=maxmin_proj_mat,
                maxmin_proj_k=int(maxmin_proj_k),
                grad_rank_mode=str(grad_rank_mode),
                grad_rank_eps=float(grad_rank_eps),
                grad_accel_hist=grad_accel_hist,
                grad_accel_hist_pos=grad_accel_hist_pos,
                grad_accel_proj=grad_accel_proj,
                grad_jerk_hist=grad_jerk_hist,
                grad_jerk_hist_pos=grad_jerk_hist_pos,
                alignment_proj_mat=alignment_proj_mat,
                alignment_proj_k=int(alignment_proj_k),
                grad_jerk_proj=grad_jerk_proj,
                dir_unique_hist=dir_unique_hist,
                dir_unique_hist_pos=dir_unique_hist_pos,
                dir_unique_k=int(dir_unique_k),
                grad_scatter_k=int(grad_scatter_k),
                prev_losses=prev_losses,
                loss_hist=loss_hist,
                loss_hist_pos=loss_hist_pos,
                loss_volatility_k=int(loss_volatility_k),
            )
            
            # Clip & accumulate gradients (no world_size/rank needed)
            curr_accumulated_gradients, scores = clip_and_accum_grads(
                model,
                curr_X, curr_y, optimizer, criterion,
                max_grad_norm, 
                drop_mask=drop_mask[global_indices.cpu().numpy()] if drop_mask is not None else None,
                block_size=block_size,
                scores=scores,
                device=device,
                global_indices=global_indices,
                aug_mult=aug_mult, 
                aug_fn=aug_fn,
                world_size=1,  # Single GPU
                rank=0,        # Single GPU
                batch_size=batch_size,
                is_gradient_space_canary=gradient_space_audit,
                global_idx_to_grad=global_idx_to_grad,
                canary_indices=np.array([len(dataset) - 1]) if gradient_space_audit else None,
                defense_cfg=defense_cfg,
                defense_apply_ascent=defense_apply_ascent
            )
            
            # Update projection matrices if they were lazily created in dpsgd.py
            if defense_cfg.grad_dir_proj is not None:
                grad_dir_proj = defense_cfg.grad_dir_proj
            if defense_cfg.rand_proj_mat is not None:
                rand_proj_mat = defense_cfg.rand_proj_mat
            if defense_cfg.maxmin_proj_mat is not None:
                maxmin_proj_mat = defense_cfg.maxmin_proj_mat
            if defense_cfg.alignment_proj_mat is not None:
                alignment_proj_mat = defense_cfg.alignment_proj_mat
            if defense_cfg.grad_accel_proj is not None:
                grad_accel_proj = defense_cfg.grad_accel_proj
            if defense_cfg.grad_jerk_proj is not None:
                grad_jerk_proj = defense_cfg.grad_jerk_proj
            
            processed = global_indices.cpu().numpy()
            drop_mask[processed[drop_mask[processed] == 1]] = 2

            # Apply the accumulated gradients
            with torch.no_grad():
                for name, param in model.named_parameters():
                    
                    if name not in curr_accumulated_gradients:
                        print(f"Warning: Parameter {name} not found in accumulated gradients")
                        continue
                        
                    grad = curr_accumulated_gradients[name].to(device)
                    
                    # Add DP noise to the sum of clipped gradients (before averaging)
                    if noise_multiplier > 0 and max_grad_norm is not None:
                        noise_std = noise_multiplier * max_grad_norm
                        noise = noise_std * torch.randn_like(grad)
                        grad.add_(noise)
                    
                    # Average the noisy gradient sum by the nominal batch size so that the
                    # effective LR and noise scale are consistent across Poisson draws.
                    grad.div_(float(batch_size))
                    
                    if param.grad is None:
                        param.grad = grad.clone()
                    else:
                        param.grad.copy_(grad)
            
            optimizer.step()
            optimizer.zero_grad()

            if defense_score_fn == 'cos_update' and prev_params is not None:
                curr_params = {n: p.detach() for n, p in model.named_parameters()}
                prev_delta_theta = {n: curr_params[n] - prev_params[n] for n in prev_params.keys()}
                prev_params = {n: curr_params[n].clone() for n in prev_params.keys()}
        
        epoch_time = time.time() - epoch_start
        print(f" | Time: {epoch_time:.2f}s")
        
        # Defense operations - only apply filtering every defense_filter_every epochs
        if defense and (epoch % defense_filter_every == 0):
            k = int(defense_k)
            unique_classes = torch.unique(y).cpu()
            active_mask = torch.from_numpy(drop_mask == 0)

            canary_global = len(dataset) - 1

            for cls in unique_classes:
                cls_indices = ((y.cpu() == cls.item()) & active_mask).nonzero(as_tuple=True)[0]
                if len(cls_indices) == 0:
                    continue

                cls_scores = torch.tensor(scores[cls_indices.cpu().numpy()], device=y.device)
                _, topk_indices = torch.topk(cls_scores, min(k, len(cls_scores)))

                topk_global_indices = cls_indices[topk_indices]

                dropped_indices = topk_global_indices.cpu().numpy()
                drop_mask[dropped_indices] = 1

                if X.shape[0] - 1 in dropped_indices and canary_dropped_epoch is None:
                    canary_score = float(scores[X.shape[0] - 1])
                    print(f"\n[INFO] Canary (index {X.shape[0]-1}) was dropped from the training set! Score: {canary_score:.6f}")
                    canary_dropped_epoch = int(epoch)

            scores.fill(0)

    if return_defense_state:
        return model, drop_mask, canary_dropped_epoch
    return model


def distribute_reps(n_reps, world_size):
    """Distribute model training repetitions across GPUs."""
    reps_per_gpu = [[] for _ in range(world_size)]
    for i in range(n_reps):
        reps_per_gpu[i % world_size].append(i)
    return reps_per_gpu


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

    out_folder = f'{args.out}/{args.data_name}_{args.model_name}_eps{args.epsilon}'
    os.makedirs(out_folder, exist_ok=True)

    if rank == 0:
        print('Loading data')
    if args.n_df == 1:
        X_out, y_out, out_dim = load_data(args.data_name, 1)
    else:
        X_out, y_out, out_dim = load_data(args.data_name, args.n_df - 1)

    # Initialize model with SAME seed across all GPUs for fixed_init
    if rank == 0:
        print('Initializing model')
    init_model = None
    if args.fixed_init is not None:
        # Use same seed for all GPUs to ensure identical initialization
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        init_model = Models[args.model_name](X_out.shape, out_dim=out_dim)
        if args.fixed_init == '':
            if args.model_name == 'cnn':
                xavier_init_model(init_model)
            elif args.model_name == 'wideresnet':
                init_wideresnet(init_model)
            else:
                xavier_init_model(init_model)
        else:
            init_model.load_state_dict(torch.load(args.fixed_init))
            X_out, y_out = X_out[len(X_out) // 2:], y_out[len(y_out) // 2:]
    
    # NOW set per-rank seeds for everything else (after init_model is created)
    # This ensures data loading and other operations are still independent per GPU
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    # Initialize gradient_space_canary_target_class early (will be set later if gradient space canary is loaded)
    gradient_space_canary_target_class = None

    # Craft target
    if rank == 0:
        print('Crafting target data point')

    if args.canary_pt is not None:
        if not os.path.exists(args.canary_pt):
            raise FileNotFoundError(f"--canary_pt not found: {args.canary_pt}")
        payload = torch.load(args.canary_pt, map_location='cpu')

        if isinstance(payload, dict):
            if 'canary' not in payload:
                raise KeyError(f"Canary .pt dict must contain key 'canary'. Found keys: {list(payload.keys())}")
            target_X = payload['canary']
            if 'target_label' in payload:
                target_y_val = payload['target_label']
            elif 'canary_label' in payload:
                target_y_val = payload['canary_label']
            elif 'label' in payload:
                target_y_val = payload['label']
            elif 'true_label' in payload:
                target_y_val = payload['true_label']
            elif 'audit_label' in payload:
                target_y_val = payload['audit_label']
            else:
                target_y_val = 9

        elif torch.is_tensor(payload):
            target_X = payload
            target_y_val = 9
        else:
            raise TypeError(f"Unsupported canary_pt payload type: {type(payload)}")

        if torch.is_tensor(target_y_val):
            target_y = target_y_val.clone().detach().long().view(-1)
        else:
            target_y = torch.tensor([int(target_y_val)], dtype=torch.long)

        if not torch.is_tensor(target_X):
            target_X = torch.tensor(target_X)

        target_X = target_X.clone().detach()
        if target_X.ndim == X_out.ndim - 1:
            target_X = target_X.unsqueeze(0)
        if target_X.ndim != X_out.ndim:
            raise ValueError(f"Loaded canary has shape {tuple(target_X.shape)} but expected {tuple(X_out[[0]].shape)}")

        if rank == 0:
            print(f"Loaded canary from {args.canary_pt}: X={tuple(target_X.shape)}, y={target_y.tolist()}")
    else:
        # check for data_names + target_types that don't match
        if args.data_name == 'mnist':
            pass # compatible with all canaries

        elif args.data_name == 'cifar10':
            pass # compatible with all canaries
        elif args.data_name == 'cifar100':
            pass # compatible with all canaries
        elif args.data_name == 'purchase':
            pass # compatible with all canaries
        elif args.data_name == 'tiny_shakespeare':
            if args.target_type != 'empty_sequence':
                raise Exception("For tiny_shakespeare, only target_type='empty_sequence' is supported.")
        elif args.target_type == 'empty_sequence':
            raise Exception("Target type 'empty_sequence' is only valid with data_name='tiny_shakespeare'.")

        # Craft target
        if args.target_type == 'gradient_space_canary':
            target_X = X_out[-1].unsqueeze(0)
            # Use the target class from the canary file if available, otherwise use the last sample's label
            if gradient_space_canary_target_class is not None:
                target_y = torch.tensor([gradient_space_canary_target_class], dtype=torch.long)
            else:
                target_y = y_out[-1].unsqueeze(0)
            if rank == 0:
                print("Using gradient-space canary")
        elif args.target_type == 'mislabeled':
            # Find first sample with true label 0
            class_0_indices = (y_out == 0).nonzero(as_tuple=True)[0]
            if len(class_0_indices) == 0:
                raise ValueError("No class 0 samples found in dataset for mislabeled canary")
            # Use first class 0 sample deterministically
            target_idx = class_0_indices[0].item()
            target_X = X_out[target_idx].unsqueeze(0)
            # Mislabel it as the specified target class
            target_y = torch.tensor([args.mislabeled_target_class], dtype=torch.long)
            if rank == 0:
                print(f"Using mislabeled canary: true class 0 sample (index {target_idx}) relabeled as class {args.mislabeled_target_class}")
        elif args.target_type == 'blank':
            blank_img = torch.zeros_like(X_out[[0]])
            if args.blank_alpha > 0:
                label_9_indices = (y_out == 9).nonzero(as_tuple=True)[0]
                if len(label_9_indices) == 0:
                    raise ValueError("No label 9 samples found in dataset")
                label_9_img = X_out[label_9_indices[0]].unsqueeze(0)
                target_X = args.blank_alpha * label_9_img + (1 - args.blank_alpha) * blank_img
            else:
                target_X = blank_img
            target_y = torch.from_numpy(np.array([9]))
        elif args.target_type == 'badnets':
            target_X = X_out[-1]
            target_y = torch.tensor(args.badnets_label)
            target_X[:, -4:, -4:] = torch.max(target_X)
            target_X = target_X.unsqueeze(0)
            target_y = target_y.unsqueeze(0)
        elif args.target_type == 'sanity_check':
            target_X = X_out[-1].unsqueeze(0)
            target_y = y_out[-1].unsqueeze(0)
        elif args.target_type == 'clipbkd':
            target_X, target_y = craft_clipbkd(X_out, init_model)
        elif args.target_type == 'fgsm':
            print("Preparing FGSM attack by training a model on the available data...")
            
            # Create a new model for FGSM
            fgsm_model = Models[args.model_name](X_out.shape, out_dim=out_dim).to(device)
            if args.model_name == 'cnn':
                xavier_init_model(fgsm_model)
            else:
                init_wideresnet(fgsm_model)
            
            # Train the model using the existing train_model function
            print("Training FGSM model...")
            
            # Use train_model with DP disabled (delta=0, max_grad_norm=inf)
            fgsm_model = train_model(
                model_name=args.model_name,
                X=X_out,
                y=y_out,
                X_target=None,
                y_target=None,
                epsilon=None,  # No DP
                delta=None,    # No DP
                max_grad_norm=None,  # No gradient clipping
                n_epochs=args.n_epochs,
                lr=args.lr,
                block_size=args.block_size,
                batch_size=args.batch_size,
                init_model=fgsm_model,
                out_dim=out_dim,
                aug_mult=1,  # No augmentation for FGSM
                rank=rank,
                world_size=world_size,
                gradient_space_audit=False,
                defense=False,
                defense_k=int(args.defense_k)
            )
            print("FGSM model training completed")
            
            # Get the last sample and its true label
            original_X = X_out[-1].unsqueeze(0).to(device)
            original_y = y_out[-1].unsqueeze(0).to(device)
            
            # Choose a target class different from the original
            num_classes = out_dim
            target_class = (original_y + 1) % num_classes  # Simple way to pick a different class
            
            print(f"Performing FGSM attack on sample (original class: {original_y.item()}, target class: {target_class.item()})")
            
            # Perform iterative FGSM attack
            print("Running iterative FGSM attack...")
            target_X, iters_used, success = fgsm_attack(
                fgsm_model, 
                original_X, 
                target_class, 
                epsilon=0.1,  # Maximum perturbation
                max_iter=20,  # Maximum iterations
                alpha=0.01    # Step size
            )
            target_y = target_class
            
            # Validate attack results
            with torch.no_grad():
                fgsm_model.eval()
                output = fgsm_model(target_X)
                pred = output.argmax(dim=1)
                perturbation = (target_X - original_X).abs().max().item()
                
                if success:
                    print(f"FGSM attack succeeded in {iters_used} iterations")
                else:
                    print(f"FGSM attack failed after {iters_used} iterations (using best adversarial example found)")
                
                print(f"  Predicted class: {pred.item()}, Target class: {target_class.item()}")
                print(f"  Perturbation L∞ norm: {perturbation:.6f} (epsilon={0.1})")
                print(f"  Attack actually fooled model: {pred.item() == target_class.item()}")
            
            # Move back to CPU if needed
            if not target_X.is_cpu:
                target_X = target_X.cpu()
            if not target_y.is_cpu:
                target_y = target_y.cpu()
                
            # Clean up
            del fgsm_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print("FGSM attack completed")
        elif args.target_type == 'empty_sequence':
            # sequence length (same as existing chunks)
            seq_len = X_out.shape[1]
            target_X = torch.zeros((1, seq_len), dtype=torch.long)
            target_y = torch.full((1, seq_len), 9, dtype=torch.long)
        elif os.path.exists(args.target_type):
            # pre-crafted target sample
            target_X = torch.from_numpy(np.load(args.target_type))
            if init_model is not None:
                target_y =  choose_worstcase_label(init_model, target_X)
            else:
                target_y = torch.from_numpy(np.array([9]))
        else:
            raise Exception(f'Target {args.target_type} not found')

    # Define datasets
    X_in, y_in = torch.vstack((X_out[:-1], target_X)), torch.cat((y_out[:-1], target_y))
    X_test, y_test, _ = load_data(args.data_name, None, split='test')

    if rank == 0:
        print('Training models')
    
    # Initialize run state (no resume support)
    worlds = [args.fit_world_only] if args.fit_world_only else ['in', 'out']
    
    outputs, losses, all_losses, train_set_accs, test_set_accs = init_run_state(
        out_folder, args.fit_world_only, rank)
    
    # Create or load crafted gradient if needed
    crafted_grad = None
    if args.target_type == 'gradient_space_canary':
        if args.gradient_space_canary_pt is not None:
            if not os.path.exists(args.gradient_space_canary_pt):
                raise FileNotFoundError(f"--gradient_space_canary_pt not found: {args.gradient_space_canary_pt}")
            payload = torch.load(args.gradient_space_canary_pt, map_location='cpu')
            if isinstance(payload, dict) and 'gradient' in payload:
                crafted_grad = payload['gradient']
                # Extract target class if available
                if 'target_class' in payload:
                    gradient_space_canary_target_class = payload['target_class']
            else:
                # Backward compatibility: direct gradient dictionary
                crafted_grad = payload
            
            # Move crafted_grad to the correct device
            if crafted_grad is not None:
                crafted_grad = {name: g.to(device) for name, g in crafted_grad.items()}
            
            if rank == 0:
                print(f"Loaded gradient space canary from {args.gradient_space_canary_pt}")
                if gradient_space_canary_target_class is not None:
                    print(f"  Target class: {gradient_space_canary_target_class}")
        elif args.canary_pt is None:
            if rank == 0:
                print('Creating crafted gradient')
            temp_model = Models[args.model_name](X_out.shape, out_dim=out_dim).to(device)
            if args.model_name == 'cnn':
                xavier_init_model(temp_model)
            else:
                init_wideresnet(temp_model)
            crafted_grad = craft_gradient(model=temp_model, device=device)
            del temp_model

    # Distribute repetitions across GPUs
    # TODO: fix distribute reps to account for reps completed
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
            
            model = train_model(
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
                out_dim=out_dim, 
                defense=args.defense,
                defense_k=int(args.defense_k),
                defense_filter_every=int(args.defense_filter_every),
                aug_mult=args.aug_mult,
                gradient_space_audit=(args.target_type == 'gradient_space_canary' and world == 'in'),
                crafted_gradient=crafted_grad if (args.target_type == 'gradient_space_canary' and world == 'in') else None,
                device=device,
                generator=generator,
                dl_generator=dl_generator,
                rank=rank,
                defense_score_norm=args.defense_score_norm,
                defense_score_fn=args.defense_score_fn,
                grad_norm_percentile_k=args.grad_norm_percentile_k,
                grad_dir_volatility_k=args.grad_dir_volatility_k,
                grad_dir_proj_dim=args.grad_dir_proj_dim,
                grad_dir_proj_seed=args.grad_dir_proj_seed,
                dir_unique_k=args.dir_unique_k,
                rand_proj_var_m=args.rand_proj_var_m,
                rand_proj_var_seed=args.rand_proj_var_seed,
                maxmin_proj_k=args.maxmin_proj_k,
                maxmin_proj_seed=args.maxmin_proj_seed,
                grad_rank_mode=args.grad_rank_mode,
                grad_rank_eps=args.grad_rank_eps,
                grad_accel_proj_dim=args.grad_accel_proj_dim,
                grad_accel_proj_seed=args.grad_accel_proj_seed,
                grad_jerk_proj_dim=args.grad_jerk_proj_dim,
                grad_jerk_proj_seed=args.grad_jerk_proj_seed,
                alignment_proj_k=args.alignment_proj_k,
                alignment_proj_seed=args.alignment_proj_seed,
                grad_scatter_k=args.grad_scatter_k,
                defense_apply_ascent=args.defense_apply_ascent,
                sampling=args.sampling,
            )
            
            # Compute outputs and losses
            model.eval()
            with torch.no_grad():
                target_X_device = target_X.to(device)
                target_y_device = target_y.to(device)
                
                output = model(target_X_device)
                
                if args.target_type == 'gradient_space_canary' and crafted_grad is not None:
                    # For gradient space canary, score by L∞ norm of parameter update
                    final_params = {n: p.detach().clone().to(device) for n, p in model.named_parameters()}
                    init_params = {n: p.detach().clone().to(device) for n, p in init_model.named_parameters()}

                    update = {n: final_params[n] - init_params[n] for n in final_params}
                    flat_update = torch.cat([p.view(-1) for p in update.values()])

                    # Score by L∞ norm of parameter update
                    loss = flat_update.abs().max().item()
                else:
                    loss = -nn.CrossEntropyLoss()(output, target_y_device).cpu().item()
                
                # Store locally - no gathering needed
                outputs[world].append(output[0].cpu().numpy())
                losses[world].append(loss)

            if args.store_all_losses:
                if args.target_type == 'gradient_space_canary':
                    # Not a per-sample loss-based audit; keep placeholder for alignment.
                    all_losses[world].append(np.array([], dtype=np.float32))
                else:
                    all_losses[world].append(
                        compute_per_sample_losses(model, curr_X, curr_y, device=device)
                    )
            
            # Each rank saves its own checkpoint
            # TODO: fix save_checkpoint to account for randomness
            save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, args.fit_world_only, rank)
            
            # Get test set accuracy from first 5 reps
            if rep < 5 and world == 'in':
                if len(X_out) > 0:
                    train_set_accs.append(test_model(model, X_in, y_in))
                    print(f'[Rank {rank}] Train set acc:', train_set_accs[-1])
                test_set_accs.append(test_model(model, X_test, y_test))
                print(f'[Rank {rank}] Test set acc:', test_set_accs[-1])
                save_checkpoint(out_folder, outputs, losses, all_losses, train_set_accs, test_set_accs, args.fit_world_only, rank)
        
        # After all reps in this world
        outputs[world] = np.array(outputs[world])

    # Synchronize only in distributed mode
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.barrier()

    # Final audit - only rank 0 needs to combine results from all ranks
    if rank == 0:
        print("\n[Rank 0] Combining results from all ranks...")
        
        # Load results from all rank files
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
                    if os.path.exists(f'{out_folder}/train_set_accs{suffix}.npy'):
                        combined_train_accs.extend(np.load(f'{out_folder}/train_set_accs{suffix}.npy'))
                    if os.path.exists(f'{out_folder}/test_set_accs{suffix}.npy'):
                        combined_test_accs.extend(np.load(f'{out_folder}/test_set_accs{suffix}.npy'))
                else:
                    combined_outputs[args.fit_world_only].extend(np.load(f'{out_folder}/outputs_{args.fit_world_only}{suffix}.npy'))
                    combined_losses[args.fit_world_only].extend(np.load(f'{out_folder}/losses_{args.fit_world_only}{suffix}.npy'))
                    if args.store_all_losses and os.path.exists(f'{out_folder}/all_losses_{args.fit_world_only}{suffix}.npy'):
                        combined_all_losses[args.fit_world_only].extend(
                            np.load(f'{out_folder}/all_losses_{args.fit_world_only}{suffix}.npy', allow_pickle=True)
                        )
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
            if combined_train_accs:
                np.save(f'{out_folder}/train_set_accs.npy', combined_train_accs)
            if combined_test_accs:
                np.save(f'{out_folder}/test_set_accs.npy', combined_test_accs)
        else:
            np.save(f'{out_folder}/outputs_{args.fit_world_only}.npy', combined_outputs[args.fit_world_only])
            np.save(f'{out_folder}/losses_{args.fit_world_only}.npy', combined_losses[args.fit_world_only])
            if args.store_all_losses:
                np.save(
                    f'{out_folder}/all_losses_{args.fit_world_only}.npy',
                    np.array(combined_all_losses[args.fit_world_only], dtype=object)
                )
        
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

                max_t, emp_eps_loss = compute_eps_lower_from_mia(mia_scores, mia_labels, args.alpha, args.delta, 'GDP', n_procs=1)

                if args.holdout_audit:
                    emp_eps_loss = compute_eps_lower_from_mia_given_t(np.concatenate(
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

        if combined_train_accs:
            print(f'Train set accuracy: {np.mean(combined_train_accs) * 100:.3f}%')
        if combined_test_accs:
            print(f'Test set accuracy: {np.mean(combined_test_accs) * 100:.3f}%')
    
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