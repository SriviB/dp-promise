"""
Shared argparse builder for DP-SGD audit entry-points.

Centralises the ~50 arguments that were previously copy-pasted into every
audit script. Entry-points call `build_parser()` and may add their own
script-specific arguments on top.
"""
import argparse


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
    parser.add_argument('--config', type=str, required=True,
                        help='DP-PROMISE config (dataset, model and training settings)')
    parser.add_argument('--dataset_size', type=int, default=0,
                        help='Dataset size |D| (0 = full dataset)')

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    parser.add_argument('--n_reps', type=int, default=200,
                        help='Number of shadow models to train')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Training epochs per model')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Training batch size')
    parser.add_argument('--block_size', type=int,
                        help='Process batch in sub-blocks to save GPU memory')
    parser.add_argument('--sampling', type=str, default='poisson',
                        choices=['poisson', 'shuffle'],
                        help='Minibatch sampling strategy. '
                             '"poisson": each sample included independently with prob q (default, '
                             'matches DP-SGD privacy analysis). '
                             '"shuffle": standard random-sampler DataLoader.')

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
    # DP-PROMISE trains unconditionally, so canaries are defined purely in image
    # (or gradient) space -- there is no label to mislabel or target.
    parser.add_argument('--target_type', type=str, default='blank',
                        choices=['blank', 'blank_interp', 'in_distribution', 'gradient'],
                        help='Canary type. '
                             '"blank": all-black image (-1 under DP-PROMISE scaling). '
                             '"blank_interp": blank interpolated with a random dataset image '
                             '(see --interp_alpha). '
                             '"in_distribution": an unmodified image from the dataset. '
                             '"gradient": gradient-space canary loaded from --canary_pt.')
    parser.add_argument('--interp_alpha', type=float, default=0.5,
                        help='Interpolation weight for --target_type blank_interp: '
                             '0 = all black, 1 = the sampled dataset image')
    parser.add_argument('--canary_pt', type=str, default=None,
                        help='Path to a .pt gradient canary (required for --target_type gradient). '
                             'Either a {name: tensor} gradient dict or {"gradient": {...}}')

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

    return parser
