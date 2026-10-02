"""
Checkpoint management for distributed audit runs.

Each distributed rank writes its own file; rank 0 later aggregates them.
"""
import os
import numpy as np
import torch
import dill


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
