"""
Shared training helpers used by the audit script.

IndexedTensorDataset is copied from bb-audit-dpsgd (utils/training.py).
test_model and compute_per_sample_losses keep bb-audit's structure but report
DP-PROMISE's Phase 1 denoising loss instead of classification accuracy / cross-entropy.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset


# ---------------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------------

class IndexedTensorDataset(Dataset):
    """TensorDataset that also returns the sample's global index."""
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == t.size(0) for t in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors) + (index,)

    def __len__(self):
        return self.tensors[0].size(0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_model(model, X: torch.Tensor, y: torch.Tensor, trainer, batch_size: int = 128) -> float:
    """Return the average Phase 1 denoising loss on (X, y), via DP-PROMISE's trainer."""
    device = next(model.parameters()).device

    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    model.eval()
    total_loss = total = 0
    with torch.no_grad():
        for curr_X, curr_y in loader:
            curr_X = curr_X.to(device)
            # Phase 1 trains with every label set to 0
            curr_y = torch.zeros_like(curr_y, dtype=torch.long).to(device)
            total_loss += trainer(model, curr_X, curr_y, phase="1").item() * len(curr_y)
            total += len(curr_y)
    model.train()
    return total_loss / total if total > 0 else 0.0


def compute_per_sample_losses(model, X: torch.Tensor, y: torch.Tensor, trainer, config, device,
                              batch_size: int = 256, num_eval_samples: int = 4, eval_seed: int = 0):
    """Return per-sample Phase 1 denoising losses as a numpy array.

    Same noising as DP-PROMISE's Phase 1 loss (timesteps [S, T), label 0), but one loss
    per image, averaged over num_eval_samples noised copies with a fixed seed.
    """
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    losses = []
    model.eval()
    eval_gen = torch.Generator(device='cpu').manual_seed(eval_seed)
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            # Phase 1 trains with every label set to 0
            batch_y = torch.zeros((batch_X.shape[0],), dtype=torch.long, device=device)
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
                pred_noise = model(x_noisy, t, batch_y)
                batch_mse += F.mse_loss(noise, pred_noise, reduction='none').mean(dim=[1, 2, 3])
            batch_mse /= num_eval_samples
            losses.append(batch_mse.cpu().numpy())
    model.train()
    return np.concatenate(losses)
