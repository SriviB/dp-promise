"""
Shared training helpers used by the audit script.

IndexedTensorDataset is copied from bb-audit-dpsgd (utils/training.py).
test_model keeps bb-audit's structure but reports DP-PROMISE's Phase 1
denoising loss instead of classification accuracy.
"""
import torch
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
