"""
Canary and adversarial example generation for DP-SGD auditing.

Consolidates canary crafting logic that was previously scattered across
audit entry-points and utils/clipbkd.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Gradient-space canary
# ---------------------------------------------------------------------------

def craft_gradient(model: nn.Module, hot_index: int = None, device: str = 'cuda'):
    """
    Craft a one-hot gradient vector spanning all trainable parameters.

    Returns a dict {param_name: grad_tensor} where the gradient is zero
    everywhere except at position `hot_index` in the flattened parameter
    space (default: middle element).
    """
    params = {}
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            params[name] = {'param': param, 'start': total, 'shape': param.shape}
            total += param.numel()

    if hot_index is None:
        hot_index = total // 2 if total > 0 else 0

    if hot_index < 0 or (total > 0 and hot_index >= total):
        raise ValueError(f"hot_index {hot_index} out of range for model with {total} parameters")

    crafted = {}
    for name, info in params.items():
        end = info['start'] + info['param'].numel()
        g = torch.zeros_like(info['param'])
        if info['start'] <= hot_index < end:
            local = hot_index - info['start']
            flat = g.view(-1)
            flat[local] = 10_000_000
            g = flat.view(info['shape'])
        crafted[name] = g.unsqueeze(0)

    flat_norm = torch.cat([g.view(-1) for g in crafted.values()]).norm()
    print(f"Crafted gradient norm: {flat_norm:.4f}")
    return crafted


# ---------------------------------------------------------------------------
# FGSM / PGD canary
# ---------------------------------------------------------------------------

def fgsm_attack(
    model: nn.Module,
    X: torch.Tensor,
    y,
    epsilon: float = 0.1,
    max_iter: int = 10,
    alpha: float = 0.01,
):
    """
    Iterative FGSM (PGD) targeted attack.

    Minimises cross-entropy for target class `y`, returning an adversarial
    example within the L∞ ball of radius `epsilon` around X.

    Args:
        model:    model to attack (set to eval mode internally)
        X:        input tensor, shape (1, ...)
        y:        target class label (tensor or int)
        epsilon:  L∞ perturbation bound
        max_iter: maximum PGD iterations
        alpha:    step size per iteration

    Returns:
        (X_adv, iters_used, success)
    """
    assert epsilon > 0
    assert 0 < alpha <= epsilon
    assert max_iter > 0

    model.eval()
    X_adv = X.clone().detach().requires_grad_(True)
    best_adv = X_adv.detach().clone()
    best_conf = -float('inf')

    y_idx = y.item() if isinstance(y, torch.Tensor) else int(y)

    for i in range(max_iter):
        output = model(X_adv)
        pred_idx = output.argmax(dim=1).item()

        if pred_idx == y_idx:
            return X_adv.detach(), i + 1, True

        conf = F.softmax(output, dim=1)[0, y_idx].item()
        if conf > best_conf:
            best_conf = conf
            best_adv = X_adv.detach().clone()

        loss = F.cross_entropy(output, torch.tensor([y_idx], device=X_adv.device))
        model.zero_grad()
        loss.backward()

        X_adv = X_adv.detach() - alpha * X_adv.grad.data.sign()
        delta = torch.clamp(X_adv - X, -epsilon, epsilon)
        X_adv = torch.clamp(X + delta, 0, 1).detach().requires_grad_(True)

    return best_adv, max_iter, False


# ---------------------------------------------------------------------------
# ClipBKD canary  (previously utils/clipbkd.py)
# ---------------------------------------------------------------------------

def choose_worstcase_label(model: nn.Module, target_X: torch.Tensor) -> torch.Tensor:
    """Return the class label that maximises the gradient norm (least confident)."""
    with torch.no_grad():
        output = model(target_X)
        return torch.unsqueeze(torch.argmin(output), dim=0)


def craft_clipbkd(X: torch.Tensor, model: nn.Module):
    """
    Craft a ClipBKD canary from the last PCA component of X.

    Reference:
        Jagielski et al. (2020) "Auditing Differentially Private Machine Learning",
        NeurIPS 2020. https://github.com/jagielski/auditing-dpsgd
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    flat_X = torch.flatten(X, start_dim=1).cpu().numpy()
    n_comps = min(flat_X.shape[0], flat_X.shape[1])
    pca = PCA(n_comps)
    pca.fit(flat_X)

    avg_norm = torch.mean(torch.norm(torch.flatten(X, start_dim=1), dim=1))
    target_X = avg_norm * torch.from_numpy(pca.components_[-1:]).to(device)
    target_X = torch.unsqueeze(target_X.reshape(X.shape[1:]), dim=0)

    model.to(device)
    target_y = choose_worstcase_label(model, target_X)
    return target_X.cpu(), target_y.cpu()
