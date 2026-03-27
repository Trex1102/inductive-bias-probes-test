"""
Contrastive Representation Regularizer for Direction B: The Heuristic Trap.

Adds a contrastive loss term that encourages the model's hidden representations
to preserve pairwise distances from the input space, preventing collapse to
next-token equivalence classes.

Reference: Direction B in the NeurIPS roadmap.
L_total = L_NTP + lambda * L_contrast

The regularizer does NOT require access to the true state -- it only requires
that the model's representations preserve input structure.
"""

import torch
import torch.nn.functional as F


def create_contrastive_loss_fn(lambda_contrast, max_time_steps=20):
    """Create a contrastive representation regularizer.

    The loss encourages pairwise distances in representation space to correlate
    with pairwise distances in input space. Uses Pearson correlation for
    scale-invariance.

    Args:
        lambda_contrast: Weight of the contrastive loss term.
        max_time_steps: Max number of time steps to subsample for efficiency.

    Returns:
        A callable (reps, X, config) -> scalar loss tensor.
    """

    def contrastive_loss_fn(reps, X, config):
        """Compute contrastive representation loss.

        Args:
            reps: Hidden representations from the model, shape (B, T, D).
            X: Input batch, shape (B, T, input_dim).
            config: Training configuration dict.

        Returns:
            Scalar loss tensor.
        """
        B, T, D = reps.shape
        if B < 3:
            return torch.tensor(0.0, device=reps.device, requires_grad=True)

        # Subsample time steps for efficiency
        T_sub = min(T, max_time_steps)
        if T_sub < T:
            idx = torch.randperm(T, device=reps.device)[:T_sub]
            reps_sub = reps[:, idx]
            X_sub = X[:, idx]
        else:
            reps_sub = reps
            X_sub = X

        # Pool representations over subsampled time steps
        h = reps_sub.mean(dim=1)  # (B, D)

        # Compute input-space distances
        vocab_size = config.get("input_vocab_size")
        if vocab_size is not None:
            # Discrete inputs: use one-hot encoding for meaningful distances
            if X_sub.shape[-1] == 1:
                x_ids = X_sub.squeeze(-1).long()
            else:
                x_ids = X_sub.long()
            # Clamp to valid range before one-hot encoding
            x_ids = x_ids.clamp(0, vocab_size - 1)
            x_onehot = F.one_hot(x_ids, num_classes=vocab_size).float()
            x_flat = x_onehot.reshape(B, -1)
        else:
            # Continuous inputs
            x_flat = X_sub.float().reshape(B, -1)

        with torch.no_grad():
            input_dist = torch.cdist(x_flat, x_flat)  # (B, B)

        # Compute representation distances
        rep_dist = torch.cdist(h, h)  # (B, B)

        # Extract upper triangular elements (unique pairs)
        mask = torch.triu(
            torch.ones(B, B, device=reps.device, dtype=torch.bool), diagonal=1
        )
        input_d = input_dist[mask]
        rep_d = rep_dist[mask]

        # Correlation-based loss: maximize Pearson correlation between distances
        # This is scale-invariant (no normalization needed)
        input_d_c = input_d - input_d.mean()
        rep_d_c = rep_d - rep_d.mean()

        numerator = (input_d_c * rep_d_c).sum()
        denominator = (
            torch.sqrt((input_d_c**2).sum() * (rep_d_c**2).sum()) + 1e-8
        )
        correlation = numerator / denominator

        # Loss: want correlation close to 1
        loss = lambda_contrast * (1.0 - correlation)
        return loss

    return contrastive_loss_fn


def get_log_spaced_steps(max_iters):
    """Generate log-spaced checkpoint steps.

    Returns steps like {10, 20, 50, 100, 200, 500, 1K, 2K, 5K, 10K, ...}
    up to max_iters.

    Args:
        max_iters: Maximum number of training iterations.

    Returns:
        Sorted list of checkpoint step numbers.
    """
    steps = set()
    for exp in range(1, 7):  # 10^1 to 10^6
        for mantissa in [1, 2, 5]:
            step = mantissa * (10**exp)
            if step <= max_iters:
                steps.add(step)
    # Always include the last step
    steps.add(max_iters - 1)
    return sorted(steps)
