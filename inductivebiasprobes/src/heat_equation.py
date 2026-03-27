"""
1D Heat Equation Data Generation for the 4th experimental domain.

Solves u_t = alpha * u_xx on [0, 1] with homogeneous Dirichlet BCs
using the exact Fourier series solution.

The "true state" for the IB probe is the diffusivity parameter alpha,
discretized into num_states bins. A model that learns the diffusion operator
should be able to distinguish different alpha values from observed temperatures.
"""

import numpy as np


def exact_heat_solution(x, t, alpha, amplitudes):
    """Compute exact solution of 1D heat equation via Fourier series.

    u(x, t) = sum_k a_k * exp(-k^2 * pi^2 * alpha * t) * sin(k * pi * x)

    Args:
        x: Spatial grid, shape (N_x,).
        t: Time value (scalar).
        alpha: Diffusivity parameter.
        amplitudes: Fourier mode amplitudes, shape (K,).

    Returns:
        Temperature field u(x, t), shape (N_x,).
    """
    u = np.zeros_like(x)
    for k, a_k in enumerate(amplitudes, start=1):
        u += a_k * np.exp(-(k**2) * np.pi**2 * alpha * t) * np.sin(k * np.pi * x)
    return u


def generate_heat_trajectories(
    num_sequences,
    alpha_values,
    N_x=21,
    obs_indices=None,
    N_t=101,
    dt=0.01,
    num_modes=5,
    rng=None,
):
    """Generate heat equation trajectory data.

    Args:
        num_sequences: Number of trajectories to generate.
        alpha_values: List of diffusivity values (defines state space).
        N_x: Number of spatial grid points (including boundaries).
        obs_indices: Indices of observed spatial points. If None, uses 5
            evenly-spaced interior points.
        N_t: Number of time steps.
        dt: Time step size.
        num_modes: Number of Fourier modes for initial conditions.
        rng: numpy RandomState for reproducibility.

    Returns:
        obs: Observed temperatures, shape (num_sequences, N_t, N_obs).
        states: State labels (alpha index), shape (num_sequences, N_t, 1).
        full_temps: Full temperature field, shape (num_sequences, N_t, N_x).
    """
    if rng is None:
        rng = np.random.RandomState(0)

    x = np.linspace(0, 1, N_x)

    if obs_indices is None:
        # 5 evenly-spaced interior observation points
        obs_indices = np.linspace(1, N_x - 2, 5).astype(int)

    N_obs = len(obs_indices)
    num_alphas = len(alpha_values)

    obs_all = np.zeros((num_sequences, N_t, N_obs))
    states_all = np.zeros((num_sequences, N_t, 1), dtype=np.int32)
    full_temps_all = np.zeros((num_sequences, N_t, N_x))

    for i in range(num_sequences):
        # Randomly select alpha
        alpha_idx = rng.randint(0, num_alphas)
        alpha = alpha_values[alpha_idx]

        # Random initial condition amplitudes
        amplitudes = rng.uniform(-1, 1, size=num_modes)

        # Compute solution at each time step
        for n in range(N_t):
            t = n * dt
            u = exact_heat_solution(x, t, alpha, amplitudes)
            full_temps_all[i, n] = u
            obs_all[i, n] = u[obs_indices]

        # State: alpha index (constant per trajectory)
        states_all[i, :, 0] = alpha_idx

    return obs_all, states_all, full_temps_all


def discretize_temperatures(obs, num_bins=50, bin_edges=None):
    """Discretize continuous temperature observations into token indices.

    Args:
        obs: Continuous observations, shape (..., N_obs).
        num_bins: Number of discretization bins.
        bin_edges: Pre-computed bin edges. If None, computed from data.

    Returns:
        obs_discrete: Discretized observations, shape (..., N_obs) of int.
        bin_edges: The bin edges used for discretization.
    """
    if bin_edges is None:
        # Compute bin edges from the data with a small margin
        vmin = obs.min()
        vmax = obs.max()
        margin = (vmax - vmin) * 0.01
        bin_edges = np.linspace(vmin - margin, vmax + margin, num_bins + 1)

    # Digitize: map each value to a bin index in [0, num_bins-1]
    obs_discrete = np.digitize(obs, bin_edges) - 1
    obs_discrete = np.clip(obs_discrete, 0, num_bins - 1)

    return obs_discrete.astype(np.int32), bin_edges
