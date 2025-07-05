#!/usr/bin/env python3
"""Compare SGD, Adam-moment and AMSGrad-moment random walks."""

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, NonNegativeFloat, PositiveFloat, ValidationError

# --- type aliases ---
Array2D = npt.NDArray[np.float64]
Metrics = tuple[float, float]

# --- logging setup ---
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# --- Pydantic Models for Configuration ---
PYDANTIC_CONFIG = {
    "arbitrary_types_allowed": True,
    "frozen": True,
}


class WalkConfig(BaseModel):
    """Configuration for the random walk."""

    n_steps: int = Field(..., gt=0)
    step_size: PositiveFloat
    rng: np.random.Generator = Field(default_factory=np.random.default_rng)

    model_config = PYDANTIC_CONFIG


class AdamConfig(BaseModel):
    """Configuration for Adam and AMSGrad optimizers."""

    alpha: PositiveFloat
    beta1: float = Field(..., ge=0.0, lt=1.0)
    beta2: float = Field(..., ge=0.0, lt=1.0)
    eps: PositiveFloat
    noise_scale: NonNegativeFloat

    model_config = PYDANTIC_CONFIG


# --- Core Logic ---
def generate_raw_steps(cfg: WalkConfig, noise_scale: float) -> Array2D:
    """Draw n_steps of 2D Gaussian steps."""
    steps = cfg.rng.normal(0.0, noise_scale, (cfg.n_steps, 2))
    return steps * cfg.step_size


def random_walk_sgd(steps: Array2D, lr: float) -> Array2D:
    """Run a stochastic gradient walk.

    Treats each raw step as 'gradient' g_t,
    and updates x_{t} = x_{t-1} + lr * g_t.
    """
    pos = np.zeros_like(steps)
    for t in range(1, len(steps)):
        pos[t] = pos[t - 1] + lr * steps[t]
    return pos


def random_walk_adam(steps: Array2D, cfg: AdamConfig) -> Array2D:
    """Run an Adam-filtered walk, treating each step as a 'gradient'."""
    pos = np.zeros_like(steps)
    m = np.zeros(2)
    v = np.zeros(2)

    for t in range(1, len(steps)):
        g = steps[t]
        m = cfg.beta1 * m + (1 - cfg.beta1) * g
        v = cfg.beta2 * v + (1 - cfg.beta2) * (g**2)
        m_hat = m / (1 - cfg.beta1**t)
        v_hat = v / (1 - cfg.beta2**t)
        delta = cfg.alpha * m_hat / (np.sqrt(v_hat) + cfg.eps)
        pos[t] = pos[t - 1] + delta

    return pos


def random_walk_amsgrad(steps: Array2D, cfg: AdamConfig) -> Array2D:
    """Run an AMSGrad walk, using the max of second-moment estimates."""
    pos = np.zeros_like(steps)
    m = np.zeros(2)
    v = np.zeros(2)
    v_hat_max = np.zeros(2)

    for t in range(1, len(steps)):
        g = steps[t]
        m = cfg.beta1 * m + (1 - cfg.beta1) * g
        v = cfg.beta2 * v + (1 - cfg.beta2) * (g**2)
        m_hat = m / (1 - cfg.beta1**t)
        v_hat = v / (1 - cfg.beta2**t)
        v_hat_max = np.maximum(v_hat_max, v_hat)
        delta = cfg.alpha * m_hat / (np.sqrt(v_hat_max) + cfg.eps)
        pos[t] = pos[t - 1] + delta

    return pos


# --- Metrics and Plotting ---
def compute_metrics(path: Array2D) -> Metrics:
    """Return (end-to-end displacement, mean squared step length)."""
    disp = np.linalg.norm(path[-1])
    msd = float(np.mean(np.sum(np.diff(path, axis=0) ** 2, axis=1)))
    return disp, msd


def plot_paths(
    sgd: Array2D,
    adam: Array2D,
    amsgrad: Array2D,
    *,
    title: str = "Random Walk Comparison",
) -> None:
    """Plot the paths of different random walks."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(*sgd.T, label="SGD RW", alpha=0.6, lw=1, color="C0")
    ax.plot(*adam.T, label="Adam RW", alpha=0.6, lw=1, color="C1")
    ax.plot(*amsgrad.T, label="AMSGrad RW", alpha=0.6, lw=1, color="C2")

    # Circle markers at endpoints
    ax.scatter(*sgd[-1], s=80, facecolors="none", edgecolors="C0", linewidths=1.5)
    ax.scatter(*adam[-1], s=80, facecolors="none", edgecolors="C1", linewidths=1.5)
    ax.scatter(*amsgrad[-1], s=80, facecolors="none", edgecolors="C2", linewidths=1.5)

    ax.set_aspect("equal", "box")
    ax.grid(visible=True)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


# --- Main Execution ---
def main() -> None:
    """Run the main script."""
    plot_results = True

    walk_cfg = WalkConfig(n_steps=500, step_size=1.0)
    adam_cfg = AdamConfig(
        alpha=0.5,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        noise_scale=1.0,
    )

    raw_steps = generate_raw_steps(walk_cfg, adam_cfg.noise_scale)
    sgd_path = random_walk_sgd(raw_steps, adam_cfg.alpha)
    adam_path = random_walk_adam(raw_steps, adam_cfg)
    amsgrad_path = random_walk_amsgrad(raw_steps, adam_cfg)

    df, msd_f = compute_metrics(sgd_path)
    da, msd_a = compute_metrics(adam_path)
    dam, msd_am = compute_metrics(amsgrad_path)

    logger.info("SGD      RW:   end-to-end = %.3f, mean step² = %.3f", df, msd_f)
    logger.info("Adam     RW:   end-to-end = %.3f, mean step² = %.3f", da, msd_a)
    logger.info("AMSGrad  RW:   end-to-end = %.3f, mean step² = %.3f", dam, msd_am)

    if plot_results:
        plot_paths(sgd_path, adam_path, amsgrad_path)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, ValidationError) as e:
        if isinstance(e, KeyboardInterrupt):
            logger.info("Interrupted by user, exiting.")
        else:
            logger.exception("Configuration error")
        sys.exit(1)
