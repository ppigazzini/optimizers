#!/usr/bin/env python3
"""Plot a_k, c_k, and a_k / c_k values for SPSA using default initialization."""

import contextlib
import logging

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, Field, PositiveFloat

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# --- Pydantic Model for SPSA Parameters ---
class SPSAParameters(BaseModel):
    """Default SPSA parameters for plotting."""

    spsa_a: PositiveFloat = 1.0
    spsa_c: PositiveFloat = 0.1
    spsa_alpha: PositiveFloat = 0.602
    spsa_gamma: PositiveFloat = 0.101
    n_steps: int = Field(1000, gt=0)

    @property
    def spsa_capital_a(self) -> float:
        """Compute SPSA_A as 10% of the number of steps."""
        return 0.1 * self.n_steps

    model_config = {"arbitrary_types_allowed": True, "frozen": True}


# --- SPSA Functions ---
def compute_a_k(params: SPSAParameters, k: int) -> float:
    """Compute the learning rate a_k for SPSA."""
    return params.spsa_a / (k + 1 + params.spsa_capital_a) ** params.spsa_alpha


def compute_c_k(params: SPSAParameters, k: int) -> float:
    """Compute the perturbation size c_k for SPSA."""
    return params.spsa_c / (k + 1) ** params.spsa_gamma


# --- Plotting ---
def plot_spsa_parameters(params: SPSAParameters) -> None:
    """Plot a_k, c_k, and a_k / c_k values for SPSA."""
    steps = np.arange(1, params.n_steps + 1)
    a_k_values = [compute_a_k(params, k) for k in steps]
    c_k_values = [compute_c_k(params, k) for k in steps]
    ratio_values = [a / c for a, c in zip(a_k_values, c_k_values, strict=False)]
    r_k_values = [a / (c**2) for a, c in zip(a_k_values, c_k_values, strict=False)]

    fig, ax = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    fig.suptitle("SPSA Parameters: a_k, c_k, a_k / c_k, and a_k / c_k^2")

    ax[0].plot(steps, a_k_values, label="a_k (Learning Rate)", color="blue")
    ax[0].set_ylabel("a_k")
    ax[0].grid(visible=True, linestyle="--", linewidth=0.5)
    ax[0].legend()

    ax[1].plot(steps, c_k_values, label="c_k (Perturbation Size)", color="green")
    ax[1].set_ylabel("c_k")
    ax[1].grid(visible=True, linestyle="--", linewidth=0.5)
    ax[1].legend()

    ax[2].plot(steps, ratio_values, label="a_k / c_k (Ratio)", color="purple")
    ax[2].set_ylabel("a_k / c_k")
    ax[2].grid(visible=True, linestyle="--", linewidth=0.5)
    ax[2].legend()

    ax[3].plot(steps, r_k_values, label="a_k / c_k^2 (r_k)", color="orange")
    ax[3].set_xlabel("Iteration")
    ax[3].set_ylabel("a_k / c_k^2")
    ax[3].grid(visible=True, linestyle="--", linewidth=0.5)
    ax[3].legend()

    plt.tight_layout()
    plt.show()


# --- Main Function ---
def main() -> None:
    """Plot SPSA parameters using default initialization."""
    params = SPSAParameters()
    logger.info("SPSA Parameters:")
    logger.info("  spsa_a: %s", params.spsa_a)
    logger.info("  spsa_c: %s", params.spsa_c)
    logger.info("  spsa_capital_a (A): %s", params.spsa_capital_a)
    logger.info("  spsa_alpha: %s", params.spsa_alpha)
    logger.info("  spsa_gamma: %s", params.spsa_gamma)
    logger.info("  n_steps: %s", params.n_steps)

    plot_spsa_parameters(params)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
