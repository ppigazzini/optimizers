#!/usr/bin/env python3
"""Plot various learning rate scheduler outputs."""

import contextlib
import logging

import matplotlib.pyplot as plt
import numpy as np

# Assuming optimizers.py is in the same directory or accessible in the python path
from optimizers import (
    ConstantScheduler,
    CosineAnnealingScheduler,
    PowerLawDecayScheduler,
    Scheduler,
    WarmupCosineScheduler,
    get_experiment_configs,
)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# --- Plotting ---
def plot_scheduler_outputs(
    schedulers: dict[str, Scheduler],
    n_steps: int,
    title: str,
) -> None:
    """Plot the learning rate output of multiple schedulers."""
    steps = np.arange(1, n_steps + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title)

    for name, scheduler in schedulers.items():
        lr_values = [scheduler.get_learning_rate(t) for t in steps]
        ax.plot(steps, lr_values, label=name, lw=2)

    ax.set(
        xlabel="Iteration",
        ylabel="Learning Rate",
        yscale="log",
    )
    ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()


# --- Main Function ---
def main() -> None:
    """Plot various learning rate schedules based on Rosenbrock experiment."""
    # Get experiment configurations from optimizers.py
    experiment_configs = get_experiment_configs()

    # Find the Rosenbrock experiment configuration to use its parameters
    rosenbrock_config = next(
        (c for c in experiment_configs if c.name == "Rosenbrock"),
        None,
    )

    if not rosenbrock_config:
        logger.error("Rosenbrock experiment configuration not found in optimizers.py.")
        return

    n_steps = rosenbrock_config.n_steps
    hyperparams = rosenbrock_config.hyperparams

    logger.info("Plotting schedulers with parameters from Rosenbrock experiment:")
    logger.info("  n_steps: %s", n_steps)
    logger.info("  learning_rate: %s", hyperparams.learning_rate)
    logger.info("  cosine_annealing_t_max: %s", hyperparams.cosine_annealing_t_max)
    logger.info("  warmup_steps: %s", hyperparams.warmup_steps)

    # Instantiate all schedulers using the loaded hyperparameters
    schedulers_to_plot = {
        "Constant": ConstantScheduler(learning_rate=hyperparams.learning_rate),
        "PowerLawDecay": PowerLawDecayScheduler(
            a=hyperparams.spsa_a,
            A=hyperparams.spsa_capital_a,
            alpha=hyperparams.spsa_alpha,
        ),
        "CosineAnnealing (Warm Restart)": CosineAnnealingScheduler(
            t_max=hyperparams.cosine_annealing_t_max,
            eta_min=hyperparams.cosine_annealing_eta_min,
            eta_max=hyperparams.learning_rate,
        ),
        "Warmup + CosineDecay": WarmupCosineScheduler(
            warmup_steps=hyperparams.warmup_steps,
            t_max=n_steps,
            eta_max=hyperparams.learning_rate,
            eta_min=hyperparams.cosine_annealing_eta_min,
        ),
    }

    plot_scheduler_outputs(
        schedulers_to_plot,
        n_steps,
        "Learning Rate Scheduler Comparison (using Rosenbrock config)",
    )


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
