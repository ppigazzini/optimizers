#!/usr/bin/env python3
"""Plot SPSA scheduler sequences: a_k, c_k, and r_k = a_k/c_k^2.

This script uses standard defaults:
    A = 0.1 * n_steps
    alpha = 0.602
    gamma = 0.101

Given desired final values c_k_end and r_k_end at k = n_steps,
it computes c_1 and a_1 so that:
    c_k_end = c_1 / n_steps^gamma
    r_k_end = a_1 / (c_k_end ** 2 * (n_steps + A) ** alpha)
and then plots a_k, c_k, and r_k over the iterations.
"""

import matplotlib.pyplot as plt
import numpy as np

# --- SPSA schedule parameters ---
DEFAULT_ALPHA = 0.602
DEFAULT_GAMMA = 0.101


def compute_c0_a0(
    n_steps: int,
    c_k_end: float,
    r_k_end: float,
    alpha: float,
    gamma: float,
):
    """Compute c_0 and a_0 given desired c_k_end and r_k_end at n_steps."""
    c_0 = c_k_end * n_steps**gamma
    a_0 = r_k_end * (c_k_end**2) * n_steps**alpha
    return c_0, a_0


def compute_c1_a1(
    n_steps: int,
    c_k_end: float,
    r_k_end: float,
    alpha: float,
    gamma: float,
    A: float,
):
    """Compute c_1 and a_1 given desired c_k_end and r_k_end at n_steps."""
    c_1 = c_k_end * n_steps**gamma
    a_1 = r_k_end * (c_k_end**2) * (n_steps + A) ** alpha
    return c_1, a_1


def spsa_schedules(
    n_steps: int,
    a_0: float,
    A: float,
    alpha: float,
    c_0: float,
    gamma: float,
):
    k = np.arange(1, n_steps + 1)
    a_k = a_0 / (k + A) ** alpha
    c_k = c_0 / k**gamma
    r_k = a_k / (c_k**2)
    return k, a_k, c_k, r_k


def spsa_schedules(
    n_steps: int,
    a_1: float,
    A: float,
    alpha: float,
    c_1: float,
    gamma: float,
):
    k = np.arange(1, n_steps + 1)
    a_k = a_1 / (k + A) ** alpha
    c_k = c_1 / k**gamma
    r_k = a_k / (c_k**2)
    return k, a_k, c_k, r_k


def plot_spsa_schedules(
    n_steps=1000,
    c_k_end=100,
    r_k_end=0.01,
    A=None,
    alpha=DEFAULT_ALPHA,
    gamma=DEFAULT_GAMMA,
) -> None:
    if A is None:
        A = 0.1 * n_steps

    # Compute c_0 and a_0 to match desired end values
    c_0, a_0 = compute_c0_a0(n_steps, c_k_end, r_k_end, alpha, gamma)

    k, a_k, c_k, r_k = spsa_schedules(n_steps, a_0, A, alpha, c_0, gamma)

    plt.figure(figsize=(10, 7))
    plt.plot(k, a_k, label="a_k", lw=2)
    plt.plot(k, c_k, label="c_k", lw=2)
    plt.plot(k, r_k, label="r_k = a_k / c_k²", lw=2)
    plt.xlabel("Iteration k")
    plt.ylabel("Value")
    plt.yscale("log")
    plt.title(
        f"SPSA Schedules (A={A:.1f}, alpha={alpha}, gamma={gamma})\n"
        f"c_k_end={c_k_end}, r_k_end={r_k_end} ⇒ c_0={c_0:.4g}, a_0={a_0:.4g}",
    )
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Parameters
    n_steps = 50000
    r_k_end = 2.00e-03
    c_k_end_1 = 1
    c_k_end_2 = 23456

    # Compute c_1 and a_1 for both c_k_end values
    A = 0.1 * n_steps
    c1_1, a1_1 = compute_c1_a1(
        n_steps,
        c_k_end_1,
        r_k_end,
        DEFAULT_ALPHA,
        DEFAULT_GAMMA,
        A,
    )
    c1_2, a1_2 = compute_c1_a1(
        n_steps,
        c_k_end_2,
        r_k_end,
        DEFAULT_ALPHA,
        DEFAULT_GAMMA,
        A,
    )

    # Compute schedules
    k, a_k_1, c_k_1, r_k_1 = spsa_schedules(
        n_steps,
        a1_1,
        A,
        DEFAULT_ALPHA,
        c1_1,
        DEFAULT_GAMMA,
    )
    _, a_k_2, c_k_2, r_k_2 = spsa_schedules(
        n_steps,
        a1_2,
        A,
        DEFAULT_ALPHA,
        c1_2,
        DEFAULT_GAMMA,
    )

    # Compute difference in r_k
    r_k_diff = np.abs(r_k_2 - r_k_1)
    max_diff = np.max(r_k_diff)

    # Print r_1 and r_end for the plotted sequence
    r1_start = r_k_1[0]
    r1_end = r_k_1[-1]

    # Plot r_k for only one c_k_end, and print c_k_end, c1, r_1, r_end in the title
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 7))
    plt.plot(k, r_k_1, label=f"r_k (c_k_end={c_k_end_1})", lw=2)
    plt.xlabel("Iteration k")
    plt.ylabel("r_k = a_k / c_k²")
    plt.yscale("log")
    plt.title(
        f"SPSA r_k for c_k_end={c_k_end_1} (c1={c1_1:.4g})\n"
        f"r_1={r1_start:.4g}, r_end={r1_end:.4g}, r_k_end={r_k_end}, n_steps={n_steps}",
    )
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()
