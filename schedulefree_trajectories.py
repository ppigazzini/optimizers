"""Schedule-free optimizer trajectories and analysis on the Rosenbrock function."""

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

logger = logging.getLogger(__name__)


def setup_logging(loglevel: str = "INFO") -> None:
    """Set up logging with the specified log level."""
    logging.basicConfig(
        level=getattr(logging, loglevel.upper(), "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def rosenbrock(z: np.ndarray) -> float:
    """Compute the Rosenbrock function value at point z."""
    x, y = z
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def grad_rosen(z: np.ndarray) -> np.ndarray:
    """Compute the gradient of the Rosenbrock function at point z."""
    x, y = z
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])


@dataclass
class SgdParams:
    """Parameters for schedule-free SGD."""

    gamma: float = 1e-3
    beta: float = 0.9
    warmup_steps: int = 0


@dataclass
class AdamParams:
    """Parameters for schedule-free Adam."""

    gamma: float = 1e-3
    beta: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    warmup_steps: int = 0


def schedule_free_sgd(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    z0: np.ndarray,
    t: int = 500,
    params: SgdParams = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Run schedule-free SGD with Polyak-Ruppert averaging."""
    if params is None:
        params = SgdParams()
    np.random.default_rng(seed)
    z_seq = np.zeros((t + 1, 2))
    x_seq = np.zeros((t + 1, 2))
    y_seq = np.zeros((t + 1, 2))
    z_seq[0] = z0
    x_seq[0] = z0.copy()
    for i in range(t):
        if params.warmup_steps > 0 and i < params.warmup_steps:
            gamma_i = params.gamma * (i + 1) / params.warmup_steps
        else:
            gamma_i = params.gamma
        y = (1 - params.beta) * z_seq[i] + params.beta * x_seq[i]
        grad = grad_fn(y)
        z_next = z_seq[i] - gamma_i * grad
        c = 1.0 / (i + 2)
        x_next = (1 - c) * x_seq[i] + c * z_next
        y_seq[i] = y
        z_seq[i + 1] = z_next
        x_seq[i + 1] = x_next
    y_seq[t] = (1 - params.beta) * z_seq[t] + params.beta * x_seq[t]
    return {"z": z_seq, "x": x_seq, "y": y_seq}


def schedule_free_adam(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    z0: np.ndarray,
    t: int = 500,
    params: AdamParams = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Run schedule-free Adam with Polyak-Ruppert averaging."""
    if params is None:
        params = AdamParams()
    np.random.default_rng(seed)
    z_seq = np.zeros((t + 1, 2))
    x_seq = np.zeros((t + 1, 2))
    y_seq = np.zeros((t + 1, 2))
    m = np.zeros(2)
    v = np.zeros(2)
    z_seq[0] = z0
    x_seq[0] = z0.copy()
    for i in range(t):
        if params.warmup_steps > 0 and i < params.warmup_steps:
            gamma_i = params.gamma * (i + 1) / params.warmup_steps
        else:
            gamma_i = params.gamma
        y = (1 - params.beta) * z_seq[i] + params.beta * x_seq[i]
        grad = grad_fn(y)
        m = params.beta1 * m + (1 - params.beta1) * grad
        v = params.beta2 * v + (1 - params.beta2) * (grad**2)
        m_hat = m / (1 - params.beta1 ** (i + 1))
        v_hat = v / (1 - params.beta2 ** (i + 1))
        z_next = z_seq[i] - gamma_i * m_hat / (np.sqrt(v_hat) + params.eps)
        c = 1.0 / (i + 2)
        x_next = (1 - c) * x_seq[i] + c * z_next
        y_seq[i] = y
        z_seq[i + 1] = z_next
        x_seq[i + 1] = x_next
    y_seq[t] = (1 - params.beta) * z_seq[t] + params.beta * x_seq[t]
    return {"z": z_seq, "x": x_seq, "y": y_seq}


def get_trajectory_bounds(
    *trajs: dict[str, np.ndarray],
) -> tuple[float, float, float, float]:
    """Compute plot bounds to include all trajectories and the minimum."""
    all_points = np.concatenate(
        [np.concatenate([t["z"], t["x"], t["y"]], axis=0) for t in trajs],
        axis=0,
    )
    min_x, max_x = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    min_y, max_y = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    min_x = min(min_x, 1.0)
    max_x = max(max_x, 1.0)
    min_y = min(min_y, 1.0)
    max_y = max(max_y, 1.0)
    margin_x = 0.1 * (max_x - min_x)
    margin_y = 0.1 * (max_y - min_y)
    return min_x - margin_x, max_x + margin_x, min_y - margin_y, max_y + margin_y


def plot_trajectories(
    traj: dict[str, np.ndarray],
    func: Callable[[np.ndarray], float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    levels: int = 20,
    save_path: str | None = None,
    title: str = "Schedule-Free SGD Trajectories on Rosenbrock Function",
) -> None:
    """Plot optimizer trajectories and distance to minimum."""
    xx = np.linspace(xlim[0], xlim[1], 400)
    yy = np.linspace(ylim[0], ylim[1], 400)
    x_mesh, y_mesh = np.meshgrid(xx, yy)
    z_mesh = func(np.array([x_mesh, y_mesh]))
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    contour_levels = np.logspace(-0.5, 3.5, levels)
    axs[0].contour(
        x_mesh,
        y_mesh,
        z_mesh,
        levels=contour_levels,
        norm=colors.LogNorm(),
        cmap="viridis",
        alpha=0.2,
    )
    axs[0].plot(
        traj["z"][:, 0],
        traj["z"][:, 1],
        "-o",
        color="C0",
        markersize=2,
        label="z (base)",
    )
    axs[0].plot(
        traj["x"][:, 0],
        traj["x"][:, 1],
        "-o",
        color="C1",
        markersize=2,
        label="x (averaged)",
    )
    axs[0].plot(
        traj["y"][:, 0],
        traj["y"][:, 1],
        "-o",
        color="C2",
        markersize=2,
        label="y (grad eval)",
    )
    axs[0].scatter(
        traj["z"][0, 0],
        traj["z"][0, 1],
        c="black",
        s=50,
        label="start",
        zorder=5,
    )
    axs[0].scatter(
        traj["x"][-1, 0],
        traj["x"][-1, 1],
        c="red",
        s=50,
        label="x_final",
        zorder=5,
    )
    axs[0].scatter(1, 1, c="gold", s=80, marker="*", label="minimum (1,1)", zorder=6)
    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title(title)
    axs[0].legend()
    axs[0].grid(visible=True)
    min_point = np.array([1.0, 1.0])
    dist_z = np.linalg.norm(traj["z"] - min_point, axis=1)
    dist_x = np.linalg.norm(traj["x"] - min_point, axis=1)
    dist_y = np.linalg.norm(traj["y"] - min_point, axis=1)
    axs[1].plot(dist_z, label="z (base)", color="C0")
    axs[1].plot(dist_x, label="x (averaged)", color="C1")
    axs[1].plot(dist_y, label="y (grad eval)", color="C2")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Distance to minimum")
    axs[1].set_title("Distance to Minimum vs Iteration")
    axs[1].legend()
    axs[1].grid(visible=True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info("Plot saved to %s", save_path)
    plt.show()


def main(args: list[str] | None = None) -> None:
    """Run schedule-free optimizers and plot results."""
    parser = argparse.ArgumentParser(
        description="Schedule-Free SGD and Adam on Rosenbrock Function",
    )
    parser.add_argument("--T", type=int, default=5000, help="Number of iterations")
    parser.add_argument(
        "--gamma_sgd",
        type=float,
        default=5e-3,
        help="Step size for SGD",
    )
    parser.add_argument(
        "--gamma_adam",
        type=float,
        default=3e-2,
        help="Step size for Adam",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.9,
        help="Momentum interpolation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps for gamma",
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default=None,
        help="Path to save plot (SGD)",
    )
    parser.add_argument(
        "--save_plot_adam",
        type=str,
        default=None,
        help="Path to save plot (Adam)",
    )
    parser.add_argument("--loglevel", type=str, default="INFO", help="Logging level")
    parsed_args = parser.parse_args(args)
    setup_logging(parsed_args.loglevel)
    z0 = np.array([-1.2, 1.0])
    sgd_params = SgdParams(
        gamma=parsed_args.gamma_sgd,
        beta=parsed_args.beta,
        warmup_steps=parsed_args.warmup_steps,
    )
    adam_params = AdamParams(
        gamma=parsed_args.gamma_adam,
        beta=parsed_args.beta,
        warmup_steps=parsed_args.warmup_steps,
    )
    logger.info(
        "Starting Schedule-Free SGD with T=%d, gamma_sgd=%g, beta=%g, seed=%d, "
        "warmup_steps=%d",
        parsed_args.T,
        parsed_args.gamma_sgd,
        parsed_args.beta,
        parsed_args.seed,
        parsed_args.warmup_steps,
    )
    traj_sgd = schedule_free_sgd(
        grad_fn=grad_rosen,
        z0=z0,
        t=parsed_args.T,
        params=sgd_params,
        seed=parsed_args.seed,
    )
    logger.info(
        "Starting Schedule-Free Adam with T=%d, gamma_adam=%g, beta=%g, seed=%d, "
        "warmup_steps=%d",
        parsed_args.T,
        parsed_args.gamma_adam,
        parsed_args.beta,
        parsed_args.seed,
        parsed_args.warmup_steps,
    )
    traj_adam = schedule_free_adam(
        grad_fn=grad_rosen,
        z0=z0,
        t=parsed_args.T,
        params=adam_params,
        seed=parsed_args.seed,
    )
    x_min, x_max, y_min, y_max = get_trajectory_bounds(traj_sgd, traj_adam)
    plot_trajectories(
        traj_sgd,
        lambda z: (1 - z[0]) ** 2 + 100 * (z[1] - z[0] ** 2) ** 2,
        xlim=(x_min, x_max),
        ylim=(y_min, y_max),
        save_path=parsed_args.save_plot,
        title="Schedule-Free SGD Trajectories on Rosenbrock Function",
    )
    plot_trajectories(
        traj_adam,
        lambda z: (1 - z[0]) ** 2 + 100 * (z[1] - z[0] ** 2) ** 2,
        xlim=(x_min, x_max),
        ylim=(y_min, y_max),
        save_path=parsed_args.save_plot_adam,
        title="Schedule-Free Adam Trajectories on Rosenbrock Function",
    )


if __name__ == "__main__":
    main()
