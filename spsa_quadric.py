"""Quadratic Function and SPSA Optimization.

Algebraic equations:

- Quadric:
    f(x, y) = (a * (x - x₀))² + (b * (y - y₀))²

- Gradient:
    ∇f(x, y) = [
        ∂f/∂x,
        ∂f/∂y
    ] = [
        2a²(x - x₀),
        2b²(y - y₀)
    ]

- Hessian:
    H = [
        [2a²,   0  ],
        [  0 , 2b² ]
    ]

- Computation of cₖ given a noise level σ²:
    For each coordinate (e.g., x):
        f(x + cₓ, y) - f(x, y) = noise_level
        Expand:
            f(x + cₓ, y) = a²(x - x₀)² + 2a²(x - x₀)cₓ + a²cₓ² + b²(y - y₀)²
            f(x, y)     = a²(x - x₀)² + b²(y - y₀)²
            Difference: 2a²(x - x₀)cₓ + a²cₓ² = noise_level
        Rearranged:
            a²cₓ² + 2a²(x - x₀)cₓ - noise_level = 0
        The positive solution:
            cₓ = ( -a(x - x₀) + sqrt( (a(x - x₀))² + noise_level ) ) / a
    Similarly for cᵧ:
        cᵧ = ( -b(y - y₀) + sqrt( (b(y - y₀))² + noise_level ) ) / b

    At the minimum (x = x₀, y = y₀):
        cₓ = sqrt(noise_level) / a
        cᵧ = sqrt(noise_level) / b

- Gradient Descent Update (expanded for the quadric, using noise_level = 1):
    θₖ₊₁ = θₖ - α ∇f(θₖ)
    For θₖ = [xₖ, yₖ]:
        xₖ₊₁ = xₖ - α * 2a²(xₖ - x₀)
        yₖ₊₁ = yₖ - α * 2b²(yₖ - y₀)

    Using cₓ and cᵧ at the minimum (with noise_level = 1):
        Since cₓ = 1 / a, so a = 1 / cₓ, and 2a² = 2 / cₓ²
        So:
            xₖ₊₁ = xₖ - α * (2 / cₓ²) * (xₖ - x₀)
        Similarly for y:
            yₖ₊₁ = yₖ - α * (2 / cᵧ²) * (yₖ - y₀)

- Newton's Method Update (expanded for the quadric, using noise_level = 1):
    θₖ₊₁ = θₖ - α H⁻¹ ∇f(θₖ)
    For the quadric, H⁻¹ = diag(1/(2a²), 1/(2b²)), so:
        xₖ₊₁ = xₖ - α * (1/(2a²)) * 2a²(xₖ - x₀) = xₖ - α * (xₖ - x₀)
        yₖ₊₁ = yₖ - α * (1/(2b²)) * 2b²(yₖ - y₀) = yₖ - α * (yₖ - y₀)

    Using cₓ and cᵧ at the minimum (with noise_level = 1):
        Since cₓ = 1 / a, so a = 1 / cₓ, and 2a² = 2 / cₓ²,
        1/(2a²) = cₓ² / 2
        So:
            xₖ₊₁ = xₖ - α * (cₓ² / 2) * (2 / cₓ²) * (xₖ - x₀)
                 = xₖ - α * (xₖ - x₀)
        Similarly for y:
            yₖ₊₁ = yₖ - α * (yₖ - y₀)
"""

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, Field
from scipy.linalg import inv

# --- Configuration using Pydantic ---


class OptimConfig(BaseModel):
    x0: float = Field(1.0, description="Minimum x-coordinate")
    y0: float = Field(10.0, description="Minimum y-coordinate")
    a: float = Field(1.0, description="Quadratic coefficient for x")
    b: float = Field(2.0, description="Quadratic coefficient for y")
    start_x: float = Field(5.0, description="Starting x-coordinate")
    start_y: float = Field(12.0, description="Starting y-coordinate")
    noise_level: float = Field(1.0, description="Noise level for noisy objective")
    alpha: float = Field(0.1, description="Step size")
    iterations: int = Field(100, description="Number of optimization steps")
    max_step: float = Field(2.0, description="Maximum allowed step per coordinate")
    perturb_eps: float = Field(1e-3, description="Minimum perturbation size for SPSA")
    grad_eps: float = Field(1e-12, description="Minimum denominator for SPSA gradient")


CONFIG = OptimConfig()


# --- Objective and Utility Functions ---


def f(xy: np.ndarray) -> float:
    x, y = xy
    return (CONFIG.a * (x - CONFIG.x0)) ** 2 + (CONFIG.b * (y - CONFIG.y0)) ** 2


def f_noisy(xy: np.ndarray, noise_level: float) -> float:
    x, y = xy
    return (
        (CONFIG.a * (x - CONFIG.x0)) ** 2
        + (CONFIG.b * (y - CONFIG.y0)) ** 2
        + np.random.normal(0, noise_level)
    )


def f_grid(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return (CONFIG.a * (X - CONFIG.x0)) ** 2 + (CONFIG.b * (Y - CONFIG.y0)) ** 2


def algebraic_gradient(theta: np.ndarray) -> np.ndarray:
    x, y = theta
    return np.array(
        [2 * CONFIG.a**2 * (x - CONFIG.x0), 2 * CONFIG.b**2 * (y - CONFIG.y0)],
    )


def algebraic_hessian() -> np.ndarray:
    return np.array([[2 * CONFIG.a**2, 0], [0, 2 * CONFIG.b**2]])


def calculate_perturbation_for_noise_level(
    theta: np.ndarray,
    noise_level: float,
    eps: float = CONFIG.perturb_eps,
) -> np.ndarray:
    x, y = theta
    cx = (
        -CONFIG.a * (x - CONFIG.x0)
        + np.sqrt(np.maximum(0, (CONFIG.a * (x - CONFIG.x0)) ** 2 + noise_level))
    ) / CONFIG.a
    cy = (
        -CONFIG.b * (y - CONFIG.y0)
        + np.sqrt(np.maximum(0, (CONFIG.b * (y - CONFIG.y0)) ** 2 + noise_level))
    ) / CONFIG.b
    cx = np.sign(cx) * max(abs(cx), eps)
    cy = np.sign(cy) * max(abs(cy), eps)
    return np.array([cx, cy])


def spsa_gradient(
    theta: np.ndarray,
    c_k: np.ndarray,
    rng: np.random.Generator,
    noise_level: float,
    eps: float = CONFIG.grad_eps,
) -> np.ndarray:
    delta = rng.choice([-1, 1], size=theta.shape)
    perturb = c_k * delta
    f_plus = f_noisy(theta + perturb, noise_level)
    f_minus = f_noisy(theta - perturb, noise_level)
    denom = 2 * delta * c_k
    denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps + eps, denom)
    return (f_plus - f_minus) / denom


def spsa_newton_gradient(
    theta: np.ndarray,
    c_k: np.ndarray,
    rng: np.random.Generator,
    noise_level: float,
    fixed_at_minimum: bool = False,
) -> np.ndarray:
    g_hat = spsa_gradient(theta, c_k, rng, noise_level)
    if fixed_at_minimum:
        x, y = CONFIG.x0, CONFIG.y0
    else:
        x, y = theta
    c_x, c_y = c_k
    H_xx = 2 * noise_level / (c_x**2 + 2 * (x - CONFIG.x0) * c_x)
    H_yy = 2 * noise_level / (c_y**2 + 2 * (y - CONFIG.y0) * c_y)
    H_diag = np.array([H_xx, H_yy])
    return g_hat / H_diag


def run_optimization(
    method: str = "spsa_gradient",
    alpha: float = CONFIG.alpha,
    iterations: int = CONFIG.iterations,
    ck_mode: str = "adaptive",
) -> np.ndarray:
    theta = np.array([CONFIG.start_x, CONFIG.start_y], dtype=float)
    path = [theta.copy()]
    rng = np.random.default_rng()
    H_true = algebraic_hessian()
    inv_H_true = inv(H_true)
    noise_level = CONFIG.noise_level

    if ck_mode == "fixed_at_minimum":
        c_k_fixed = calculate_perturbation_for_noise_level(
            np.array([CONFIG.x0, CONFIG.y0]),
            noise_level,
        )
    else:
        c_k_fixed = None

    max_step = CONFIG.max_step

    for _ in range(iterations):
        if method in {"spsa_gradient", "spsa_newton"}:
            if ck_mode == "adaptive":
                c_k = calculate_perturbation_for_noise_level(theta, noise_level)
            elif ck_mode == "fixed_at_minimum":
                c_k = c_k_fixed
            else:
                msg = f"Unknown ck_mode: {ck_mode}"
                raise ValueError(msg)

        if method == "spsa_gradient":
            grad = spsa_gradient(theta, c_k, rng, noise_level)
        elif method == "spsa_newton":
            grad = spsa_newton_gradient(
                theta,
                c_k,
                rng,
                noise_level,
                fixed_at_minimum=(ck_mode == "fixed_at_minimum"),
            )
        elif method == "algebraic_gradient":
            grad = algebraic_gradient(theta)
        elif method == "algebraic_newton":
            grad = inv_H_true @ algebraic_gradient(theta)
        else:
            msg = f"Unknown method: {method}"
            raise ValueError(msg)

        grad = np.clip(grad, -max_step / alpha, max_step / alpha)
        theta -= alpha * grad
        path.append(theta.copy())

    return np.array(path)


def plot_paths(optimizers: dict[str, np.ndarray]) -> None:
    plt.figure(figsize=(12, 10))
    all_paths = np.vstack(list(optimizers.values()))
    finite_paths = all_paths[np.isfinite(all_paths).all(axis=1)]
    x_min, y_min = finite_paths.min(axis=0) - 0.5
    x_max, y_max = finite_paths.max(axis=0) + 0.5

    x_grid, y_grid = np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = f_grid(X, Y)

    plt.contourf(X, Y, Z, levels=20, cmap="viridis", alpha=0.5)
    plt.colorbar(label="Function Value")

    colors = ["red", "darkred", "magenta", "purple", "orange", "cyan"]
    markers = [".", ".", "v", "v", "o", "x"]
    linestyles = ["-", "--", "-", "--", "--", "-"]
    for (name, path), color, marker, ls in zip(
        optimizers.items(),
        colors,
        markers,
        linestyles,
        strict=False,
    ):
        plt.plot(
            path[:, 0],
            path[:, 1],
            label=name,
            marker=marker,
            linestyle=ls,
            color=color,
        )
    plt.scatter(
        CONFIG.x0,
        CONFIG.y0,
        color="black",
        s=150,
        zorder=5,
        marker="*",
        label="Minimum",
    )
    plt.title("Comparison of SPSA and Ideal Optimizers")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def print_final_distances(optimizers: dict[str, np.ndarray]) -> None:
    for path in optimizers.values():
        np.linalg.norm(path[-1] - np.array([CONFIG.x0, CONFIG.y0]))


def plot_distance_vs_iteration(optimizers: dict[str, np.ndarray]) -> None:
    plt.figure(figsize=(10, 6))
    for name, path in optimizers.items():
        distances = np.linalg.norm(path - np.array([CONFIG.x0, CONFIG.y0]), axis=1)
        plt.plot(distances, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Distance from Minimum")
    plt.yscale("log")
    plt.title("Distance to Minimum vs. Iteration")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()


def main() -> None:
    optimizers = {
        "SPSA (adaptive c_k)": run_optimization("spsa_gradient", ck_mode="adaptive"),
        "SPSA (fixed c_k@min)": run_optimization(
            "spsa_gradient",
            ck_mode="fixed_at_minimum",
        ),
        "SPSA Newton (adaptive c_k)": run_optimization(
            "spsa_newton",
            ck_mode="adaptive",
        ),
        "SPSA Newton (fixed c_k@min)": run_optimization(
            "spsa_newton",
            ck_mode="fixed_at_minimum",
        ),
        "Algebraic Gradient (Ideal)": run_optimization("algebraic_gradient"),
        "Algebraic Newton (Ideal)": run_optimization("algebraic_newton"),
    }
    plot_paths(optimizers)
    print_final_distances(optimizers)
    plot_distance_vs_iteration(optimizers)


if __name__ == "__main__":
    main()
