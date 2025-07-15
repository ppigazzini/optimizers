"""A script to compare various optimizers on the Rosenbrock function."""

import abc
import logging
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# --- logging setup ---
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

Array1D = npt.NDArray[np.float64]
Array2D = npt.NDArray[np.float64]


# Rosenbrock function and gradient
def rosenbrock(x: Array1D) -> float:
    """Calculate the Rosenbrock function value with minimum at (1, 1000)."""
    x_prime = x[0]
    y_prime = x[1] / 10.0 - 99.0
    return (1 - x_prime) ** 2 + 100 * (y_prime - x_prime**2) ** 2


def grad_rosenbrock(
    x: Array1D,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Array1D:
    """Calculate the gradient of the Rosenbrock function with minimum at (1, 1000)."""
    x_prime = x[0]
    y_prime = x[1] / 10.0 - 99.0
    # Chain rule for gradient
    # Let f(x', y') = (1 - x')^2 + 100 * (y' - x'^2)^2
    # x' = x, y' = y/10 - 99
    # df/dx = (df/dx' * dx'/dx) + (df/dy' * dy'/dx)
    # df/dy = (df/dx' * dx'/dy) + (df/dy' * dy'/dy)
    # dx'/dx = 1, dy'/dx = 0
    # dx'/dy = 0, dy'/dy = 1/10
    df_dx_prime = -2 * (1 - x_prime) - 400 * x_prime * (y_prime - x_prime**2)
    df_dy_prime = 200 * (y_prime - x_prime**2)
    dx = df_dx_prime
    dy = df_dy_prime * (1.0 / 10.0)
    gradient = np.array([dx, dy])
    if noise_scale > 0 and rng:
        noise = rng.normal(scale=noise_scale, size=gradient.shape)
        gradient += noise
    return gradient


# ==============================================================================
# === SURROGATE MODELS =========================================================
# ==============================================================================


class BaseGradSurrogate(abc.ABC):
    """Abstract base class for gradient surrogate models."""

    @abc.abstractmethod
    def fit(self, x_new: np.ndarray, g_new: np.ndarray) -> None:
        """Fit the surrogate model to new data."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict_grad(self, x: np.ndarray) -> np.ndarray:
        """Predict the gradient at a given point."""
        raise NotImplementedError


# Kernel Grad Surrogate Class
class KernelRidgeSurrogate(BaseGradSurrogate):
    """A surrogate model for gradients using dual-form kernel ridge regression."""

    def __init__(
        self,
        d: int,
        lengthscale: float = 1.0,
        sigma_f: float = 1.0,
        lambda_reg: float = 1e-3,
        memory_size: int = 100,
    ) -> None:
        """Initialize the Kernel Ridge surrogate model."""
        self.d = d
        self.l2 = lengthscale**2
        self.sigma_f2 = sigma_f**2
        self.lambda_reg = lambda_reg
        self.memory_size = memory_size
        # Use deque for efficient fixed-size memory
        self.x_data: deque = deque(maxlen=self.memory_size)
        self.g_data: deque = deque(maxlen=self.memory_size)
        self.alpha: np.ndarray | None = None

    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Compute the squared exponential kernel matrix between two sets of points."""
        # Using broadcasting to compute pairwise squared Euclidean distances
        # x1 is (n, d), x2 is (m, d)
        # The result is an (n, m) matrix of distances.
        x1_sq = np.sum(x1**2, axis=1)[:, np.newaxis]
        x2_sq = np.sum(x2**2, axis=1)[np.newaxis, :]
        sq_dist = x1_sq + x2_sq - 2 * (x1 @ x2.T)
        return self.sigma_f2 * np.exp(-0.5 * sq_dist / self.l2)

    def fit(self, x_new: np.ndarray, g_new: np.ndarray) -> None:
        """Fit the surrogate model to the data, managing a fixed-size memory."""
        self.x_data.append(x_new)
        self.g_data.append(g_new)

        x_fit = np.array(self.x_data)
        g_fit = np.array(self.g_data)
        n_samples = x_fit.shape[0]

        # K: (n_samples, n_samples) Gram matrix
        kernel_matrix = self._kernel(x_fit, x_fit)

        # Solve (K + λI)alpha = G for the dual coefficients alpha.
        # alpha is a matrix of size (n_samples, d).
        regularized_kernel = kernel_matrix + self.lambda_reg * np.eye(n_samples)
        self.alpha = np.linalg.solve(regularized_kernel, g_fit)

    def predict_grad(self, x: np.ndarray) -> np.ndarray:
        """Predict the gradient at a given point."""
        if self.alpha is None or not self.x_data:
            msg = "The model has not been fitted yet."
            raise RuntimeError(msg)

        # Ensure x is 2D for consistent kernel computation
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # k_x: (1, n_samples) kernel vector between new point x and training data
        kernel_vector = self._kernel(x, np.array(self.x_data))

        # Predicted gradient is a weighted sum of training gradients,
        # where weights are determined by the kernel.
        # More accurately, it's k_x @ alpha.
        # Result is (1, d), so we flatten it to (d,).
        return (kernel_vector @ self.alpha).flatten()


class RFFGradSurrogate(BaseGradSurrogate):
    """A surrogate model for gradients using random Fourier features.
    https://jmlr.org/papers/volume22/20-1369/20-1369.pdf
    https://arxiv.org/pdf/2307.05855.
    """

    def __init__(  # noqa: PLR0913
        self,
        d: int,
        d_rff: int = 300,
        lengthscale: float = 1.0,
        sigma_f: float = 1.0,
        lambda_reg: float = 1e-3,
        memory_size: int = 100,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize the RFF gradient surrogate model."""
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        self.d, self.D = d, d_rff
        self.l2 = lengthscale**2
        self.sigma_f2 = sigma_f**2
        self.lambda_reg = lambda_reg
        self.memory_size = memory_size
        # random weights for RFF
        self.omega = self.rng.standard_normal((self.D, self.d)) / lengthscale
        self.b = 2 * np.pi * self.rng.random(self.D)
        self.W: np.ndarray | None = None
        self.x_data: deque = deque(maxlen=self.memory_size)
        self.g_data: deque = deque(maxlen=self.memory_size)

    def z(self, x: np.ndarray) -> np.ndarray:
        """Compute the random Fourier features for a given input."""
        # φ:  R^d → R^D
        return np.sqrt(2.0 / self.D) * np.cos(self.omega @ x + self.b)

    def fit(self, x_new: np.ndarray, g_new: np.ndarray) -> None:
        """Fit the surrogate model to the data."""
        self.x_data.append(x_new)
        self.g_data.append(g_new)

        x_fit = np.array(self.x_data)
        g_fit = np.array(self.g_data)
        n, d = x_fit.shape
        z_matrix = np.vstack([self.z(x_fit[i]) for i in range(n)])  # n x D
        # Solve for W in ||Z W - G||^2 + λ||W||^2
        # Each column of W solves (Z^T Z + λ I) w_j = Z^T G[:,j]
        zz = z_matrix.T @ z_matrix
        reg = zz + self.lambda_reg * np.eye(self.D)
        self.W = np.zeros((self.D, d))
        for j in range(d):
            self.W[:, j] = np.linalg.solve(reg, z_matrix.T @ g_fit[:, j])

    def predict_grad(self, x: np.ndarray) -> np.ndarray:
        """Predict the gradient at a given point."""
        # ∇f̃(x) ≈ W^T z(x)
        if self.W is None:
            msg = "The model has not been fitted yet."
            raise RuntimeError(msg)
        return self.W.T @ self.z(x)


# ==============================================================================
# === STANDARD OPTIMIZERS ======================================================
# ==============================================================================


def adam(  # noqa: PLR0913
    x0: Array1D,
    lr: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    steps: int = 40000,
    warmup_steps: int = 100,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Array2D:
    """Adam optimizer."""
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    path = [x.copy()]
    for t in range(1, steps + 1):
        lr_t = lr * t / warmup_steps if t < warmup_steps else lr
        g = grad_rosenbrock(x, noise_scale=noise_scale, rng=rng)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x -= lr_t * m_hat / (np.sqrt(v_hat) + eps)
        path.append(x.copy())
    return np.array(path)


def adabelief(  # noqa: PLR0913
    x0: Array1D,
    lr: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    steps: int = 40000,
    warmup_steps: int = 100,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Array2D:
    """AdaBelief optimizer."""
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    path = [x.copy()]
    for t in range(1, steps + 1):
        lr_t = lr * t / warmup_steps if t < warmup_steps else lr
        g = grad_rosenbrock(x, noise_scale=noise_scale, rng=rng)
        m = beta1 * m + (1 - beta1) * g
        s = g - m
        v = beta2 * v + (1 - beta2) * (s**2) + eps
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x -= lr_t * m_hat / (np.sqrt(v_hat) + eps)
        path.append(x.copy())
    return np.array(path)


def ademamix(  # noqa: PLR0913
    x0: Array1D,
    lr: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    beta3: float = 0.9999,
    alpha: float = 5.0,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    steps: int = 40000,
    warmup_steps: int = 100,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Array2D:
    """AdEMAMix optimizer."""
    x = x0.copy()
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    v = np.zeros_like(x)
    path = [x.copy()]
    for t in range(1, steps + 1):
        lr_t = lr * t / warmup_steps if t < warmup_steps else lr
        g = grad_rosenbrock(x, noise_scale=noise_scale, rng=rng)

        # Decoupled weight decay
        if weight_decay > 0:
            x -= lr * weight_decay * x

        m1 = beta1 * m1 + (1 - beta1) * g
        m2 = beta3 * m2 + (1 - beta3) * g
        v = beta2 * v + (1 - beta2) * (g**2)

        m1_hat = m1 / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        update_numerator = m1_hat + alpha * m2
        update_denominator = np.sqrt(v_hat) + eps

        x -= lr_t * update_numerator / update_denominator
        path.append(x.copy())
    return np.array(path)


# ==============================================================================
# === SCHEDULE-FREE OPTIMIZERS =================================================
# ==============================================================================


def schedule_free_sgd(  # noqa: PLR0913
    x0: Array1D,
    lr: float = 0.001,
    beta: float = 0.98,
    steps: int = 40000,
    warmup_steps: int = 1000,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Array2D:
    """Schedule-Free SGD optimizer."""
    z = x0.copy()
    x = x0.copy()
    path = [x.copy()]
    for t in range(1, steps + 1):
        lr_t = lr * t / warmup_steps if t < warmup_steps else lr
        y = (1 - beta) * z + beta * x
        g = grad_rosenbrock(y, noise_scale=noise_scale, rng=rng)
        z = z - lr_t * g
        c = 1 / (t + 1)
        x = (1 - c) * x + c * z
        path.append(x.copy())
    return np.array(path)


def schedule_free_adam(  # noqa: PLR0913
    x0: Array1D,
    lr: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    beta_sf: float = 0.9,
    eps: float = 1e-8,
    steps: int = 40000,
    warmup_steps: int = 100,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Array2D:
    """Schedule-Free Adam optimizer."""
    z = x0.copy()
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    path = [x.copy()]
    for t in range(1, steps + 1):
        lr_t = lr * t / warmup_steps if t < warmup_steps else lr
        y = (1 - beta_sf) * z + beta_sf * x
        g = grad_rosenbrock(y, noise_scale=noise_scale, rng=rng)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        update = m_hat / (np.sqrt(v_hat) + eps)
        z -= lr_t * update
        c = 1 / (t + 1)
        x = (1 - c) * x + c * z
        path.append(x.copy())
    return np.array(path)


def schedule_free_adabelief(  # noqa: PLR0913
    x0: Array1D,
    lr: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    beta_sf: float = 0.9,
    eps: float = 1e-8,
    steps: int = 40000,
    warmup_steps: int = 100,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Array2D:
    """Schedule-Free AdaBelief optimizer."""
    z = x0.copy()
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    path = [x.copy()]
    for t in range(1, steps + 1):
        lr_t = lr * t / warmup_steps if t < warmup_steps else lr
        y = (1 - beta_sf) * z + beta_sf * x
        g = grad_rosenbrock(y, noise_scale=noise_scale, rng=rng)
        m = beta1 * m + (1 - beta1) * g
        s = g - m
        v = beta2 * v + (1 - beta2) * (s**2) + eps
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        update = m_hat / (np.sqrt(v_hat) + eps)
        z -= lr_t * update
        c = 1 / (t + 1)
        x = (1 - c) * x + c * z
        path.append(x.copy())
    return np.array(path)


def schedule_free_ademamix(  # noqa: PLR0913
    x0: Array1D,
    lr: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    beta3: float = 0.9999,
    alpha: float = 5.0,
    beta_sf: float = 0.9,
    eps: float = 1e-8,
    steps: int = 40000,
    warmup_steps: int = 100,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Array2D:
    """Schedule-Free AdEMAMix optimizer."""
    z = x0.copy()
    x = x0.copy()
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    v = np.zeros_like(x)
    path = [x.copy()]
    for t in range(1, steps + 1):
        lr_t = lr * t / warmup_steps if t < warmup_steps else lr
        y = (1 - beta_sf) * z + beta_sf * x
        g = grad_rosenbrock(y, noise_scale=noise_scale, rng=rng)

        m1 = beta1 * m1 + (1 - beta1) * g
        m2 = beta3 * m2 + (1 - beta3) * g
        v = beta2 * v + (1 - beta2) * (g**2)

        m1_hat = m1 / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        update_numerator = m1_hat + alpha * m2
        update_denominator = np.sqrt(v_hat) + eps
        update = update_numerator / update_denominator

        z -= lr_t * update
        c = 1 / (t + 1)
        x = (1 - c) * x + c * z
        path.append(x.copy())
    return np.array(path)


# ==============================================================================
# === SURROGATE OPTIMIZERS =====================================================
# ==============================================================================


def surrogate_adam_optimizer(  # noqa: PLR0913
    x0: Array1D,
    lr: float = 0.1,
    steps: int = 40000,
    warmup_steps: int = 100,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
    refit_every: int = 20,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    surrogate_type: str = "kernel_ridge",
) -> Array2D:
    """Optimizer using a surrogate gradient model and an Adam update rule."""
    if rng is None:
        rng = np.random.default_rng()
    x = x0.copy()
    path = [x.copy()]
    if surrogate_type == "kernel_ridge":
        surrogate: BaseGradSurrogate = KernelRidgeSurrogate(d=x0.size)
    elif surrogate_type == "rff":
        surrogate = RFFGradSurrogate(d=x0.size, rng=rng)
    else:
        msg = f"Unknown surrogate type: {surrogate_type}"
        raise ValueError(msg)

    # Adam state variables
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for t in range(1, steps + 1):
        lr_t = lr * t / warmup_steps if t < warmup_steps else lr
        # Decide whether to use the true gradient or the surrogate.
        is_model_fitted = (
            hasattr(surrogate, "alpha") and surrogate.alpha is not None
        ) or (hasattr(surrogate, "W") and surrogate.W is not None)
        if not is_model_fitted or t % refit_every == 0:
            # Expensive step: calculate the true gradient
            g_for_step = grad_rosenbrock(x, noise_scale=noise_scale, rng=rng)
            # Store this new ground-truth data point and refit
            surrogate.fit(x.copy(), g_for_step.copy())
        else:
            # Cheap step: use the surrogate model to predict the gradient
            g_for_step = surrogate.predict_grad(x)

        # Adam update logic
        m = beta1 * m + (1 - beta1) * g_for_step
        v = beta2 * v + (1 - beta2) * (g_for_step**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x -= lr_t * m_hat / (np.sqrt(v_hat) + eps)
        path.append(x.copy())
    return np.array(path)


def surrogate_adabelief_optimizer(  # noqa: PLR0913
    x0: Array1D,
    lr: float = 0.1,
    steps: int = 40000,
    warmup_steps: int = 100,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
    refit_every: int = 20,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    surrogate_type: str = "kernel_ridge",
) -> Array2D:
    """Optimizer using a surrogate gradient model and an AdaBelief update rule."""
    if rng is None:
        rng = np.random.default_rng()
    x = x0.copy()
    path = [x.copy()]
    if surrogate_type == "kernel_ridge":
        surrogate: BaseGradSurrogate = KernelRidgeSurrogate(d=x0.size)
    elif surrogate_type == "rff":
        surrogate = RFFGradSurrogate(d=x0.size, rng=rng)
    else:
        msg = f"Unknown surrogate type: {surrogate_type}"
        raise ValueError(msg)

    # AdaBelief state variables
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for t in range(1, steps + 1):
        lr_t = lr * t / warmup_steps if t < warmup_steps else lr
        # Decide whether to use the true gradient or the surrogate.
        is_model_fitted = (
            hasattr(surrogate, "alpha") and surrogate.alpha is not None
        ) or (hasattr(surrogate, "W") and surrogate.W is not None)
        if not is_model_fitted or t % refit_every == 0:
            # Expensive step: calculate the true gradient
            g_for_step = grad_rosenbrock(x, noise_scale=noise_scale, rng=rng)
            # Store this new ground-truth data point and refit
            surrogate.fit(x.copy(), g_for_step.copy())
        else:
            # Cheap step: use the surrogate model to predict the gradient
            g_for_step = surrogate.predict_grad(x)

        # AdaBelief update logic
        m = beta1 * m + (1 - beta1) * g_for_step
        s = g_for_step - m
        v = beta2 * v + (1 - beta2) * (s**2) + eps
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x -= lr_t * m_hat / (np.sqrt(v_hat) + eps)
        path.append(x.copy())
    return np.array(path)


def surrogate_ademamix_optimizer(  # noqa: PLR0913
    x0: Array1D,
    lr: float = 0.1,
    steps: int = 40000,
    warmup_steps: int = 100,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
    refit_every: int = 20,
    beta1: float = 0.9,
    beta2: float = 0.999,
    beta3: float = 0.9999,
    alpha: float = 5.0,
    eps: float = 1e-8,
    surrogate_type: str = "kernel_ridge",
) -> Array2D:
    """Optimizer using a surrogate gradient model and an AdEMAMix update rule."""
    if rng is None:
        rng = np.random.default_rng()
    x = x0.copy()
    path = [x.copy()]
    if surrogate_type == "kernel_ridge":
        surrogate: BaseGradSurrogate = KernelRidgeSurrogate(d=x0.size)
    elif surrogate_type == "rff":
        surrogate = RFFGradSurrogate(d=x0.size, rng=rng)
    else:
        msg = f"Unknown surrogate type: {surrogate_type}"
        raise ValueError(msg)

    # AdEMAMix state variables
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    v = np.zeros_like(x)

    for t in range(1, steps + 1):
        lr_t = lr * t / warmup_steps if t < warmup_steps else lr
        # Decide whether to use the true gradient or the surrogate.
        is_model_fitted = (
            hasattr(surrogate, "alpha") and surrogate.alpha is not None
        ) or (hasattr(surrogate, "W") and surrogate.W is not None)
        if not is_model_fitted or t % refit_every == 0:
            # Expensive step: calculate the true gradient
            g_for_step = grad_rosenbrock(x, noise_scale=noise_scale, rng=rng)
            # Store this new ground-truth data point and refit
            surrogate.fit(x.copy(), g_for_step.copy())
        else:
            # Cheap step: use the surrogate model to predict the gradient
            g_for_step = surrogate.predict_grad(x)

        # AdEMAMix update logic
        m1 = beta1 * m1 + (1 - beta1) * g_for_step
        m2 = beta3 * m2 + (1 - beta3) * g_for_step
        v = beta2 * v + (1 - beta2) * (g_for_step**2)

        m1_hat = m1 / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        update_numerator = m1_hat + alpha * m2
        update_denominator = np.sqrt(v_hat) + eps

        x -= lr_t * update_numerator / update_denominator
        path.append(x.copy())
    return np.array(path)


# ==============================================================================
# === SURROGATE SCHEDULE-FREE OPTIMIZERS =======================================
# ==============================================================================


def surrogate_schedule_free_adam_optimizer(  # noqa: PLR0913
    x0: Array1D,
    lr: float = 0.1,
    steps: int = 40000,
    warmup_steps: int = 100,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
    refit_every: int = 20,
    beta1: float = 0.9,
    beta2: float = 0.999,
    beta_sf: float = 0.9,
    eps: float = 1e-8,
    surrogate_type: str = "kernel_ridge",
) -> Array2D:
    """Optimizer using a surrogate model and a Schedule-Free Adam rule."""
    if rng is None:
        rng = np.random.default_rng()
    z = x0.copy()
    x = x0.copy()
    path = [x.copy()]
    if surrogate_type == "kernel_ridge":
        surrogate: BaseGradSurrogate = KernelRidgeSurrogate(d=x0.size)
    elif surrogate_type == "rff":
        surrogate = RFFGradSurrogate(d=x0.size, rng=rng)
    else:
        msg = f"Unknown surrogate type: {surrogate_type}"
        raise ValueError(msg)

    # Adam state variables
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for t in range(1, steps + 1):
        lr_t = lr * t / warmup_steps if t < warmup_steps else lr
        y = (1 - beta_sf) * z + beta_sf * x

        # Decide whether to use the true gradient or the surrogate.
        # We calculate the true gradient only if the model isn't ready,
        # or if it's a designated "refit" step.
        is_model_fitted = (
            hasattr(surrogate, "alpha") and surrogate.alpha is not None
        ) or (hasattr(surrogate, "W") and surrogate.W is not None)
        if not is_model_fitted or t % refit_every == 0:
            # Expensive step: calculate the true gradient
            g_for_step = grad_rosenbrock(y, noise_scale=noise_scale, rng=rng)
            # Store this new ground-truth data point and refit
            surrogate.fit(y.copy(), g_for_step.copy())
        else:
            # Cheap step: use the surrogate model to predict the gradient
            g_for_step = surrogate.predict_grad(y)

        # Schedule-Free Adam update logic (uses g_for_step from either source)
        m = beta1 * m + (1 - beta1) * g_for_step
        v = beta2 * v + (1 - beta2) * (g_for_step**2)

        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        update = m_hat / (np.sqrt(v_hat) + eps)
        z -= lr_t * update

        c = 1 / (t + 1)
        x = (1 - c) * x + c * z

        path.append(x.copy())
    return np.array(path)


def surrogate_schedule_free_adabelief_optimizer(  # noqa: PLR0913
    x0: Array1D,
    lr: float = 0.1,
    steps: int = 40000,
    warmup_steps: int = 100,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
    refit_every: int = 20,
    beta1: float = 0.9,
    beta2: float = 0.999,
    beta_sf: float = 0.9,
    eps: float = 1e-8,
    surrogate_type: str = "kernel_ridge",
) -> Array2D:
    """Optimizer using a surrogate model and a Schedule-Free AdaBelief rule."""
    if rng is None:
        rng = np.random.default_rng()
    z = x0.copy()
    x = x0.copy()
    path = [x.copy()]
    if surrogate_type == "kernel_ridge":
        surrogate: BaseGradSurrogate = KernelRidgeSurrogate(d=x0.size)
    elif surrogate_type == "rff":
        surrogate = RFFGradSurrogate(d=x0.size, rng=rng)
    else:
        msg = f"Unknown surrogate type: {surrogate_type}"
        raise ValueError(msg)

    # AdaBelief state variables
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for t in range(1, steps + 1):
        lr_t = lr * t / warmup_steps if t < warmup_steps else lr
        y = (1 - beta_sf) * z + beta_sf * x

        # Decide whether to use the true gradient or the surrogate.
        is_model_fitted = (
            hasattr(surrogate, "alpha") and surrogate.alpha is not None
        ) or (hasattr(surrogate, "W") and surrogate.W is not None)
        if not is_model_fitted or t % refit_every == 0:
            # Expensive step: calculate the true gradient
            g_for_step = grad_rosenbrock(y, noise_scale=noise_scale, rng=rng)
            # Store this new ground-truth data point and refit
            surrogate.fit(y.copy(), g_for_step.copy())
        else:
            # Cheap step: use the surrogate model to predict the gradient
            g_for_step = surrogate.predict_grad(y)

        # Schedule-Free AdaBelief update logic
        m = beta1 * m + (1 - beta1) * g_for_step
        s = g_for_step - m
        v = beta2 * v + (1 - beta2) * (s**2) + eps
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        update = m_hat / (np.sqrt(v_hat) + eps)
        z -= lr_t * update
        c = 1 / (t + 1)
        x = (1 - c) * x + c * z
        path.append(x.copy())
    return np.array(path)


def surrogate_schedule_free_ademamix_optimizer(  # noqa: PLR0913
    x0: Array1D,
    lr: float = 0.1,
    steps: int = 40000,
    warmup_steps: int = 100,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
    refit_every: int = 20,
    beta1: float = 0.9,
    beta2: float = 0.999,
    beta3: float = 0.9999,
    alpha: float = 5.0,
    beta_sf: float = 0.9,
    eps: float = 1e-8,
    surrogate_type: str = "kernel_ridge",
) -> Array2D:
    """Optimizer using a surrogate gradient model and a Schedule-Free AdEMAMix rule."""
    if rng is None:
        rng = np.random.default_rng()
    z = x0.copy()
    x = x0.copy()
    path = [x.copy()]
    if surrogate_type == "kernel_ridge":
        surrogate: BaseGradSurrogate = KernelRidgeSurrogate(d=x0.size)
    elif surrogate_type == "rff":
        surrogate = RFFGradSurrogate(d=x0.size, rng=rng)
    else:
        msg = f"Unknown surrogate type: {surrogate_type}"
        raise ValueError(msg)

    # AdEMAMix state variables
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    v = np.zeros_like(x)

    for t in range(1, steps + 1):
        lr_t = lr * t / warmup_steps if t < warmup_steps else lr
        y = (1 - beta_sf) * z + beta_sf * x

        # Decide whether to use the true gradient or the surrogate.
        is_model_fitted = (
            hasattr(surrogate, "alpha") and surrogate.alpha is not None
        ) or (hasattr(surrogate, "W") and surrogate.W is not None)
        if not is_model_fitted or t % refit_every == 0:
            # Expensive step: calculate the true gradient
            g_for_step = grad_rosenbrock(y, noise_scale=noise_scale, rng=rng)
            # Store this new ground-truth data point and refit
            surrogate.fit(y.copy(), g_for_step.copy())
        else:
            # Cheap step: use the surrogate model to predict the gradient
            g_for_step = surrogate.predict_grad(y)

        # Schedule-Free AdEMAMix update logic
        m1 = beta1 * m1 + (1 - beta1) * g_for_step
        m2 = beta3 * m2 + (1 - beta3) * g_for_step
        v = beta2 * v + (1 - beta2) * (g_for_step**2)

        m1_hat = m1 / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        update_numerator = m1_hat + alpha * m2
        update_denominator = np.sqrt(v_hat) + eps
        update = update_numerator / update_denominator

        z -= lr_t * update
        c = 1 / (t + 1)
        x = (1 - c) * x + c * z
        path.append(x.copy())
    return np.array(path)


# ==============================================================================
# === SPSA (for reference, not used in main comparison) ========================
# ==============================================================================


def spsa_grad_rosenbrock(
    rng: np.random.Generator,
    x: Array1D,
    c: float = 1e-2,
    noise_std: float = 1e-3,
) -> Array1D:
    """Estimate Rosenbrock gradient with SPSA."""
    d = x.size
    delta = rng.choice([1.0, -1.0], size=d)
    y_plus = rosenbrock(x + c * delta)
    y_minus = rosenbrock(x - c * delta)
    g_hat = (y_plus - y_minus) / (2 * c) * (1 / delta)
    # add extra observation noise
    return g_hat + noise_std * rng.standard_normal(d)


# ==============================================================================
# === MAIN EXECUTION ===========================================================
# ==============================================================================

# --- Hyperparameters ---
x0 = np.array([-1.5, 1015.0])
rng = np.random.default_rng()
steps = 50000
noise_scale = 0.1
lr = 0.2  # Default learning rate
schedule_free_lr = 2 * lr
surrogate_lr = lr * 0.01  # Special learning rate for surrogate models
surrogate_sf_lr = lr * 0.1
surrogate_type = "kernel_ridge"  # Options: "kernel_ridge", "rff"

# --- Run Optimizers ---
logger.info("Running standard optimizers...")
adam_path = adam(x0, lr=lr, steps=steps, noise_scale=noise_scale, rng=rng)
adabelief_path = adabelief(x0, lr=lr, steps=steps, noise_scale=noise_scale, rng=rng)
ademamix_path = ademamix(x0, lr=lr, steps=steps, noise_scale=noise_scale, rng=rng)

logger.info("Running schedule-free optimizers...")
sfsgd_path = schedule_free_sgd(
    x0,
    lr=lr * 0.1,
    steps=steps,
    noise_scale=noise_scale,
    rng=rng,
)
sfadam_path = schedule_free_adam(
    x0,
    lr=schedule_free_lr,
    steps=steps,
    noise_scale=noise_scale,
    rng=rng,
)
sfadabelief_path = schedule_free_adabelief(
    x0,
    lr=schedule_free_lr,
    steps=steps,
    noise_scale=noise_scale,
    rng=rng,
)
sfademamix_path = schedule_free_ademamix(
    x0,
    lr=schedule_free_lr,
    steps=steps,
    noise_scale=noise_scale,
    rng=rng,
)

logger.info("Running surrogate optimizers...")
surrogate_adam_path = surrogate_adam_optimizer(
    x0,
    lr=surrogate_lr,
    steps=steps,
    noise_scale=noise_scale,
    rng=rng,
    surrogate_type=surrogate_type,
)
surrogate_adabelief_path = surrogate_adabelief_optimizer(
    x0,
    lr=surrogate_lr,
    steps=steps,
    noise_scale=noise_scale,
    rng=rng,
    surrogate_type=surrogate_type,
)
surrogate_ademamix_path = surrogate_ademamix_optimizer(
    x0,
    lr=surrogate_lr,
    steps=steps,
    noise_scale=noise_scale,
    rng=rng,
    surrogate_type=surrogate_type,
)


logger.info("Running surrogate schedule-free optimizers...")
(surrogate_sf_adam_path) = surrogate_schedule_free_adam_optimizer(
    x0,
    lr=surrogate_sf_lr,
    steps=steps,
    noise_scale=noise_scale,
    rng=rng,
    surrogate_type=surrogate_type,
)
(surrogate_sf_adabelief_path) = surrogate_schedule_free_adabelief_optimizer(
    x0,
    lr=surrogate_sf_lr,
    steps=steps,
    noise_scale=noise_scale,
    rng=rng,
    surrogate_type=surrogate_type,
)
(surrogate_sf_ademamix_path) = surrogate_schedule_free_ademamix_optimizer(
    x0,
    lr=surrogate_sf_lr,
    steps=steps,
    noise_scale=noise_scale,
    rng=rng,
    surrogate_type=surrogate_type,
)


# --- Results and Plotting ---
minimum = np.array([1.0, 1000.0])
optimizers = {
    "Adam": adam_path,
    "AdaBelief": adabelief_path,
    "AdEMAMix": ademamix_path,
    "Schedule-Free SGD": sfsgd_path,
    "Schedule-Free Adam": sfadam_path,
    "Schedule-Free AdaBelief": sfadabelief_path,
    "Schedule-Free AdEMAMix": sfademamix_path,
    "Surrogate Adam": surrogate_adam_path,
    "Surrogate AdaBelief": surrogate_adabelief_path,
    "Surrogate AdEMAMix": surrogate_ademamix_path,
    "Surrogate SF-Adam": surrogate_sf_adam_path,
    "Surrogate SF-AdaBelief": surrogate_sf_adabelief_path,
    "Surrogate SF-AdEMAMix": surrogate_sf_ademamix_path,
}

# Log results in a formatted table
logger.info("\n%-45s %s", "Optimizer", "End Distance")
logger.info("-" * 65)
for name, path in optimizers.items():
    dist = np.linalg.norm(path[-1] - minimum)
    logger.info("%-45s %.4g", name, dist)


# Plotting
plt.figure(figsize=(10, 6))

# Dynamically determine plot bounds from all paths
all_paths = np.vstack(list(optimizers.values()))
path_x_min, path_y_min = all_paths.min(axis=0)
path_x_max, path_y_max = all_paths.max(axis=0)

# Ensure the plot bounds include the minimum
x_min = min(path_x_min, minimum[0])
y_min = min(path_y_min, minimum[1])
x_max = max(path_x_max, minimum[0])
y_max = max(path_y_max, minimum[1])

x_margin = (x_max - x_min) * 0.1
y_margin = (y_max - y_min) * 0.1

X, Y = np.meshgrid(
    np.linspace(x_min - x_margin, x_max + x_margin, 400),
    np.linspace(y_min - y_margin, y_max + y_margin, 400),
)
Z = rosenbrock(np.array([X, Y]))
plt.contourf(X, Y, Z, levels=np.logspace(0, 7, 20), cmap="jet", alpha=0.1)
for name, path in optimizers.items():
    (line,) = plt.plot(path[:, 0], path[:, 1], label=name)
    # Add a circle for the final position
    plt.scatter(
        path[-1, 0],
        path[-1, 1],
        facecolors="none",
        edgecolors=line.get_color(),
        s=100,
        linewidths=1.5,
    )
plt.scatter(
    [minimum[0]],
    [minimum[1]],
    c="red",
    label="Minimum",
    zorder=5,
    marker="x",
    s=100,
)
plt.legend()
plt.title("Rosenbrock Optimization Path")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(visible=True)

# Plot distance from minimum vs iteration
plt.figure(figsize=(10, 6))
for name, path in optimizers.items():
    distances = np.linalg.norm(path - minimum, axis=1)
    plt.plot(distances, label=name)
plt.xlabel("Iteration")
plt.ylabel("Distance from Minimum")
plt.yscale("log")
plt.title("Distance to Minimum vs. Iteration")
plt.legend()
plt.grid(visible=True, which="both", ls="--")

# Plot function value vs iteration
plt.figure(figsize=(10, 6))
for name, path in optimizers.items():
    # For schedule-free, plot the value at the evaluation point y
    # For standard optimizers, plot the value at the iterate x
    values = [rosenbrock(p) for p in path]
    plt.plot(values, label=name)
if noise_scale > 0:
    plt.axhline(
        y=noise_scale,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Noise Level ({noise_scale:.2g})",
    )
plt.xlabel("Iteration")
plt.ylabel("Function Value (log scale)")
plt.yscale("log")
plt.title("Function Value vs. Iteration")
plt.legend()
plt.grid(visible=True, which="both", ls="--")

if plt.get_fignums():
    logger.info("\nPress Enter to close plots and continue...")
    plt.show(block=False)
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        logger.info("Skipping wait.")
    plt.close("all")
