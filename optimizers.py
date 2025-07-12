#!/usr/bin/env python3
"""Compare Optimizers.

This script implements and compares various optimization algorithms and schedulers.

Optimizers:
- SPSA (Simultaneous Perturbation Stochastic Approximation)
  Paper: https://www.jhuapl.edu/spsa/PDF-SPSA/Spall_An_Overview.pdf
- Adam (Adaptive Moment Estimation)
  Paper: https://arxiv.org/abs/1412.6980
- AMSGrad (Variant of Adam with improved convergence)
  Paper: https://openreview.net/forum?id=ryQu7f-RZ
- AdaBelief (Improved Adam optimizer)
  Paper: https://arxiv.org/abs/2010.07468
- AdamW/AdamWR (Adam with Decoupled Weight Decay and Warm Restarts)
  Paper: https://arxiv.org/abs/1711.05101
- Radam (Rectified Adam)
  Paper: https://arxiv.org/abs/1908.03265
- Schedule-Free SGD non convex optimization
  Paper: https://arxiv.org/pdf/2411.07061


Schedulers:
- SPSA (Power Law Decay scheduler)
  Paper: https://www.jhuapl.edu/spsa/PDF-SPSA/Spall_An_Overview.pdf
- Cosine Annealing with Warm Restarts
  Paper: https://arxiv.org/abs/1608.03983
- Schedule-Free (Schedule-Free Learning)
  Paper: https://arxiv.org/abs/2405.15682
"""

import logging
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, NonNegativeFloat, PositiveFloat, ValidationError
from pydantic import Field as PydanticField

# --- type aliases ---
Array1D = npt.NDArray[np.float64]
Array2D = npt.NDArray[np.float64]
Metrics = tuple[float, float]
ObjectiveFuncType = Callable[[Array1D], float]


@dataclass
class OptimizerResult:
    """Dataclass to hold the results of an optimization run."""

    path: Array2D
    objective_history: Array1D
    update_history: Array2D
    m_hat_history: Array2D | None = None
    v_hat_history: Array2D | None = None
    correction_history: Array2D | None = None


# --- logging setup ---
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# --- Pydantic Models for Configuration ---
PYDANTIC_CONFIG = {
    "arbitrary_types_allowed": True,
    "frozen": True,
}


class OptimizerConfig(BaseModel):
    """Configuration for the optimization process."""

    n_steps: int = PydanticField(..., gt=0)
    rng: np.random.Generator = PydanticField(default_factory=np.random.default_rng)

    model_config = PYDANTIC_CONFIG


class Hyperparameters(BaseModel):
    """Hyperparameters for the optimizers."""

    learning_rate: PositiveFloat
    beta1: float = PydanticField(..., ge=0.0, lt=1.0)
    beta2: float = PydanticField(..., ge=0.0, lt=1.0)
    epsilon: PositiveFloat
    noise_scale: NonNegativeFloat
    spsa_c: PositiveFloat | Array1D
    spsa_a: PositiveFloat
    spsa_capital_a: NonNegativeFloat = PydanticField(..., alias="spsa_A")
    spsa_alpha: PositiveFloat
    spsa_gamma: PositiveFloat
    # AdamWR specific
    weight_decay: NonNegativeFloat = 0.0
    cosine_annealing_t_max: int = PydanticField(100, gt=0)
    cosine_annealing_eta_min: NonNegativeFloat = 0.0
    # AMSGrad specific
    amsgrad_warmup_steps: int = PydanticField(100, gt=0)
    # WarmupCosineScheduler specific
    warmup_steps: int = PydanticField(100, gt=0)
    # ScheduleFreeSDG specific
    schedule_free_beta: float = PydanticField(0.9, ge=0.0, lt=1.0)

    model_config = PYDANTIC_CONFIG


class ObjectiveFunction(BaseModel):
    """Represents the function to be optimized."""

    name: str
    func: ObjectiveFuncType
    minimum: Array1D
    initial_position: Array1D
    hyperparams: Hyperparameters
    n_steps: int

    model_config = PYDANTIC_CONFIG


# --- 10D to 2D Projection ---
# Define projection matrices A and B (5x5)
# Using diagonal matrices for simplicity
A_matrix = np.diag([1.0, 0.5, 0.2, 0.1, 0.05])
B_matrix = np.diag([1.0, 0.5, 0.2, 0.1, 0.05])


def project_10d_to_2d_wrapper(
    func_2d: ObjectiveFuncType,
    offset_2d: Array1D,
) -> ObjectiveFuncType:
    """Wrap a 2D objective function to accept 10D input via projection.

    The 10D vector is split into two 5D vectors, xi and yi.
    The 2D point (x, y) is computed as:
    x = xi.T @ A @ xi
    y = yi.T @ B @ yi
    """

    def projected_func(z_10d: Array1D) -> float:
        if len(z_10d) != 10:
            msg = "Input vector for projected function must be 10-dimensional."
            raise ValueError(msg)

        xi = z_10d[:5]
        yi = z_10d[5:]

        x_proj = xi.T @ A_matrix @ xi
        y_proj = yi.T @ B_matrix @ yi

        # The projected (x, y) is then used in the original 2D function.
        # An offset is added to match the original problem's minimum location.
        point_2d = np.array([x_proj, y_proj]) + offset_2d
        return func_2d(point_2d)

    return projected_func


# --- Objective Functions ---
def quadratic_function(x: Array1D) -> float:
    """Quadratic function with minimum at (1, 1000)."""
    minimum = np.array([1.0, 1000.0])
    scaling_factors = np.array([2.0, 1.0])
    return np.sum(scaling_factors * (x - minimum) ** 2)


def rosenbrock_function(x: Array1D) -> float:
    """Rosenbrock's banana function with minimum at (1, 1000)."""
    # The standard Rosenbrock function has a minimum at (1, 1).
    # We translate x to move the minimum to (1, 1000).
    # The translation for y is y = x_1 - 999, so that when x_1=1000, y=1.
    x_translated = np.array([x[0], x[1] - 999.0])
    return (1 - x_translated[0]) ** 2 + 100 * (
        x_translated[1] - x_translated[0] ** 2
    ) ** 2


def ackley_function(x: Array1D) -> float:
    """Ackley's function with minimum at (1, 1000)."""
    minimum = np.array([1.0, 1000.0])
    x_translated = x - minimum
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x_translated)
    sum1 = np.sum(x_translated**2)
    sum2 = np.sum(np.cos(c * x_translated))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)


# --- Core Logic ---
def noisy_function(
    x: Array1D,
    noise_scale: float,
    objective_func: ObjectiveFuncType,
    rng: np.random.Generator,
) -> float:
    """Add Gaussian noise to an objective function."""
    base_value = objective_func(x)
    return base_value + rng.normal(0.0, noise_scale) if noise_scale > 0 else base_value


def spsa_gradient(
    x: Array1D,
    c: Array1D,
    noise_scale: float,
    objective: ObjectiveFunction,
    rng: np.random.Generator,
) -> Array1D:
    """Estimate gradient with SPSA using a per-dimension perturbation size."""
    delta = rng.choice([-1.0, 1.0], size=x.shape)
    # Perturb each dimension according to its own c_i
    x_plus = x + c * delta
    x_minus = x - c * delta
    y_plus = noisy_function(x_plus, noise_scale, objective.func, rng)
    y_minus = noisy_function(x_minus, noise_scale, objective.func, rng)
    return (y_plus - y_minus) / (2 * c * delta)


# --- Schedulers ---
class Scheduler(Protocol):
    """Protocol for learning rate schedulers."""

    def get_learning_rate(self, t: int) -> float:
        """Get the learning rate for the current step."""
        ...


@dataclass(frozen=True)
class ConstantScheduler(Scheduler):
    """Scheduler that returns a constant learning rate."""

    learning_rate: float

    def get_learning_rate(self, _t: int) -> float:
        """Get the learning rate for the current step."""
        return self.learning_rate


@dataclass(frozen=True)
class PowerLawDecayScheduler(Scheduler):
    """A scheduler that decays the learning rate using a power-law function."""

    a: float
    A: float
    alpha: float

    def get_learning_rate(self, t: int) -> float:
        """Get the learning rate for the current step."""
        k = t - 1
        return self.a / (k + 1 + self.A) ** self.alpha


@dataclass(frozen=True)
class CosineAnnealingScheduler(Scheduler):
    """Cosine annealing with warm restarts."""

    t_max: int
    eta_min: float
    eta_max: float

    def get_learning_rate(self, t: int) -> float:
        """Get the learning rate for the current step."""
        t_cur = t % self.t_max
        return self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (
            1 + np.cos(t_cur / self.t_max * np.pi)
        )


@dataclass(frozen=True)
class WarmupCosineScheduler(Scheduler):
    """Scheduler with a linear warmup followed by a cosine decay."""

    warmup_steps: int
    t_max: int  # Total steps for the entire schedule
    eta_max: float
    eta_min: float = 0.0

    def get_learning_rate(self, t: int) -> float:
        """Get the learning rate for the current step."""
        if t < self.warmup_steps:
            # Linear warmup
            return self.eta_max * t / self.warmup_steps

        # Cosine decay
        t_cur = t - self.warmup_steps
        t_max_decay = self.t_max - self.warmup_steps
        if t_max_decay <= 0:
            return self.eta_min

        return self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (
            1 + np.cos(t_cur / t_max_decay * np.pi)
        )


# --- Optimizers ---
@dataclass
class OptimizerState:
    """Holds the mutable state for an optimizer."""

    path: Array2D
    objective_history: Array1D
    update_history: Array2D
    m_hat_history: Array2D | None = None
    v_hat_history: Array2D | None = None
    correction_history: Array2D | None = None
    # Adaptive-specific state
    m: Array1D | None = None
    v: Array1D | None = None
    v_hat_max: Array1D | None = None
    # ScheduleFreeSDG specific state
    z: Array1D | None = None
    x: Array1D | None = None


class Optimizer(ABC):
    """Abstract base class for optimizers."""

    def __init__(
        self,
        params: Hyperparameters,
        scheduler: Scheduler,
        objective: ObjectiveFunction,
        cfg: OptimizerConfig,
    ) -> None:
        """Initialize the optimizer."""
        self.params = params
        self.scheduler = scheduler
        self.objective = objective
        self.cfg = cfg
        self.state = self._initialize_state(objective, cfg)

    def _initialize_state(
        self,
        objective: ObjectiveFunction,
        cfg: OptimizerConfig,
    ) -> OptimizerState:
        """Initialize the optimizer's state."""
        dim = len(objective.initial_position)
        return OptimizerState(
            path=np.zeros((cfg.n_steps, dim)),
            objective_history=np.zeros(cfg.n_steps),
            update_history=np.zeros((cfg.n_steps, dim)),
        )

    def optimize(self) -> OptimizerResult:
        """Run the full optimization process."""
        self.state.path[0] = self.objective.initial_position
        self.state.objective_history[0] = noisy_function(
            self.objective.initial_position,
            0.0,
            self.objective.func,
            self.cfg.rng,
        )
        self.state.update_history[0] = np.zeros_like(self.objective.initial_position)

        for t in range(1, self.cfg.n_steps):
            self.step(t)

        return OptimizerResult(
            path=self.state.path,
            objective_history=self.state.objective_history,
            update_history=self.state.update_history,
            m_hat_history=self.state.m_hat_history,
            v_hat_history=self.state.v_hat_history,
            correction_history=self.state.correction_history,
        )

    @abstractmethod
    def step(self, t: int) -> None:
        """Perform a single optimization step."""
        ...

    def _get_spsa_params(self, t: int) -> tuple[Array1D, float]:
        """Get SPSA parameters for the current step."""
        k = t - 1
        c_k = self.params.spsa_c / (k + 1) ** self.params.spsa_gamma
        a_k = self.scheduler.get_learning_rate(t)
        return c_k, a_k


class SPSA(Optimizer):
    """Standard SPSA optimizer."""

    def step(self, t: int) -> None:
        """Perform a single optimization step."""
        path_t_minus_1 = self.state.path[t - 1]
        c_k, a_k = self._get_spsa_params(t)

        grad = spsa_gradient(
            path_t_minus_1,
            c_k,
            self.params.noise_scale,
            self.objective,
            self.cfg.rng,
        )
        update_step = a_k * grad

        new_path = path_t_minus_1 - update_step
        self.state.path[t] = new_path
        self.state.update_history[t] = -update_step
        # Record the objective value of the *new* position
        self.state.objective_history[t] = noisy_function(
            new_path,
            0.0,  # Use zero noise for history tracking
            self.objective.func,
            self.cfg.rng,
        )


class MomentumOptimizer(Optimizer):
    """Gradient descent with momentum."""

    def _initialize_state(
        self,
        objective: ObjectiveFunction,
        cfg: OptimizerConfig,
    ) -> OptimizerState:
        """Initialize the optimizer's state."""
        state = super()._initialize_state(objective, cfg)
        dim = len(objective.initial_position)
        state.m = np.zeros(dim)
        return state

    def step(self, t: int) -> None:
        """Perform a single optimization step."""
        if self.state.m is None:
            msg = "Optimizer state not initialized correctly."
            raise ValueError(msg)

        path_t_minus_1 = self.state.path[t - 1]
        c_k, a_k = self._get_spsa_params(t)  # a_k is the learning rate

        grad = spsa_gradient(
            path_t_minus_1,
            c_k,
            self.params.noise_scale,
            self.objective,
            self.cfg.rng,
        )

        # Momentum update (using biased momentum)
        self.state.m = self.params.beta1 * self.state.m + (1 - self.params.beta1) * grad
        update_step = a_k * self.state.m

        new_path = path_t_minus_1 - update_step
        self.state.path[t] = new_path
        self.state.update_history[t] = -update_step
        self.state.objective_history[t] = noisy_function(
            new_path,
            0.0,  # Use zero noise for history tracking
            self.objective.func,
            self.cfg.rng,
        )


class AdaptiveOptimizer(Optimizer, ABC):
    """Abstract base class for adaptive optimizers like Adam."""

    use_adabelief: bool = False

    def _initialize_state(
        self,
        objective: ObjectiveFunction,
        cfg: OptimizerConfig,
    ) -> OptimizerState:
        """Initialize the optimizer's state."""
        state = super()._initialize_state(objective, cfg)
        dim = len(objective.initial_position)
        state.m = np.zeros(dim)
        state.v = np.zeros(dim)
        state.v_hat_max = np.zeros(dim)
        state.m_hat_history = np.zeros((cfg.n_steps, dim))
        state.v_hat_history = np.zeros((cfg.n_steps, dim))
        state.correction_history = np.zeros((cfg.n_steps, dim))
        return state

    def step(self, t: int) -> None:
        """Perform a single optimization step."""
        path_t_minus_1 = self.state.path[t - 1]
        c_k, a_k = self._get_spsa_params(t)

        grad = spsa_gradient(
            path_t_minus_1,
            c_k,
            self.params.noise_scale,
            self.objective,
            self.cfg.rng,
        )

        update_step, m_hat, v_hat, correction = self._calculate_update(
            grad,
            a_k,
            t,
        )

        new_path = path_t_minus_1 - update_step
        self.state.path[t] = new_path
        self.state.update_history[t] = -update_step
        self.state.objective_history[t] = noisy_function(
            new_path,
            0.0,  # Use zero noise for history tracking
            self.objective.func,
            self.cfg.rng,
        )
        if self.state.m_hat_history is not None:
            self.state.m_hat_history[t] = m_hat
        if self.state.v_hat_history is not None:
            self.state.v_hat_history[t] = v_hat
        if self.state.correction_history is not None:
            self.state.correction_history[t] = correction

    def _get_momentum_for_update(self, m_hat: Array1D, _t: int) -> Array1D:
        """Get the momentum term for the update. Can be overridden by subclasses."""
        return m_hat

    def _calculate_update(
        self,
        grad: Array1D,
        a_k: float,
        t: int,
    ) -> tuple[Array1D, Array1D, Array1D, Array1D]:
        """Calculate the update step for adaptive optimizers."""
        # This check is for mypy to know that m and v are not None
        if self.state.m is None or self.state.v is None:
            msg = "Optimizer state not initialized correctly."
            raise ValueError(msg)

        self.state.m = self.params.beta1 * self.state.m + (1 - self.params.beta1) * grad
        if self.use_adabelief:
            s_t = grad - self.state.m
            # Add epsilon to the belief tracking for numerical stability
            self.state.v = self.params.beta2 * self.state.v + (
                1 - self.params.beta2
            ) * (s_t**2 + self.params.epsilon)
        else:
            self.state.v = self.params.beta2 * self.state.v + (
                1 - self.params.beta2
            ) * (grad**2)

        m_hat = self.state.m / (1 - self.params.beta1**t)
        v_hat = self.state.v / (1 - self.params.beta2**t)

        momentum_for_update = self._get_momentum_for_update(m_hat, t)
        denominator, v_hat_for_history = self._get_denominator(v_hat, t)
        correction = momentum_for_update / denominator
        update_step = a_k * correction

        return update_step, m_hat, v_hat_for_history, correction

    @abstractmethod
    def _get_denominator(self, v_hat: Array1D, t: int) -> tuple[Array1D, Array1D]: ...


class Adam(AdaptiveOptimizer):
    """Adam optimizer."""

    def _get_denominator(self, v_hat: Array1D, _t: int) -> tuple[Array1D, Array1D]:
        denominator = np.sqrt(v_hat) + self.params.epsilon
        return denominator, v_hat


class AMSGrad(AdaptiveOptimizer):
    """AMSGrad optimizer."""

    def _get_denominator(self, v_hat: Array1D, t: int) -> tuple[Array1D, Array1D]:
        if self.state.v_hat_max is None:
            msg = "Optimizer state not initialized correctly."
            raise ValueError(msg)

        # Warm-up phase: behave like Adam for the first few steps
        if t < self.params.amsgrad_warmup_steps:
            denominator = np.sqrt(v_hat) + self.params.epsilon
            return denominator, v_hat

        # Standard AMSGrad logic after warm-up with decay
        self.state.v_hat_max = np.maximum(
            self.params.beta2 * self.state.v_hat_max,
            v_hat,
        )
        denominator = np.sqrt(self.state.v_hat_max) + self.params.epsilon
        return denominator, self.state.v_hat_max


class AdaBelief(AdaptiveOptimizer):
    """AdaBelief optimizer."""

    use_adabelief = True

    def _get_denominator(self, v_hat: Array1D, _t: int) -> tuple[Array1D, Array1D]:
        denominator = np.sqrt(v_hat) + self.params.epsilon
        return denominator, v_hat


class AdamW(Adam):
    """Adam with decoupled weight decay (AdamW)."""

    def _calculate_update(
        self,
        grad: Array1D,
        a_k: float,
        t: int,
    ) -> tuple[Array1D, Array1D, Array1D, Array1D]:
        """Calculate the update step for AdamW."""
        update_step, m_hat, v_hat, correction = super()._calculate_update(
            grad,
            a_k,
            t,
        )
        # Decoupled weight decay
        if self.params.weight_decay > 0:
            path_t_minus_1 = self.state.path[t - 1]
            update_step += self.params.weight_decay * a_k * path_t_minus_1

        return update_step, m_hat, v_hat, correction


class RAdam(AdaptiveOptimizer):
    """Rectified Adam (RAdam) optimizer."""

    RHO_THRESHOLD = 5

    def _calculate_update(
        self,
        grad: Array1D,
        a_k: float,
        t: int,
    ) -> tuple[Array1D, Array1D, Array1D, Array1D]:
        """Calculate the update step for RAdam."""
        if self.state.m is None or self.state.v is None:
            msg = "Optimizer state not initialized correctly."
            raise ValueError(msg)

        # Standard moment updates
        self.state.m = self.params.beta1 * self.state.m + (1 - self.params.beta1) * grad
        self.state.v = self.params.beta2 * self.state.v + (1 - self.params.beta2) * (
            grad**2
        )

        # Bias-corrected moments for history tracking
        m_hat = self.state.m / (1 - self.params.beta1**t)
        v_hat = self.state.v / (1 - self.params.beta2**t)

        # RAdam rectification logic
        beta2_t = self.params.beta2**t
        rho_max = 2 / (1 - self.params.beta2) - 1
        rho_t = rho_max - (2 * t * beta2_t) / (1 - beta2_t)

        if rho_t > self.RHO_THRESHOLD:  # Approximated SMA length is tractable
            # Rectification term
            r_t = np.sqrt(
                ((rho_t - 4) * (rho_t - 2) * rho_max)
                / ((rho_max - 4) * (rho_max - 2) * rho_t),
            )
            # Adaptive update
            # The update uses the biased momentum (self.state.m) and
            # bias-corrected variance (v_hat)
            correction = (r_t * self.state.m) / (np.sqrt(v_hat) + self.params.epsilon)
        else:
            # Fallback to SGD with momentum (using biased momentum)
            correction = self.state.m

        update_step = a_k * correction
        return update_step, m_hat, v_hat, correction

    def _get_denominator(self, v_hat: Array1D, _t: int) -> tuple[Array1D, Array1D]:
        """Bypass this method for RAdam's custom update logic."""
        return np.ones_like(v_hat), v_hat


class ScheduleFreeOptimizer(Optimizer, ABC):
    """Abstract base class for schedule-free optimizers."""

    def _initialize_state(
        self,
        objective: ObjectiveFunction,
        cfg: OptimizerConfig,
    ) -> OptimizerState:
        """Initialize the optimizer's state for schedule-free methods."""
        state = super()._initialize_state(objective, cfg)
        state.z = objective.initial_position.copy()
        state.x = objective.initial_position.copy()
        return state

    def step(self, t: int) -> None:
        """Perform a single optimization step using the schedule-free framework."""
        if self.state.z is None or self.state.x is None:
            msg = "Optimizer state not initialized correctly."
            raise ValueError(msg)

        c_k, _ = self._get_spsa_params(t)
        warmup_steps = self.params.warmup_steps
        z_t, x_t = self.state.z, self.state.x

        if t <= warmup_steps:
            grad = spsa_gradient(
                z_t,
                c_k,
                self.params.noise_scale,
                self.objective,
                self.cfg.rng,
            )
            update_direction = self._calculate_update_direction(grad, t)
            z_t_plus_1 = z_t - update_direction
            x_t_plus_1 = z_t_plus_1
        else:
            beta = self.params.schedule_free_beta
            y_t = (1 - beta) * z_t + beta * x_t
            grad = spsa_gradient(
                y_t,
                c_k,
                self.params.noise_scale,
                self.objective,
                self.cfg.rng,
            )
            update_direction = self._calculate_update_direction(grad, t)
            z_t_plus_1 = z_t - update_direction
            averaging_step = t - warmup_steps
            c_t_plus_1 = 1 / (averaging_step + 1)
            x_t_plus_1 = (1 - c_t_plus_1) * x_t + c_t_plus_1 * z_t_plus_1

        self.state.z, self.state.x = z_t_plus_1, x_t_plus_1
        self.state.path[t] = x_t_plus_1
        self.state.update_history[t] = x_t_plus_1 - x_t
        self.state.objective_history[t] = noisy_function(
            x_t_plus_1,
            0.0,
            self.objective.func,
            self.cfg.rng,
        )

    @abstractmethod
    def _calculate_update_direction(self, grad: Array1D, t: int) -> Array1D:
        """Calculate the update direction based on the gradient."""
        ...


class ScheduleFreeSDG(ScheduleFreeOptimizer):
    """Schedule-Free SDG optimizer."""

    def _calculate_update_direction(self, grad: Array1D, t: int) -> Array1D:
        """Calculate the update direction for SDG."""
        gamma = self.scheduler.get_learning_rate(t)
        return gamma * grad


class ScheduleFreeAdam(ScheduleFreeOptimizer, Adam):
    """Schedule-Free Adam optimizer."""

    def _initialize_state(
        self,
        objective: ObjectiveFunction,
        cfg: OptimizerConfig,
    ) -> OptimizerState:
        """Initialize state for both ScheduleFree and Adam."""
        # SLF001: Call super() to correctly chain initializations
        # across multiple inheritance.
        state = super()._initialize_state(objective, cfg)
        # Adam-specific state is already initialized by the super() call
        # through the MRO, so we just need to ensure the schedule-free
        # specific state is also there.
        state.z = objective.initial_position.copy()
        state.x = objective.initial_position.copy()
        return state

    def _calculate_update_direction(self, grad: Array1D, t: int) -> Array1D:
        """Calculate the update direction using Adam's logic."""
        gamma = self.scheduler.get_learning_rate(t)
        update_step, _, _, _ = self._calculate_update(grad, gamma, t)
        return update_step


class ScheduleFreeAdaBelief(ScheduleFreeAdam):
    """Schedule-Free AdaBelief optimizer."""

    use_adabelief = True


class ScheduleFreeAMSGrad(ScheduleFreeOptimizer, AMSGrad):
    """Schedule-Free AMSGrad optimizer."""

    def _initialize_state(
        self,
        objective: ObjectiveFunction,
        cfg: OptimizerConfig,
    ) -> OptimizerState:
        """Initialize state for both ScheduleFree and AMSGrad."""
        state = super()._initialize_state(objective, cfg)
        state.z = objective.initial_position.copy()
        state.x = objective.initial_position.copy()
        return state

    def _calculate_update_direction(self, grad: Array1D, t: int) -> Array1D:
        """Calculate the update direction using AMSGrad's logic."""
        gamma = self.scheduler.get_learning_rate(t)
        update_step, _, _, _ = self._calculate_update(grad, gamma, t)
        return update_step


# --- Plotting and Metrics ---
def compute_metrics(path: Array2D, minimum: Array1D) -> Metrics:
    """Return (final distance from minimum, mean squared step length)."""
    disp = np.linalg.norm(path[-1] - minimum)
    msd = float(np.mean(np.sum(np.diff(path, axis=0) ** 2, axis=1)))
    return disp, msd


def plot_paths(
    paths: list[Array2D],
    labels: list[str],
    minimum: Array1D,
    title: str,
    *,
    objective_func: ObjectiveFuncType | None = None,
    alpha: float = 0.6,
) -> None:
    """Plot optimizer paths with optional function contours."""
    fig, ax = plt.subplots(figsize=(6, 6))
    valid_paths = [path for path in paths if np.all(np.isfinite(path))]
    dim = paths[0].shape[1] if valid_paths else 0

    # For 10D paths, we need to project them to 2D for plotting
    if dim == 10:
        # Project the 10D minimum to its 2D equivalent for plotting
        # The minimum of x = xi'A'xi is at xi=0. So the projected minimum is the offset.
        minimum_2d = np.array([1.0, 1000.0])

        projected_paths = []
        for path in valid_paths:
            projected_path = np.zeros((path.shape[0], 2))
            for i, z_10d in enumerate(path):
                xi, yi = z_10d[:5], z_10d[5:]
                x_proj = xi.T @ A_matrix @ xi
                y_proj = yi.T @ B_matrix @ yi
                projected_path[i] = np.array([x_proj, y_proj]) + minimum_2d
            projected_paths.append(projected_path)
        paths_to_plot = projected_paths
        min_to_plot = minimum_2d
    else:
        paths_to_plot = valid_paths
        min_to_plot = minimum

    # Plot the function contours and color background
    if objective_func is not None and dim == 2:
        x_min, y_min = np.min([p.min(axis=0) for p in paths_to_plot], axis=0)
        x_max, y_max = np.max([p.max(axis=0) for p in paths_to_plot], axis=0)
        x_margin, y_margin = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1
        x_min, x_max = x_min - x_margin, x_max + x_margin
        y_min, y_max = y_min - y_margin, y_max + y_margin

        x_grid_vals = np.linspace(x_min, x_max, 100)
        y_grid_vals = np.linspace(y_min, y_max, 100)
        x_grid, y_grid = np.meshgrid(x_grid_vals, y_grid_vals)
        z_grid = np.array(
            [
                objective_func(np.array([xi, yi]))
                for xi, yi in zip(x_grid.ravel(), y_grid.ravel(), strict=False)
            ],
        ).reshape(x_grid.shape)

        ax.contourf(x_grid, y_grid, z_grid, levels=50, cmap="viridis", alpha=0.1)

    for i, path in enumerate(paths_to_plot):
        ax.plot(*path.T, label=labels[i], alpha=alpha, lw=1, color=f"C{i}")
        ax.scatter(
            *path[-1],
            s=80,
            facecolors="none",
            edgecolors=f"C{i}",
            linewidths=1.5,
        )

    ax.scatter(*min_to_plot, marker="x", color="red", s=100, label="Minimum")

    if paths_to_plot:
        all_points = np.vstack([*paths_to_plot, min_to_plot.reshape(1, -1)])
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        x_margin, y_margin = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    ax.set_aspect("equal", "box")
    ax.grid(visible=True)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()


def plot_performance(
    results: dict[str, dict[str, Any]],
    objective: ObjectiveFunction,
    noise_level: float,
    title: str,
) -> None:
    """Plot objective function value and distance from minimum vs. iteration."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(title)

    # Plot 1: Objective Function Value vs. Iteration
    for name, res in results.items():
        ax1.plot(
            np.arange(len(res["objective_history"])),
            res["objective_history"],
            label=name,
            alpha=0.8,
            lw=2,
        )
    ax1.axhline(
        y=noise_level,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Noise Level ({noise_level:.2g})",
    )
    ax1.set(
        ylabel="Objective Function Value (log scale)",
        yscale="log",
    )
    ax1.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    ax1.legend()

    # Plot 2: Distance from Minimum vs. Iteration
    for name, res in results.items():
        path = res["path"]
        distances = np.linalg.norm(path - objective.minimum, axis=1)
        ax2.plot(
            np.arange(len(distances)),
            distances,
            label=name,
            alpha=0.8,
            lw=2,
        )
    ax2.set(
        xlabel="Iteration",
        ylabel="Distance from Minimum (log scale)",
        yscale="log",
    )
    ax2.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.97])


def plot_history(
    histories: list[Array2D | None],
    labels: list[str],
    title: str,
    yscale: str = "log",
) -> None:
    """Plot history of a variable vs. iteration."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    fig.suptitle(title)

    for i, hist in enumerate(histories):
        if hist is None:
            continue
        iterations = np.arange(len(hist))
        for dim in range(hist.shape[1]):
            for ax in axes:
                ax.plot(
                    iterations,
                    hist[:, dim],
                    label=f"{labels[i]} (dim {dim + 1})",
                    alpha=0.8,
                    lw=1,
                )

    axes[0].set(title="Linear Scale", ylabel="Value")
    axes[1].set(
        title=f"Log Scale{' (Symmetric)' if yscale == 'symlog' else ''}",
        xlabel="Iteration",
        ylabel=f"Value ({yscale} scale)",
        yscale=yscale,
    )

    for ax in axes:
        ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
        ax.legend()

    plt.tight_layout()


# --- Experiment Execution ---
def _estimate_spsa_c_vector(
    objective: ObjectiveFunction,
    noise_level: float,
    rng: np.random.Generator,
) -> Array1D:
    """Estimate a good per-dimension spsa_c by finding a perturbation.

    This finds a perturbation that causes a function value change
    significantly larger than the noise level.
    """
    logger.info("Estimating optimal spsa_c vector...")
    theta = objective.minimum
    spsa_c_vector = np.zeros_like(theta)

    # Target a function value change that is higher than the noise level
    target_delta_f = 20.0 * noise_level

    c_upper_bound = 100.0
    min_perturbation = 1e-10

    # Estimate f_minimum as the average of 10 measurements
    f_minimum = np.mean(
        [noisy_function(theta, noise_level, objective.func, rng) for _ in range(10)],
    )

    for i in range(len(theta)):
        low_c = 1e-6
        high_c = c_upper_bound  # A reasonable upper bound for perturbation size

        # Binary search for a c_i that produces the target change in f
        for _ in range(100):
            c_i = (low_c + high_c) / 2.0
            if c_i < min_perturbation:
                break

            perturb_vec = np.zeros_like(theta)
            perturb_vec[i] = c_i

            f_plus = np.mean(
                [
                    noisy_function(
                        theta + perturb_vec,
                        noise_level,
                        objective.func,
                        rng,
                    )
                    for _ in range(10)
                ],
            )

            # Compute delta_f as the absolute difference: |f_plus - f_initial|
            delta_f = abs(f_plus - f_minimum)

            if delta_f < target_delta_f:
                # Change was too small, need a larger perturbation
                low_c = c_i
            else:
                # Change was large enough, try a smaller perturbation
                high_c = c_i

        # After the search, high_c is the smallest value found that met the target
        best_c_i = high_c
        if best_c_i >= c_upper_bound:
            logger.warning("SPSA c estimation for dim %d hit upper bound.", i)

        spsa_c_vector[i] = best_c_i

    return spsa_c_vector


def get_experiment_configs() -> list[ObjectiveFunction]:
    """Get all experiment configurations."""
    # Base hyperparameters from the original papers
    base_hyperparams = Hyperparameters(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        noise_scale=0.1,  # Placeholder, will be tuned
        spsa_c=0.1,  # Placeholder, will be tuned
        spsa_a=1.0,
        spsa_A=0,  # Placeholder, will be set per experiment
        spsa_alpha=0.602,
        spsa_gamma=0.101,
        # AdamWR defaults
        weight_decay=0.01,
        cosine_annealing_t_max=1,  # Placeholder, will be set per experiment
        cosine_annealing_eta_min=0.0,
        # ScheduleFreeSDG defaults
        schedule_free_beta=0.9,
    )

    # --- Experiment 1: Quadratic Function (2D) ---
    quadratic_n_steps = 1000
    quadratic_hyperparams = base_hyperparams.model_copy(
        update={
            "spsa_A": 0.1 * quadratic_n_steps,  # SPSA paper recommendation
            "learning_rate": 0.05,  # Tune LR
            "cosine_annealing_t_max": 1.0 * quadratic_n_steps,
            "warmup_steps": 0.1 * quadratic_n_steps,
        },
    )

    # --- Experiment 2: Rosenbrock Function (2D) ---
    rosenbrock_n_steps = 30000
    rosenbrock_hyperparams = base_hyperparams.model_copy(
        update={
            "spsa_A": 0.1 * rosenbrock_n_steps,  # SPSA paper recommendation
            "learning_rate": 0.01,  # Tune LR for this harder problem
            "cosine_annealing_t_max": 1.0 * rosenbrock_n_steps,
            "warmup_steps": 0.1 * rosenbrock_n_steps,
        },
    )

    # --- Experiment 3: Ackley Function (2D) ---
    ackley_n_steps = 10000
    ackley_hyperparams = base_hyperparams.model_copy(
        update={
            "spsa_A": 0.1 * ackley_n_steps,  # SPSA paper recommendation
            "learning_rate": 0.05,  # Tune LR for this harder problem
            "cosine_annealing_t_max": 1.0 * ackley_n_steps,
            "warmup_steps": 0.1 * ackley_n_steps,
        },
    )

    # --- Experiment 4: Quadratic Function (10D search space) ---
    quadratic_10d_n_steps = 2000
    quadratic_10d_hyperparams = base_hyperparams.model_copy(
        update={
            "spsa_A": 0.1 * quadratic_10d_n_steps,
            "learning_rate": 0.05,
            "cosine_annealing_t_max": 1.0 * quadratic_10d_n_steps,
            "warmup_steps": 0.1 * quadratic_10d_n_steps,
        },
    )
    # The minimum in the 10D space is at the origin.
    # The projection adds the offset to match the 2D problem's minimum.
    minimum_10d = np.zeros(10)
    # Start at a point that projects to approx [5.0, 1002.0] in 2D
    initial_pos_10d = np.array(
        [2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    )
    quadratic_10d_func = project_10d_to_2d_wrapper(
        quadratic_function,
        offset_2d=np.array([1.0, 1000.0]),
    )

    return [
        ObjectiveFunction(
            name="Quadratic",
            func=quadratic_function,
            minimum=np.array([1.0, 1000.0]),
            initial_position=np.array([5.0, 1002.0]),
            hyperparams=quadratic_hyperparams,
            n_steps=quadratic_n_steps,
        ),
        ObjectiveFunction(
            name="Rosenbrock",
            func=rosenbrock_function,
            minimum=np.array([1.0, 1000.0]),
            initial_position=np.array([-1.5, 1001.5]),
            hyperparams=rosenbrock_hyperparams,
            n_steps=rosenbrock_n_steps,
        ),
        ObjectiveFunction(
            name="Ackley",
            func=ackley_function,
            minimum=np.array([1.0, 1000.0]),
            initial_position=np.array([5.0, 1002.0]),
            hyperparams=ackley_hyperparams,
            n_steps=ackley_n_steps,
        ),
        ObjectiveFunction(
            name="Quadratic10D_Projected",
            func=quadratic_10d_func,
            minimum=minimum_10d,
            initial_position=initial_pos_10d,
            hyperparams=quadratic_10d_hyperparams,
            n_steps=quadratic_10d_n_steps,
        ),
    ]


def get_optimizer_setups(
    params: Hyperparameters,
) -> dict[str, tuple[type[Optimizer], Scheduler]]:
    """Return a dictionary of optimizer and scheduler configurations."""
    power_law_decay_sched = PowerLawDecayScheduler(
        a=params.spsa_a,
        A=params.spsa_capital_a,
        alpha=params.spsa_alpha,
    )
    ConstantScheduler(learning_rate=params.learning_rate)
    # Higher learning rate for schedule-free optimizers
    const_sched_sf = ConstantScheduler(learning_rate=params.learning_rate * 5)
    cosine_scheduler = CosineAnnealingScheduler(
        t_max=params.cosine_annealing_t_max,
        eta_min=params.cosine_annealing_eta_min,
        eta_max=params.learning_rate,
    )
    WarmupCosineScheduler(
        warmup_steps=params.warmup_steps,
        t_max=params.cosine_annealing_t_max,  # Reusing t_max for total steps
        eta_max=params.learning_rate,
        eta_min=params.cosine_annealing_eta_min,
    )

    return {
        "SPSA-PowerLawDecay": (SPSA, power_law_decay_sched),
        # "Momentum-Constant": (MomentumOptimizer, const_sched),
        # "Momentum-PowerLawDecay": (MomentumOptimizer, power_law_decay_sched),
        "Momentum-CosineAnnealing": (MomentumOptimizer, cosine_scheduler),
        # "Adam-Constant": (Adam, const_sched),
        # "Adam-PowerLawDecay": (Adam, power_law_decay_sched),
        "Adam-CosineAnnealing": (Adam, cosine_scheduler),
        # "AMSGrad-Constant": (AMSGrad, const_sched),
        # "AMSGrad-PowerLawDecay": (AMSGrad, power_law_decay_sched),
        "AMSGrad-CosineAnnealing": (AMSGrad, cosine_scheduler),
        # "AMSGrad-WarmupCosine": (AMSGrad, warmup_cosine_sched),
        # "AdaBelief-Constant": (AdaBelief, const_sched),
        # "AdaBelief-PowerLawDecay": (AdaBelief, power_law_decay_sched),
        "AdaBelief-CosineAnnealing": (AdaBelief, cosine_scheduler),
        # "AdamW-Constant": (AdamW, const_sched),
        # "AdamW-CosineAnnealing": (AdamW, cosine_scheduler),
        # "AdamW-PowerLawDecay": (AdamW, power_law_decay_sched),
        # "RAdam-Constant": (RAdam, const_sched),
        # "ScheduleFreeSDG-Constant": (ScheduleFreeSDG, const_sched_sf),
        "ScheduleFreeAdam-Constant": (ScheduleFreeAdam, const_sched_sf),
        "ScheduleFreeAdaBelief-Constant": (ScheduleFreeAdaBelief, const_sched_sf),
        "ScheduleFreeAMSGrad-Constant": (ScheduleFreeAMSGrad, const_sched_sf),
    }


def run_experiment(
    objective: ObjectiveFunction,
    *,
    plot_results: bool = True,
) -> None:
    """Run a full optimization experiment for a given objective function."""
    logger.info("--- Optimizing %s Function ---", objective.name)
    opt_cfg = OptimizerConfig(n_steps=objective.n_steps)

    minimum_val = objective.func(objective.minimum)
    # Calculate a point slightly offset from the minimum position
    offset = np.full_like(objective.minimum, 0.001)
    offset_position = objective.minimum + offset
    offset_val = objective.func(offset_position)

    # Estimate the noise level at the minimum position
    noise_level = abs(offset_val - minimum_val) * 100

    spsa_c_vector = _estimate_spsa_c_vector(objective, noise_level, opt_cfg.rng)

    logger.info(
        "Noise Level: %.4g, Estimated SPSA c vector: %s",
        noise_level,
        np.array2string(spsa_c_vector, formatter={"float_kind": lambda x: f"{x:.4g}"}),
    )

    current_hyperparams = objective.hyperparams.model_copy(
        update={"noise_scale": noise_level, "spsa_c": spsa_c_vector},
    )

    optimizer_setups = get_optimizer_setups(current_hyperparams)

    results: dict[str, dict[str, Any]] = {}
    for name, (optimizer_class, scheduler) in optimizer_setups.items():
        if objective.name == "Rosenbrock" and name in [
            "SPSA-PowerLawDecay",
            "Momentum-Constant",
            "Momentum-PowerLawDecay",
            "Momentum-CosineAnnealing",
            "ScheduleFreeSDG-Constant",
            "ScheduleFreeSDG-PowerLawDecay",
        ]:
            logger.info("Skipping %s for %s due to instability.", name, objective.name)
            continue

        hyperparams_for_optimizer = current_hyperparams
        if "ScheduleFreeAdaBelief" in name:
            hyperparams_for_optimizer = current_hyperparams.model_copy(
                update={"epsilon": 1e-4},
            )

        optimizer = optimizer_class(
            hyperparams_for_optimizer,
            scheduler,
            objective,
            opt_cfg,
        )

        result = optimizer.optimize()
        dist, _ = compute_metrics(result.path, objective.minimum)
        results[name] = {
            "path": result.path,
            "objective_history": result.objective_history,
            "update_history": result.update_history,
            "dist": dist,
            "m_hat_hist": result.m_hat_history,
            "v_hat_hist": result.v_hat_history,
            "correction_hist": result.correction_history,
            "scheduler_name": scheduler.__class__.__name__,
        }

    # Log results in a formatted table
    logger.info("\n%-45s %s", "Optimizer-Scheduler", "End Distance")
    logger.info("-" * 65)
    for name, res in results.items():
        logger.info("%-45s %.4g", name, res["dist"])

    if plot_results:
        plot_experiment_results(results, objective, noise_level)


def plot_experiment_results(
    results: dict[str, dict[str, Any]],
    objective: ObjectiveFunction,
    noise_level: float,
) -> None:
    """Plot the results of an experiment."""
    title_suffix = f"({objective.name} Function)"
    labels = list(results.keys())

    plot_paths(
        [res["path"] for res in results.values()],
        labels,
        objective.minimum,
        title=f"Optimizer Path {title_suffix}",
        objective_func=objective.func,
    )
    plot_performance(
        results,
        objective,
        noise_level,
        title=f"Optimizer Performance {title_suffix}",
    )
    plot_history(
        [np.cumsum(res["update_history"], axis=0) for res in results.values()],
        labels,
        title=f"Cumulative Update {title_suffix}",
        yscale="symlog",
    )
    plot_labels = [
        name for name, res in results.items() if res["m_hat_hist"] is not None
    ]
    if plot_labels:
        plot_history(
            [results[name]["m_hat_hist"] for name in plot_labels],
            plot_labels,
            title=f"m_hat History {title_suffix}",
            yscale="symlog",
        )
        plot_history(
            [results[name]["v_hat_hist"] for name in plot_labels],
            plot_labels,
            title=f"v_hat History {title_suffix}",
            yscale="symlog",
        )
        plot_history(
            [results[name]["correction_hist"] for name in plot_labels],
            plot_labels,
            title=f"Adaptive Correction History {title_suffix}",
            yscale="symlog",
        )


def main() -> None:
    """Run the main entry point of the script."""
    experiments = get_experiment_configs()
    for experiment in experiments:
        run_experiment(experiment, plot_results=True)
        if plt.get_fignums():
            logger.info("\nPress Enter to close plots and continue...")
            plt.show(block=False)
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                logger.info("Skipping wait.")
            plt.close("all")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, ValidationError) as e:
        if isinstance(e, KeyboardInterrupt):
            logger.info("Interrupted by user, exiting.")
        else:
            logger.exception("Configuration error")
        sys.exit(1)
