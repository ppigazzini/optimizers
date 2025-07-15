"""SPSA simulation for a chess engine with multiple parameters.

This script simulates the optimization of a set of parameters using the
Simultaneous Perturbation Stochastic Approximation (SPSA) algorithm. A subset of
these parameters is defined as "Elo-sensitive," meaning they influence the
outcome probabilities of simulated games based on an Elo model.

The simulation features:
- A pentanomial distribution for game outcomes: LL, DL, DD, WD, WW
- SPSA optimization to maximize net wins.
- A comprehensive analysis suite, including:
  - PCA on parameter trajectories.
  - Run-length analysis of parameter updates.
  - Clustering to identify parameter behaviors.
  - Statistical tests for significance.
  - Convergence analysis.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel, Field
from scipy.stats import chi2_contingency
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# --- logging setup ---
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# --- Constants ---
PENTA_OUTCOMES: int = 5

# Constants for magic numbers
MIN_PCA_COMPONENTS = 2
MIN_CHI2_COLUMNS = 2


class SPSAConfig(BaseModel):
    """Configuration for SPSA optimization simulation.

    Attributes:
        num_pairs: Total number of game pairs to simulate.
        batch_size: Number of pairs per SPSA step.
        learning_rate: SPSA gradient ascent learning rate.
        c_k: Perturbation scale for finite differences.
        k_elo: Elo sensitivity scaling factor.
        seed: Random seed for reproducibility.
        num_params: Total number of parameters to optimize.
        num_elo_params: Number of Elo-sensitive parameters.
        penta_base: Base pentanomial outcome probabilities.
        penta_shift: Shifts applied based on Elo difference.
        penta_labels: Labels for pentanomial outcomes.
        early_stopping: Whether to use early stopping.
        patience: Patience for early stopping.
        convergence_threshold: Threshold for parameter convergence.
        analysis_n_clusters: Number of clusters for parameter analysis.

    """

    num_pairs: int = Field(1_000_000, description="Total number of game pairs")
    batch_size: int = Field(100, description="Batch size for each SPSA step")
    learning_rate: float = Field(0.01, description="SPSA learning rate")
    c_k: float = Field(1.0, description="SPSA perturbation scale")
    k_elo: float = Field(5.0, description="Elo sensitivity per unit theta[0]")
    seed: int | None = Field(None, description="Random seed")
    num_params: int = Field(100, description="Total number of parameters")
    num_elo_params: int = Field(25, description="Number of Elo-sensitive parameters")
    penta_base: list[float] = Field(
        default=[0.0175, 0.2225, 0.52, 0.2225, 0.0175],
        description="Base pentanomial probabilities",
    )
    penta_shift: list[float] = Field(
        default=[-0.04, -0.10, 0.0, 0.10, 0.04],
        description="Pentanomial probability shifts",
    )
    penta_labels: tuple[str, ...] = Field(
        ("LL", "DL", "DD", "WD", "WW"),
        description="Pentanomial outcome labels",
    )
    early_stopping: bool = Field(default=True, description="Enable early stopping")
    patience: int = Field(1000, description="Patience for early stopping")
    convergence_threshold: float = Field(
        0.01,
        description="Threshold for parameter convergence",
    )
    analysis_n_clusters: int = Field(
        3,
        description="Number of clusters for parameter analysis",
    )

    def __init__(self, **data: object) -> None:
        """Initialize the configuration and validate it."""
        super().__init__(**data)
        if (
            len(self.penta_base) != PENTA_OUTCOMES
            or len(self.penta_shift) != PENTA_OUTCOMES
        ):
            msg = f"penta_base and penta_shift must have {PENTA_OUTCOMES} elements."
            raise ValueError(msg)

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


@dataclass
class SPSAResult:
    """Container for SPSA optimization results."""

    config: SPSAConfig
    trajectory: np.ndarray
    cumulative_spsa_signal: np.ndarray
    final_params: np.ndarray
    convergence_metrics: dict[str, float | bool]
    elapsed_time: float


def elo_to_pentanomial_probs(elo: float, config: SPSAConfig) -> np.ndarray:
    """Return pentanomial probabilities as a function of Elo."""
    scale = np.clip(elo / 1000, -0.5, 0.5)
    probs = np.array(config.penta_base) + np.array(config.penta_shift) * scale
    probs = np.clip(probs, 0, 1)
    probs /= probs.sum()
    return probs


def simulate_batch(
    batch_size: int,
    probs: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, np.ndarray]:
    """Simulate a batch of pairs and return net wins and outcome counts.

    Outcome order: [LL, DL, DD, WD, WW]
    Net wins = (2*WW + WD) - (2*LL + DL).
    """
    counts = rng.multinomial(batch_size, probs)
    net_wins = (2 * counts[4] + counts[3]) - (2 * counts[0] + counts[1])
    return int(net_wins), counts


def spsa_optimization(config: SPSAConfig) -> SPSAResult:
    """Run SPSA optimization for parameters."""
    start_time = time.time()
    num_batches = config.num_pairs // config.batch_size
    theta = np.zeros(config.num_params, dtype=float)
    trajectory = [theta.copy()]
    spsa_signal_history = []
    rng = np.random.default_rng(config.seed)

    # Early stopping variables
    best_theta = theta.copy()
    best_score = -np.inf
    no_improvement_count = 0
    stopped_early = False

    for batch_idx in range(num_batches):
        delta = rng.choice([-1, 1], size=config.num_params)
        theta_plus = theta + config.c_k * delta
        theta_minus = theta - config.c_k * delta

        elo_plus = config.k_elo * np.sum(theta_plus[: config.num_elo_params])
        elo_minus = config.k_elo * np.sum(theta_minus[: config.num_elo_params])

        probs_plus = elo_to_pentanomial_probs(elo_plus, config)
        probs_minus = elo_to_pentanomial_probs(elo_minus, config)

        net_wins_plus, _ = simulate_batch(config.batch_size, probs_plus, rng)
        net_wins_minus, _ = simulate_batch(config.batch_size, probs_minus, rng)

        grad_est = (net_wins_plus - net_wins_minus) / (2 * config.c_k * delta)
        theta += config.learning_rate * grad_est

        trajectory.append(theta.copy())
        spsa_signal_history.append((net_wins_plus - net_wins_minus) / delta)

        if config.early_stopping and batch_idx > config.patience:
            # Use the sum of net wins over the patience window as the score
            recent_score = np.mean(
                [np.sum(s) for s in spsa_signal_history[-config.patience :]],
            )
            if recent_score > best_score:
                best_score = recent_score
                best_theta = theta.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= config.patience:
                logger.info("Early stopping at batch %d", batch_idx)
                stopped_early = True
                break

    elapsed_time = time.time() - start_time
    final_theta = best_theta if config.early_stopping and stopped_early else theta

    # Calculate convergence metrics
    final_distances = np.linalg.norm(
        np.diff(trajectory[-100:], axis=0),
        axis=1,
    )
    convergence_metrics = {
        "final_gradient_norm": np.mean(final_distances),
        "parameters_converged": np.sum(
            final_distances < config.convergence_threshold,
        ),
        "total_batches": len(trajectory) - 1,
        "early_stopped": stopped_early,
    }

    return SPSAResult(
        config=config,
        trajectory=np.array(trajectory),
        cumulative_spsa_signal=np.cumsum(spsa_signal_history, axis=0),
        final_params=final_theta,
        convergence_metrics=convergence_metrics,
        elapsed_time=elapsed_time,
    )


def plot_basic_results(result: SPSAResult) -> None:
    """Plot parameter trajectories and cumulative SPSA signal."""
    trajectory = result.trajectory
    cumulative_signal = result.cumulative_spsa_signal
    num_params = trajectory.shape[1]
    elo_idx = 0
    non_elo_idx = (
        result.config.num_elo_params
        if num_params > result.config.num_elo_params
        else None
    )

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("SPSA Optimization Basic Results", fontsize=16)

    # Plot Trajectory
    ax = axs[0]
    if non_elo_idx is not None:
        ax.plot(
            trajectory[:, elo_idx],
            trajectory[:, non_elo_idx],
            marker="o",
            markersize=2,
            linewidth=1,
        )
        ax.scatter(
            trajectory[0, elo_idx],
            trajectory[0, non_elo_idx],
            color="green",
            s=80,
            label="Start",
            zorder=5,
        )
        ax.scatter(
            trajectory[-1, elo_idx],
            trajectory[-1, non_elo_idx],
            color="red",
            s=80,
            label="End",
            zorder=5,
        )
        ax.set_title(
            f"Trajectory: Param {elo_idx} (Elo) vs {non_elo_idx} (non-Elo)",
        )
        ax.set_xlabel(f"Parameter {elo_idx} (Elo-sensitive)")
        ax.set_ylabel(f"Parameter {non_elo_idx} (non-Elo-sensitive)")
    else:
        ax.plot(trajectory[:, elo_idx], marker="o", markersize=2, linewidth=1)
        ax.set_title(f"Trajectory: Parameter {elo_idx} (Elo-sensitive)")
        ax.set_xlabel("Step")
        ax.set_ylabel(f"Parameter {elo_idx} Value")
    ax.legend()
    ax.grid(visible=True)

    # Plot Cumulative SPSA Signal
    ax = axs[1]
    final_elo_signal = cumulative_signal[-1, elo_idx]
    ax.plot(
        cumulative_signal[:, elo_idx],
        label=f"Param {elo_idx} (Elo) | Final: {final_elo_signal:.1f}",
        color="blue",
    )
    if non_elo_idx is not None and non_elo_idx < cumulative_signal.shape[1]:
        final_non_elo_signal = cumulative_signal[-1, non_elo_idx]
        ax.plot(
            cumulative_signal[:, non_elo_idx],
            label=f"Param {non_elo_idx} (non-Elo) | Final: {final_non_elo_signal:.1f}",
            color="orange",
        )
    ax.set_title("Cumulative SPSA Signal")
    ax.set_xlabel("Batch Number")
    ax.set_ylabel("Cumulative Signal Value")
    ax.legend()
    ax.grid(visible=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def analyze_and_plot_pca(trajectory: np.ndarray) -> None:
    """Analyze parameter trajectories using PCA and plot results."""
    logger.info("\n--- PCA Analysis ---")
    n_components = min(trajectory.shape[0], trajectory.shape[1], 20)
    if n_components < MIN_PCA_COMPONENTS:
        logger.warning("Not enough data for PCA analysis. Skipping.")
        return

    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(trajectory)
    explained = pca.explained_variance_ratio_

    logger.info(
        "Explained variance (first %d components): %s",
        len(explained),
        explained,
    )
    logger.info("Cumulative explained variance: %.3f", np.sum(explained))

    # Plot explained variance
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(1, len(explained) + 1), explained, color="royalblue")
    plt.title("PCA Explained Variance Ratio")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.tight_layout()
    plt.show()

    # Plot trajectory in PC space
    plt.figure(figsize=(6, 5))
    plt.plot(pcs[:, 0], pcs[:, 1], marker="o", markersize=2, linewidth=1)
    title = (
        "PCA of Parameter Trajectories\n"
        f"Explained: PC1={explained[0]:.2%}, PC2={explained[1]:.2%}"
    )
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(visible=True)
    plt.tight_layout()
    plt.show()


def _get_runs(signs: np.ndarray, sign_value: int) -> list[int]:
    """Extract run lengths for a specific sign."""
    runs = []
    current_run = 0
    for sign in signs:
        if sign == sign_value:
            current_run += 1
        elif current_run > 0:
            runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)
    return runs


def build_run_length_contingency_table(trajectory: np.ndarray) -> pd.DataFrame:
    """Build a contingency table of run lengths."""
    delta = np.diff(trajectory, axis=0)
    n_params = delta.shape[1]
    all_pos_runs, all_neg_runs = [], []
    max_run_pos, max_run_neg = 0, 0

    for i in range(n_params):
        signs = np.sign(delta[:, i])
        pos_runs = _get_runs(signs, 1)
        neg_runs = _get_runs(signs, -1)

        if pos_runs:
            max_run_pos = max(max_run_pos, *pos_runs)
        if neg_runs:
            max_run_neg = max(max_run_neg, *neg_runs)

        all_pos_runs.append(pos_runs)
        all_neg_runs.append(neg_runs)

    columns = [f"pos_{k}" for k in range(1, max_run_pos + 1)] + [
        f"neg_{k}" for k in range(1, max_run_neg + 1)
    ]
    table = np.zeros((n_params, len(columns)), dtype=int)

    for i in range(n_params):
        pos_counts = np.bincount(all_pos_runs[i], minlength=max_run_pos + 1)[1:]
        neg_counts = np.bincount(all_neg_runs[i], minlength=max_run_neg + 1)[1:]
        table[i, :max_run_pos] = pos_counts
        table[i, max_run_pos:] = neg_counts

    return pd.DataFrame(
        table,
        columns=columns,
        index=[f"param_{i}" for i in range(n_params)],
    )


def analyze_and_plot_run_lengths(trajectory: np.ndarray) -> None:
    """Perform chi-square analysis on run lengths and plot results."""
    logger.info("\n--- Run-Length Analysis ---")
    contingency_df = build_run_length_contingency_table(trajectory)
    observed = contingency_df.to_numpy()
    nonzero_col_mask = observed.sum(axis=0) > 0
    filtered_observed = observed[:, nonzero_col_mask]

    if filtered_observed.shape[1] < MIN_CHI2_COLUMNS:
        logger.warning("Not enough data for chi-square test. Skipping.")
        return

    chi2, p, dof, expected = chi2_contingency(filtered_observed, correction=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        residuals = (filtered_observed - expected) / np.sqrt(expected)
        residuals = np.where(np.isfinite(residuals), residuals, 0.0)

    full_residuals = np.zeros_like(observed, dtype=float)
    full_residuals[:, nonzero_col_mask] = residuals
    residuals_df = pd.DataFrame(
        full_residuals,
        index=contingency_df.index,
        columns=contingency_df.columns,
    )
    logger.info("Chi-square test on run lengths: stat=%.2f, p-value=%.4g", chi2, p)

    # Plot residuals heatmap
    plt.figure(figsize=(min(24, 0.25 * residuals_df.shape[1]), 10))
    sns.heatmap(
        residuals_df,
        cmap="coolwarm",
        center=0,
        vmax=5,
        vmin=-5,
        cbar_kws={"label": "Standardized Residual"},
    )
    plt.title("Standardized Residuals of Run-Length Counts")
    plt.xlabel("Run-Length Bins")
    plt.ylabel("Parameter Index")
    plt.tight_layout()
    plt.show()


def get_non_elo_param_indices(
    cumulative_spsa_signal: np.ndarray,
    num_params: int,
    num_elo_params: int,
) -> list[int]:
    """Return indices of parameters with the lowest variance in cumulative signal."""
    num_non_elo = num_params - num_elo_params
    stds = np.std(cumulative_spsa_signal, axis=0)
    sorted_idx = np.argsort(stds)
    return sorted_idx[:num_non_elo].tolist()


def analyze_and_plot_parameter_separation(result: SPSAResult) -> None:
    """Analyze and plot the separation between Elo and non-Elo parameters."""
    logger.info("\n--- Elo vs. Non-Elo Parameter Analysis ---")
    stds = np.std(result.cumulative_spsa_signal, axis=0)
    df = pd.DataFrame({"parameter": np.arange(len(stds)), "std_dev": stds})
    df = df.sort_values("std_dev", ascending=True)
    logger.info("Top 20 parameters with lowest signal standard deviation:")
    logger.info(df.head(20).to_string(index=False))

    non_elo_indices = get_non_elo_param_indices(
        result.cumulative_spsa_signal,
        result.config.num_params,
        result.config.num_elo_params,
    )

    plt.figure(figsize=(12, 5))
    idx = np.arange(len(stds))
    colors = np.array(
        ["orange" if i in non_elo_indices else "blue" for i in idx],
    )
    plt.bar(idx, stds, color=colors)
    plt.xlabel("Parameter Index")
    plt.ylabel("Std. Dev. of Cumulative SPSA Signal")
    plt.title("Separation of Parameters by Signal Variance")
    plt.tight_layout()
    plt.show()


def analyze_parameter_clusters(result: SPSAResult) -> None:
    """Cluster parameters based on trajectory behavior and plot results."""
    logger.info("\n--- Parameter Clustering Analysis ---")
    # Feature 1: Variance of cumulative signal
    variance_feature = np.std(result.cumulative_spsa_signal, axis=0)
    # Feature 2: Mean absolute change
    delta = np.diff(result.trajectory, axis=0)
    mean_abs_change = np.mean(np.abs(delta), axis=0)
    # Feature 3: Final absolute value
    final_abs_value = np.abs(result.final_params)

    feature_matrix = np.column_stack(
        [variance_feature, mean_abs_change, final_abs_value],
    )
    x_scaled = StandardScaler().fit_transform(feature_matrix)

    kmeans = KMeans(
        n_clusters=result.config.analysis_n_clusters,
        random_state=result.config.seed,
        n_init=10,
    )
    labels = kmeans.fit_predict(x_scaled)

    logger.info("Parameter distribution in clusters:")
    for i in range(result.config.analysis_n_clusters):
        param_indices = np.where(labels == i)[0]
        if param_indices.size > 0:
            logger.info(
                "  Cluster %d (%d parameters): %s",
                i,
                len(param_indices),
                param_indices.tolist(),
            )
        else:
            logger.info("  Cluster %d (0 parameters)", i)

    # Plot clustering results
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        x_scaled[:, 0],
        x_scaled[:, 1],
        x_scaled[:, 2],
        c=labels,
        cmap="viridis",
        s=50,
        alpha=0.7,
    )
    ax.set_xlabel("Scaled Signal Variance")
    ax.set_ylabel("Scaled Mean Absolute Change")
    ax.set_zlabel("Scaled Final Absolute Value")
    ax.set_title("Parameter Clustering (3D)")
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    plt.show()

    # --- Advanced Clustering Analysis ---
    logger.info("\n--- Advanced Feature-Based Clustering ---")
    advanced_results = advanced_feature_clustering(
        result.trajectory,
        true_elo_params=result.config.num_elo_params,
        n_clusters=2,
        random_state=result.config.seed or 42,
    )

    logger.info("Advanced clustering results:")
    for key, value in advanced_results.items():
        if key not in {"features", "features_scaled", "features_pca"}:
            logger.info("  %s: %s", key, value)

    # Plot advanced clustering results
    plt.figure(figsize=(8, 6))
    for label in range(result.config.analysis_n_clusters):
        idx = np.where(labels == label)[0]
        plt.scatter(
            advanced_results["features_pca"][idx, 0],
            advanced_results["features_pca"][idx, 1],
            label=f"Cluster {label} ({len(idx)})",
            s=60,
        )
    plt.title("Advanced Clustering of Parameters (PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.show()


def advanced_feature_clustering(
    trajectory: np.ndarray,
    true_elo_params: int | None = None,
    n_clusters: int = 2,
    random_state: int = 42,
) -> dict:
    """Advanced feature-based clustering for parameter time series.

    Returns a dict with clustering results and validation metrics.
    """
    n_steps, n_params = trajectory.shape
    features = []
    for i in range(n_params):
        series = trajectory[:, i]
        diff = np.diff(series)
        # Step 1: Feature extraction
        var = np.var(series)
        mean_abs_diff = np.mean(np.abs(diff))
        autocorr = np.corrcoef(diff[:-1], diff[1:])[0, 1] if len(diff) > 1 else 0.0
        final_value = series[-1]
        value_range = np.max(series) - np.min(series)
        # Spectral feature: power in low freq (trend) vs. high freq (noise)
        fft = np.fft.rfft(series - np.mean(series))
        power = np.abs(fft) ** 2
        low_freq_power = np.sum(power[: max(1, len(power) // 10)])
        high_freq_power = np.sum(power[max(1, len(power) // 10) :])
        features.append(
            [
                var,
                mean_abs_diff,
                autocorr,
                final_value,
                value_range,
                low_freq_power,
                high_freq_power,
            ],
        )
    features = np.array(features)

    # Step 2: Normalization
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Step 3: Dimensionality reduction (optional, for visualization)
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)

    # Step 4: Clustering (KMeans, GMM, DBSCAN)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans_labels = kmeans.fit_predict(features_scaled)
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gmm_labels = gmm.fit_predict(features_scaled)
    dbscan = DBSCAN(eps=1.5, min_samples=2)
    dbscan_labels = dbscan.fit_predict(features_scaled)

    # Step 5: Validation
    sil_kmeans = silhouette_score(features_scaled, kmeans_labels)
    sil_gmm = silhouette_score(features_scaled, gmm_labels)
    sil_dbscan = (
        silhouette_score(features_scaled, dbscan_labels)
        if len(set(dbscan_labels)) > 1
        else float("nan")
    )

    # If true labels are known, compute ARI/NMI
    ari_kmeans = nmi_kmeans = None
    if true_elo_params is not None:
        true_labels = np.zeros(n_params, dtype=int)
        true_labels[true_elo_params:] = 1  # 0: Elo, 1: non-Elo
        ari_kmeans = adjusted_rand_score(true_labels, kmeans_labels)
        nmi_kmeans = normalized_mutual_info_score(true_labels, kmeans_labels)

    # Step 6: Interpretation
    cluster_summary = []
    for label in range(n_clusters):
        idx = np.where(kmeans_labels == label)[0]
        feature_means = features[idx].mean(axis=0).tolist() if len(idx) > 0 else []
        cluster_summary.append(
            {
                "cluster": label,
                "count": len(idx),
                "params": idx.tolist(),
                "feature_means": feature_means,
            },
        )

    # Step 7: Visualization
    plt.figure(figsize=(8, 6))
    for label in range(n_clusters):
        idx = np.where(kmeans_labels == label)[0]
        plt.scatter(
            features_pca[idx, 0],
            features_pca[idx, 1],
            label=f"Cluster {label} ({len(idx)})",
            s=60,
        )
    plt.title("KMeans Clustering of Parameters (PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Step 8: Return results
    logger.info("KMeans Silhouette score: %.3f", sil_kmeans)
    logger.info("GMM Silhouette score: %.3f", sil_gmm)
    if not np.isnan(sil_dbscan):
        logger.info("DBSCAN Silhouette score: %.3f", sil_dbscan)
    else:
        logger.info("DBSCAN Silhouette score: N/A")
    if ari_kmeans is not None:
        logger.info("KMeans ARI: %.3f, NMI: %.3f", ari_kmeans, nmi_kmeans)
    for c in cluster_summary:
        logger.info(
            "Cluster %d (%d params): %s",
            c["cluster"],
            c["count"],
            c["params"],
        )
        logger.info("  Feature means: %s", c["feature_means"])
    return {
        "features": features,
        "features_scaled": features_scaled,
        "features_pca": features_pca,
        "kmeans_labels": kmeans_labels,
        "gmm_labels": gmm_labels,
        "dbscan_labels": dbscan_labels,
        "silhouette": {
            "kmeans": sil_kmeans,
            "gmm": sil_gmm,
            "dbscan": sil_dbscan,
        },
        "ari_kmeans": ari_kmeans,
        "nmi_kmeans": nmi_kmeans,
        "cluster_summary": cluster_summary,
    }


def main() -> None:
    """Run SPSA optimization simulation and analysis."""
    config = SPSAConfig()
    logger.info(
        "Starting SPSA optimization with config:\n%s",
        config.model_dump_json(indent=2),
    )

    result = spsa_optimization(config)

    logger.info("\n--- Simulation Summary ---")
    logger.info("Simulation complete in %.2f seconds.", result.elapsed_time)
    logger.info("Final parameters (first 5): %s", result.final_params[:5])
    logger.info("Convergence metrics: %s", result.convergence_metrics)

    # --- Plotting and Analysis ---
    plot_basic_results(result)
    analyze_and_plot_parameter_separation(result)
    analyze_and_plot_pca(result.trajectory)
    analyze_and_plot_run_lengths(result.trajectory)
    analyze_parameter_clusters(result)

    logger.info("\nAnalysis complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user, exiting.")
        sys.exit(0)
