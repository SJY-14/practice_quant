"""
TDA (Topological Data Analysis) Feature Extraction
Based on: "Enhancing financial time series forecasting through topological data analysis"
https://link.springer.com/article/10.1007/s00521-024-10787-x
"""

import numpy as np
from typing import Tuple, List
from ripser import ripser
from persim import PersistenceImager


def time_delay_embedding(x: np.ndarray, dim: int = 3, tau: int = 1) -> np.ndarray:
    """
    Takens' time delay embedding
    Converts 1D time series to point cloud in higher dimension

    Args:
        x: 1D time series
        dim: Embedding dimension
        tau: Time delay

    Returns:
        Point cloud of shape (n_points, dim)
    """
    n = len(x) - (dim - 1) * tau
    if n <= 0:
        raise ValueError(f"Time series too short for embedding: len={len(x)}, dim={dim}, tau={tau}")

    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = x[i * tau: i * tau + n]

    return embedded


def compute_persistence_diagram(point_cloud: np.ndarray, max_dim: int = 1) -> List[np.ndarray]:
    """
    Compute persistence diagram using Vietoris-Rips complex

    Args:
        point_cloud: Point cloud from time delay embedding
        max_dim: Maximum homology dimension

    Returns:
        List of persistence diagrams for each dimension
    """
    result = ripser(point_cloud, maxdim=max_dim)
    return result['dgms']


def persistence_entropy(diagram: np.ndarray) -> float:
    """
    Compute persistent entropy from persistence diagram
    Measures complexity and irregularity of the time series

    Args:
        diagram: Persistence diagram (birth, death) pairs

    Returns:
        Entropy value
    """
    # Filter out infinite death times
    finite_mask = np.isfinite(diagram[:, 1])
    diagram = diagram[finite_mask]

    if len(diagram) == 0:
        return 0.0

    # Compute lifetimes
    lifetimes = diagram[:, 1] - diagram[:, 0]
    lifetimes = lifetimes[lifetimes > 0]

    if len(lifetimes) == 0:
        return 0.0

    # Normalize to get probabilities
    total_lifetime = np.sum(lifetimes)
    if total_lifetime == 0:
        return 0.0

    probs = lifetimes / total_lifetime

    # Shannon entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    return entropy


def persistence_amplitude(diagram: np.ndarray) -> float:
    """
    Compute amplitude (spread) of persistence diagram
    Indicates stability and recurrence of patterns

    Args:
        diagram: Persistence diagram

    Returns:
        Amplitude value (max lifetime)
    """
    finite_mask = np.isfinite(diagram[:, 1])
    diagram = diagram[finite_mask]

    if len(diagram) == 0:
        return 0.0

    lifetimes = diagram[:, 1] - diagram[:, 0]
    return np.max(lifetimes) if len(lifetimes) > 0 else 0.0


def persistence_num_points(diagram: np.ndarray, threshold: float = 0.0) -> int:
    """
    Count significant topological features
    Indicates persistent structures (cycles, trends)

    Args:
        diagram: Persistence diagram
        threshold: Minimum lifetime to be considered significant

    Returns:
        Number of significant points
    """
    finite_mask = np.isfinite(diagram[:, 1])
    diagram = diagram[finite_mask]

    if len(diagram) == 0:
        return 0

    lifetimes = diagram[:, 1] - diagram[:, 0]
    return int(np.sum(lifetimes > threshold))


def extract_tda_features(
    window: np.ndarray,
    embedding_dim: int = 3,
    embedding_tau: int = 1,
    homology_dim: int = 1
) -> Tuple[float, float, int]:
    """
    Extract TDA features from a time series window

    Args:
        window: Time series window (1D array)
        embedding_dim: Takens embedding dimension
        embedding_tau: Takens time delay
        homology_dim: Maximum homology dimension

    Returns:
        Tuple of (entropy, amplitude, num_points)
    """
    # Normalize window
    window = (window - np.mean(window)) / (np.std(window) + 1e-10)

    # Time delay embedding
    point_cloud = time_delay_embedding(window, dim=embedding_dim, tau=embedding_tau)

    # Compute persistence diagram
    diagrams = compute_persistence_diagram(point_cloud, max_dim=homology_dim)

    # Extract features from H1 (1-dimensional holes/loops)
    h1_diagram = diagrams[1] if len(diagrams) > 1 else diagrams[0]

    entropy = persistence_entropy(h1_diagram)
    amplitude = persistence_amplitude(h1_diagram)
    num_points = persistence_num_points(h1_diagram)

    return entropy, amplitude, num_points


class TDAFeatureExtractor:
    """
    TDA Feature Extractor for time series
    Extracts entropy, amplitude, and number of points from persistence diagrams
    """

    def __init__(
        self,
        window_size: int = 50,
        embedding_dim: int = 3,
        embedding_tau: int = 1,
        homology_dim: int = 1
    ):
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.embedding_tau = embedding_tau
        self.homology_dim = homology_dim

    def extract_features(self, series: np.ndarray) -> np.ndarray:
        """
        Extract TDA features for entire series using sliding windows

        Args:
            series: Full time series

        Returns:
            Array of shape (n_windows, 3) with [entropy, amplitude, num_points]
        """
        n_windows = len(series) - self.window_size + 1
        features = np.zeros((n_windows, 3))

        for i in range(n_windows):
            window = series[i:i + self.window_size]
            try:
                entropy, amplitude, num_points = extract_tda_features(
                    window,
                    embedding_dim=self.embedding_dim,
                    embedding_tau=self.embedding_tau,
                    homology_dim=self.homology_dim
                )
                features[i] = [entropy, amplitude, num_points]
            except Exception as e:
                # Handle edge cases
                features[i] = [0.0, 0.0, 0]

        return features


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    t = np.linspace(0, 10 * np.pi, 500)
    test_series = np.sin(t) + 0.1 * np.random.randn(500)

    extractor = TDAFeatureExtractor(window_size=50)
    features = extractor.extract_features(test_series)

    print(f"Extracted {len(features)} feature vectors")
    print(f"Sample features (entropy, amplitude, num_points):")
    print(features[:5])
