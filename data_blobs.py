# data_blobs.py
from __future__ import annotations
import numpy as np
from sklearn.datasets import make_blobs

def make_blob_dataset(
    n_samples: int = 6000,
    n_features: int = 20,
    centers: int = 10,
    cluster_std: float = 2.0,
    seed: int = 42,
):
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=seed,
    )
    return X.astype(np.float32), y

def split_initial_and_stream(X, y, split_ratio: float = 0.5, seed: int = 42):
    """split_ratio fraction goes to init; rest to stream."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n0 = int(split_ratio * len(X))
    init_idx, rem_idx = idx[:n0], idx[n0:]
    return X[init_idx], y[init_idx], X[rem_idx], y[rem_idx]

def stream_batches(X, y, batch_size: int = 500):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]
