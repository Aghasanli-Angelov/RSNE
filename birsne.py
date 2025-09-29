# birsne.py
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

_EPS = 1e-9

class BiRSNE:
    """
    Batch-incremental, cluster-seeded t-SNE (Bi-RSNE).
    - Fit once on an initial batch (KMeans + t-SNE).
    - For each new batch: init near cluster low-means and do a few (P-Q) updates in parallel.
    """
    def __init__(self, K: int = 50, eta: float = 10.0, iters: int = 2, seed: int = 42):
        self.K = K
        self.eta = eta
        self.iters = iters
        self.seed = seed

        self._clusters = None
        self._C_high = None  # (K,D)
        self._C_low  = None  # (K,2)
        self._sigma  = None  # (K,)
        self._D = None

        self._X_list, self._Y_list, self._y_list = [], [], []

    @staticmethod
    def _stack(arrs):
        return np.stack(arrs) if len(arrs) else None

    def _refresh_cache(self):
        self._C_high = self._stack([c["high_mean"] for c in self._clusters]).astype(np.float32)
        self._C_low  = self._stack([c["low_mean"]  for c in self._clusters]).astype(np.float32)
        self._sigma  = np.clip(
            np.array([max(c["std"], 1e-3) for c in self._clusters], dtype=np.float32),
            1e-3, None
        )

    def _init_clusters(self, X_init, Y_init, labels):
        K, D = self.K, X_init.shape[1]
        clusters = []
        for k in range(K):
            m = (labels == k)
            if not np.any(m):
                clusters.append(dict(
                    high_mean=np.zeros(D, dtype=np.float32),
                    low_mean =np.zeros(2, dtype=np.float32),
                    std=1.0, count=1
                ))
            else:
                hd, ld = X_init[m], Y_init[m]
                norms = np.linalg.norm(hd, axis=1)
                clusters.append(dict(
                    high_mean=hd.mean(axis=0).astype(np.float32),
                    low_mean =ld.mean(axis=0).astype(np.float32),
                    std=float(np.std(norms)),
                    count=int(len(hd))
                ))
        self._clusters = clusters
        self._refresh_cache()

    def fit_init(self, X_init: np.ndarray, y_init: np.ndarray):
        """KMeans + t-SNE on the initial batch; caches cluster stats."""
        self._D = X_init.shape[1]
        km = KMeans(n_clusters=self.K, random_state=self.seed).fit(X_init)
        Y_init = TSNE(n_components=2, method="barnes_hut", random_state=self.seed)\
                 .fit_transform(X_init).astype(np.float32)
        self._init_clusters(X_init, Y_init, km.labels_)
        # store
        self._X_list.extend([x for x in X_init])
        self._Y_list.extend([y for y in Y_init])
        self._y_list.extend(list(y_init))
        return Y_init

    def add_batch(self, Xb: np.ndarray, yb: np.ndarray):
        """Embed a new batch (parallel)."""
        if Xb.size == 0:
            return
        Xb = Xb.astype(np.float32)
        d2 = np.sum((Xb[:, None, :] - self._C_high[None, :, :])**2, axis=2)  # (B,K)
        idx = np.argmin(d2, axis=1)

        Yb = self._C_low[idx] + 0.1 * np.random.randn(len(Xb), 2).astype(np.float32)

        for _ in range(self.iters):
            P = np.exp(-d2 / (2 * self._sigma[None, :]**2))
            P /= (P.sum(axis=1, keepdims=True) + _EPS)

            d2l = np.sum((Yb[:, None, :] - self._C_low[None, :, :])**2, axis=2)
            Q = 1.0 / (1.0 + d2l)
            Q /= (Q.sum(axis=1, keepdims=True) + _EPS)

            coef = 2.0 * (P - Q) / (1.0 + d2l)
            grads = np.einsum('ik,ikj->ij', coef, (Yb[:, None, :] - self._C_low[None, :, :]))
            Yb -= self.eta * grads

        self._X_list.extend([x for x in Xb])
        self._Y_list.extend([y for y in Yb])
        self._y_list.extend(list(yb))

    def get_embedding(self):
        return np.vstack(self._X_list), np.vstack(self._Y_list), np.array(self._y_list)
