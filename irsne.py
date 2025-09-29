# irsne.py
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

_EPS = 1e-9

class IRSNE:
    """
    Incremental (point-by-point) r-SNE.
    - KMeans + t-SNE on init
    - For each new point: nearest cluster → init near low-mean → a few (P-Q) nudges
    - Updates cluster running stats per point
    """
    def __init__(self, K: int = 50, eta: float = 10.0, iters: int = 1, seed: int = 42):
        self.K = K
        self.eta = eta
        self.iters = iters
        self.seed = seed

        self._clusters = None
        self._C_high = None
        self._C_low  = None
        self._sigma  = None
        self._D = None

        self._X_list, self._Y_list, self._y_list = [], [], []

    def _refresh_cache(self):
        self._C_high = np.stack([c["high_mean"] for c in self._clusters]).astype(np.float32)
        self._C_low  = np.stack([c["low_mean"]  for c in self._clusters]).astype(np.float32)
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
                    std=1.0, count=1, sum_sq=np.dot(np.zeros(D, np.float32), np.zeros(D, np.float32))
                ))
            else:
                hd, ld = X_init[m], Y_init[m]
                norms = np.linalg.norm(hd, axis=1)
                clusters.append(dict(
                    high_mean=hd.mean(axis=0).astype(np.float32),
                    low_mean =ld.mean(axis=0).astype(np.float32),
                    std=float(np.std(norms)),
                    count=int(len(hd)),
                    sum_sq=float((norms**2).mean())
                ))
        self._clusters = clusters
        self._refresh_cache()

    def fit_init(self, X_init: np.ndarray, y_init: np.ndarray):
        self._D = X_init.shape[1]
        km = KMeans(n_clusters=self.K, random_state=self.seed).fit(X_init)
        Y_init = TSNE(n_components=2, method="barnes_hut", random_state=self.seed)\
                 .fit_transform(X_init).astype(np.float32)
        self._init_clusters(X_init, Y_init, km.labels_)
        self._X_list.extend([x for x in X_init])
        self._Y_list.extend([y for y in Y_init])
        self._y_list.extend(list(y_init))
        return Y_init

    def add_point(self, x: np.ndarray, y_label: int):
        x = x.astype(np.float32)
        d2h = np.sum((self._C_high - x[None, :])**2, axis=1)  # (K,)
        k = int(np.argmin(d2h))
        y = self._C_low[k] + 0.1 * np.random.randn(2).astype(np.float32)

        for _ in range(self.iters):
            P = np.exp(-d2h / (2 * self._sigma**2))
            P /= (P.sum() + _EPS)
            d2l = np.sum((self._C_low - y[None, :])**2, axis=1)
            Q = 1.0 / (1.0 + d2l)
            Q /= (Q.sum() + _EPS)
            coef = 2.0 * (P - Q) / (1.0 + d2l)
            grad = np.einsum('k,kj->j', coef, (y[None, :] - self._C_low))
            y -= self.eta * grad

        # update cluster stats
        c = self._clusters[k]
        n0, m0 = c["count"], c["high_mean"]
        sum0 = c.get("sum_sq", float(np.dot(m0, m0)) * max(n0, 1))
        total = n0 + 1
        m1 = (n0 * m0 + x) / total
        norm2 = float(np.dot(x, x))
        sum_sq1 = (sum0 + norm2) / total
        var = float(sum_sq1 - float(np.dot(m1, m1)))
        c["high_mean"] = m1
        c["sum_sq"] = sum_sq1
        c["std"] = float(np.sqrt(max(var, 1e-9)))
        c["count"] = total

        self._refresh_cache()
        self._X_list.append(x)
        self._Y_list.append(y)
        self._y_list.append(int(y_label))

    def get_embedding(self):
        return np.vstack(self._X_list), np.vstack(self._Y_list), np.array(self._y_list)
