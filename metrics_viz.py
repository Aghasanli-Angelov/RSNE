# metrics_viz.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def clustering_quality(Y, labels):
    sil = silhouette_score(Y, labels)
    db  = davies_bouldin_score(Y, labels)
    return sil, db


def scatter_embedding(Y, labels, out_path: str | None = None, title: str = "Embedding"):
    plt.figure(figsize=(8, 8))
    for cls in np.unique(labels):
        m = (labels == cls)
        plt.scatter(Y[m, 0], Y[m, 1], alpha=0.4, s=6)
    plt.xticks([]); plt.yticks([])
    plt.title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300)
    plt.show()
