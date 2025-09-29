#!/usr/bin/env python
# run_benchmark_features.py
from __future__ import annotations
import argparse
import time
import numpy as np
from sklearn.manifold import TSNE

from birsne import BiRSNE
from irsne import IRSNE
from metrics_viz import clustering_quality, scatter_embedding

def _print_block(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def eval_and_print(name: str, Y_all: np.ndarray, labels: np.ndarray, n0: int, t_sec: float):
    sil, db = clustering_quality(Y_all, labels)
    print(f"{name:>12} | time: {t_sec:7.2f}s | Silhouette: {sil:6.4f} | DB: {db:6.4f}")
    return dict(name=name, time=t_sec, silhouette=sil, db=db)

def stratified_init_stream_split(features: np.ndarray, labels: np.ndarray, split: float, seed: int):
    """Stratified split per class: `split` fraction to init, remainder to stream."""
    X_init, y_init, X_rem, y_rem = [], [], [], []
    rng = np.random.default_rng(seed)
    for cls in np.unique(labels):
        idxs = np.where(labels == cls)[0]
        rng.shuffle(idxs)
        n0 = int(split * len(idxs))
        init_idx, rem_idx = idxs[:n0], idxs[n0:]
        X_init.append(features[init_idx]); y_init.append(labels[init_idx])
        X_rem.append(features[rem_idx]);   y_rem.append(labels[rem_idx])
    return (np.vstack(X_init), np.hstack(y_init),
            np.vstack(X_rem),  np.hstack(y_rem))

def stream_batches(X: np.ndarray, y: np.ndarray, batch_size: int):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

def main():
    ap = argparse.ArgumentParser(description="Benchmark i-RSNE vs Bi-RSNE vs BH t-SNE on precomputed features")
    # data sources
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prefix", type=str,
                   help="Prefix for .npy files (uses {prefix}_features.npy and {prefix}_labels.npy)")
    g.add_argument("--features", type=str, help="Path to features .npy (overrides --prefix)")
    ap.add_argument("--labels", type=str, help="Path to labels .npy (required if --features is used)")
    # split % for initialization
    ap.add_argument("--split", type=float, default=0.5, help="Fraction used for initialization (0..1)")
    ap.add_argument("--seed", type=int, default=42)
    # RSNE params
    ap.add_argument("--K", type=int, default=100, help="KMeans clusters for (Bi/i)-RSNE")
    ap.add_argument("--batch", type=int, default=1000, help="Batch size for Bi-RSNE and stream loop")
    ap.add_argument("--eta", type=float, default=10.0, help="Step size for (P-Q) updates")
    ap.add_argument("--iters", type=int, default=2, help="Iterations per batch (Bi) / per point (i)")
    # optional class cap (for quick runs)
    ap.add_argument("--per-class", type=int, default=None,
                    help="If set, cap to N samples per class before splitting (stratified).")
    # plotting
    ap.add_argument("--plots", action="store_true", help="Save scatter plots for each method")
    ap.add_argument("--plot_prefix", default="bench_features")
    args = ap.parse_args()

    # 1) load features/labels
    if args.features:
        if not args.labels:
            raise SystemExit("--labels is required when --features is provided")
        X = np.load(args.features).astype(np.float32)
        y = np.load(args.labels)
        prefix_for_name = args.features.rsplit(".", 1)[0]
    else:
        X = np.load(f"{args.prefix}_features.npy").astype(np.float32)
        y = np.load(f"{args.prefix}_labels.npy")
        prefix_for_name = args.prefix

    # optional cap per class (before split)
    if args.per_class is not None:
        X_cap, y_cap = [], []
        rng = np.random.default_rng(args.seed)
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            rng.shuffle(idx)
            take = idx[:min(args.per_class, len(idx))]
            X_cap.append(X[take]); y_cap.append(y[take])
        X = np.vstack(X_cap); y = np.hstack(y_cap)

    # stratified init/stream split
    X_init, y_init, X_rem, y_rem = stratified_init_stream_split(X, y, split=args.split, seed=args.seed)
    n0 = len(y_init)
    D = X.shape[1]
    print("\n" + "=" * 70)
    print(f"Dataset: {prefix_for_name} | {len(X)} samples, D={D} | init={n0}, stream={len(X_rem)} (split={args.split:.2f})")
    print("=" * 70)

    # guard K
    K_safe = min(args.K, max(2, len(X_init) - 1))
    results = []

    # 2) i-RSNE
    print("\n" + "=" * 70 + "\ni-RSNE\n" + "=" * 70)
    irsne = IRSNE(K=K_safe, eta=args.eta, iters=max(1, args.iters), seed=args.seed)
    t0 = time.time()
    _ = irsne.fit_init(X_init, y_init)
    # stream one-by-one (iterate batches for convenience)
    for Xb, yb in stream_batches(X_rem, y_rem, batch_size=args.batch):
        for x, lbl in zip(Xb, yb):
            irsne.add_point(x, int(lbl))
    t_i = time.time() - t0
    X_all_i, Y_all_i, labels_i = irsne.get_embedding()
    results.append(eval_and_print("i-RSNE", Y_all_i, labels_i, n0=n0, t_sec=t_i))
    if args.plots:
        scatter_embedding(Y_all_i, labels_i, out_path=f"{args.plot_prefix}_irsne.png", title="i-RSNE (features)")

    # 3) Bi-RSNE
    print("\n" + "=" * 70 + "\nBi-RSNE\n" + "=" * 70)
    birsne = BiRSNE(K=K_safe, eta=args.eta, iters=max(2, args.iters), seed=args.seed)
    t0 = time.time()
    _ = birsne.fit_init(X_init, y_init)
    for Xb, yb in stream_batches(X_rem, y_rem, batch_size=args.batch):
        birsne.add_batch(Xb, yb)
    t_bi = time.time() - t0
    X_all_b, Y_all_b, labels_b = birsne.get_embedding()
    results.append(eval_and_print("Bi-RSNE", Y_all_b, labels_b, n0=n0, t_sec=t_bi))
    if args.plots:
        scatter_embedding(Y_all_b, labels_b, out_path=f"{args.plot_prefix}_birsne.png", title="Bi-RSNE (features)")

    # 4) Barnes–Hut t-SNE on ALL features
    print("\n" + "=" * 70 + "\nBarnes–Hut t-SNE (full)\n" + "=" * 70)
    t0 = time.time()
    Y_full = TSNE(n_components=2, method="barnes_hut", random_state=args.seed)\
             .fit_transform(X).astype(np.float32)
    t_tsne = time.time() - t0
    results.append(eval_and_print("BH t-SNE", Y_full, y, n0=n0, t_sec=t_tsne))
    if args.plots:
        scatter_embedding(Y_full, y, out_path=f"{args.plot_prefix}_bh.png", title="Barnes–Hut t-SNE (features)")

    # 5) summary
    print("\n" + "=" * 70 + "\nSummary\n" + "=" * 70)
    print(f"{'Method':>12} | {'time(s)':>8} | {'Silhouette':>10} | {'DB':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:>12} | {r['time']:8.2f} | {r['silhouette']:10.4f} | {r['db']:8.4f}")

if __name__ == "__main__":
    main()
