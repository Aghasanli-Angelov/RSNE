# run_benchmark.py
from __future__ import annotations
import argparse
import time
import numpy as np
from sklearn.manifold import TSNE

from birsne import BiRSNE
from irsne import IRSNE
from data_blobs import make_blob_dataset, split_initial_and_stream, stream_batches
from metrics_viz import clustering_quality, scatter_embedding

def _print_block(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def eval_and_print(name: str, Y_all: np.ndarray, labels: np.ndarray, n0: int, t_sec: float):
    sil, db = clustering_quality(Y_all, labels)
    print(f"{name:>12} | time: {t_sec:7.2f}s | Silhouette: {sil:6.4f} | DB: {db:6.4f}")
    return dict(name=name, time=t_sec, silhouette=sil, db=db)

def main():
    ap = argparse.ArgumentParser(description="Benchmark i-RSNE vs Bi-RSNE vs Barnes–Hut t-SNE on blobs")
    # data
    ap.add_argument("--samples", type=int, default=8000)
    ap.add_argument("--features", type=int, default=20)
    ap.add_argument("--centers", type=int, default=12)
    ap.add_argument("--std", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    # split % for initialization
    ap.add_argument("--split", type=float, default=0.5, help="Fraction of data for initialization (0..1)")
    # RSNE params
    ap.add_argument("--K", type=int, default=60, help="KMeans clusters for (Bi/i)-RSNE")
    ap.add_argument("--batch", type=int, default=800, help="Batch size for Bi-RSNE and stream loop")
    ap.add_argument("--eta", type=float, default=10.0, help="Step size for (P-Q) updates")
    ap.add_argument("--iters", type=int, default=2, help="Iterations per batch (Bi) / per point (i)")
    # plotting
    ap.add_argument("--plots", action="store_true", help="Save scatter plots for each method")
    ap.add_argument("--plot_prefix", default="bench_blobs")
    args = ap.parse_args()

    # 1) dataset
    X, y = make_blob_dataset(
        n_samples=args.samples, n_features=args.features,
        centers=args.centers, cluster_std=args.std, seed=args.seed
    )
    X_init, y_init, X_rem, y_rem = split_initial_and_stream(
        X, y, split_ratio=args.split, seed=args.seed
    )
    n0 = len(y_init)
    _print_block(f"Dataset: {args.samples} samples, {args.features}D, centers={args.centers} | init={n0}, stream={len(X_rem)} (split={args.split:.2f})")

    # Keep K valid
    K_safe = min(args.K, max(2, len(X_init) - 1))

    results = []

    # 2) i-RSNE
    _print_block("i-RSNE")
    irsne = IRSNE(K=K_safe, eta=args.eta, iters=max(1, args.iters), seed=args.seed)
    t0 = time.time()
    _ = irsne.fit_init(X_init, y_init)
    for Xb, yb in stream_batches(X_rem, y_rem, batch_size=args.batch):
        for x, lbl in zip(Xb, yb):
            irsne.add_point(x, int(lbl))
    t_i = time.time() - t0
    X_all_i, Y_all_i, labels_i = irsne.get_embedding()
    results.append(eval_and_print("i-RSNE", Y_all_i, labels_i, n0=n0, t_sec=t_i))
    if args.plots:
        scatter_embedding(Y_all_i, labels_i, out_path=f"{args.plot_prefix}_irsne.png", title="i-RSNE (blobs)")

    # 3) Bi-RSNE
    _print_block("Bi-RSNE")
    birsne = BiRSNE(K=K_safe, eta=args.eta, iters=max(2, args.iters), seed=args.seed)
    t0 = time.time()
    _ = birsne.fit_init(X_init, y_init)
    for Xb, yb in stream_batches(X_rem, y_rem, batch_size=args.batch):
        birsne.add_batch(Xb, yb)
    t_bi = time.time() - t0
    X_all_b, Y_all_b, labels_b = birsne.get_embedding()
    results.append(eval_and_print("Bi-RSNE", Y_all_b, labels_b, n0=n0, t_sec=t_bi))
    if args.plots:
        scatter_embedding(Y_all_b, labels_b, out_path=f"{args.plot_prefix}_birsne.png", title="Bi-RSNE (blobs)")

    # 4) Barnes–Hut t-SNE (full)
    _print_block("Barnes–Hut t-SNE (full)")
    t0 = time.time()
    Y_full = TSNE(n_components=2, method="barnes_hut", random_state=args.seed)\
             .fit_transform(X).astype(np.float32)
    t_tsne = time.time() - t0
    labels_full = y
    results.append(eval_and_print("BH t-SNE", Y_full, labels_full, n0=n0, t_sec=t_tsne))
    if args.plots:
        scatter_embedding(Y_full, labels_full, out_path=f"{args.plot_prefix}_bh.png", title="Barnes–Hut t-SNE (blobs)")

    # 5) summary
    _print_block("Summary")
    print(f"{'Method':>12} | {'time(s)':>8} | {'Silhouette':>10} | {'DB':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:>12} | {r['time']:8.2f} | {r['silhouette']:10.4f} | {r['db']:8.4f}")

if __name__ == "__main__":
    main()
