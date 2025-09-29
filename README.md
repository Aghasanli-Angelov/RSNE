# Recursive SNE: Fast Prototype-Based t-SNE for Large-Scale and Online Data

This repository contains the official implementation of the paper:

> Aghasanli, A., & Angelov, P. (2025). Recursive SNE: Fast prototype-based t-SNE for large-scale and online data. *Transactions on Machine Learning Research (TMLR)*. [OpenReview page](https://openreview.net/forum?id=7wCPAFMDWM)

**Recursive SNE (RSNE)** provides fast, incremental 2‑D (or 3-D) visualizations by seeding new points near **cluster centroids** learned from an initial batch, then applying a few light t‑SNE–style refinement steps.

This repo includes two variants and ready-to-run benchmarks:

- **i-RSNE** — incremental; embeds **one point at a time** (sequential).
- **Bi-RSNE** — batch-incremental; embeds **batches in parallel** (typically faster).
- **Baseline** — original **Barnes–Hut t‑SNE**.

---

## Repository Structure

```
rSNE_submission/
├─ birsne.py                  # Bi-RSNE (batch) implementation
├─ irsne.py                   # i-RSNE (point-by-point) implementation
├─ data_blobs.py              # synthetic blobs + batching utilities
├─ metrics_viz.py             # Silhouette/DB metrics + plotting
├─ feature_extraction.py      # CLI to extract CLIP/DINOv2 features on CIFAR-10/100
├─ run_benchmark.py           # benchmark on synthetic blobs
└─ run_benchmark_features.py  # benchmark on precomputed deep features
```

---

## Dependencies

- `numpy`, `scikit-learn`, `matplotlib`
- `torch`, `torchvision`, `tqdm`
- (Optional) **CLIP** for CLIP features: `git+https://github.com/openai/CLIP.git`
- (Optional) **DINOv2** via `torch.hub` (`facebookresearch/dinov2`)

> A GPU is optional; it mainly speeds up feature extraction.

---

## What r‑SNE Does

1. **Seed with clusters:** Run KMeans on an initial batch and compute a 2‑D t‑SNE map for that same batch.  
2. **Store cluster stats:** For each cluster, keep a high‑D centroid, a low‑D centroid, and a spread estimate.  
3. **Embed new data:**  
   - **i‑RSNE:** for each new point → find nearest high‑D cluster → initialize near its low‑D mean → apply 1–few (P−Q) nudges → (optionally) update stats.  
   - **Bi‑RSNE:** same idea, but **vectorized for a whole batch** (parallel updates; faster processing).  
4. **Evaluate:** Silhouette, Davies–Bouldin (lower is better).

---

## Benchmark on Synthetic Blobs

Sanity‑check the methods without external data.

```bash
python run_benchmark.py   --samples 8000 --features 20 --centers 12   --split 0.5   --K 60 --batch 800 --iters 2   --plots
```

**Key flags**

- `--split` : fraction of data used for initialization (remainder streams in)  
- `--K`     : KMeans clusters that define the r‑SNE scaffold  
- `--iters` : refinement steps (use small number of iterations)  
- `--batch` : stream batch size (for Bi‑RSNE)

**Output**

- Console table comparing **i‑RSNE**, **Bi‑RSNE**, and **Barnes–Hut t‑SNE**: time, Silhouette, DB, and kNN hold‑out.  
- Optional PNG plots (`--plots`) saved next to the script.

---

## Replicating Paper Results with Deep Features

### 1) Extract features (DINOv2 or CLIP) on CIFAR‑10/100

```bash
# DINOv2 ViT‑L/14 on CIFAR‑100 train
python feature_extraction.py   --model dinov2_vitl14   --dataset cifar100   --split train   --batch-size 128   --output-prefix cifar100_dino_train

# CLIP ViT‑L/14 on CIFAR‑10 test
python feature_extraction.py   --model clip_vitl14   --dataset cifar10   --split test   --batch-size 128   --output-prefix cifar10_clip_test
```

This produces:

```
<prefix>_features.npy   # shape (N, D)
<prefix>_labels.npy     # shape (N,)
```

### 2) Benchmark on those features

Using a **prefix**:

```bash
python run_benchmark_features.py   --prefix cifar100_dino_train   --split 0.5   --K 200 --batch 1000 --iters 2   --plots
```

Or pass explicit paths:

```bash
python run_benchmark_features.py   --features cifar10_clip_test_features.npy   --labels   cifar10_clip_test_labels.npy   --split 0.3 --K 100 --batch 800 --iters 2   --plots
```

**What it prints**

- Timing and quality metrics (Silhouette, DB)—same protocol as the blobs benchmark.

---

## When to Use Which Variant?

- **Bi‑RSNE** — prefer for speed and stability (parallel batch updates).  
- **i‑RSNE** — for truly one‑by‑one streams.

Both variants target similar asymptotic costs for the refinement step; Bi‑RSNE is typically faster due to vectorized processing.

---

## Tips & Common Tweaks
  
- Keep `--iters` **small** (1–3). More steps rarely improve much here.  
- For CLIP features, install the CLIP package (`git+https://github.com/openai/CLIP.git`).


## Citation

If you find this repository useful, please cite:

```bibtex
@article{rsne2025,
  title={Recursive SNE: Fast Prototype-Based t-SNE for Large-Scale and Online Data},
  author={Aghasanli, Agil and Angelov, Plamen},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2025},
  note={Accepted, camera-ready pending},
  url={https://openreview.net/forum?id=7wCPAFMDWM}
}
