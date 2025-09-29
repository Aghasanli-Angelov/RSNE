#!/usr/bin/env python
# extract_features.py
import argparse
from typing import Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import torchvision.transforms as T
import torchvision.datasets as dsets


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract CIFAR features with CLIP ViT-L/14 or DINOv2 ViT-L/14")
    ap.add_argument("--model", choices=["clip_vitl14", "dinov2_vitl14"], required=True,
                    help="Backbone to use for feature extraction")
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True,
                    help="Dataset to use")
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="Use train or test split")
    ap.add_argument("--output-prefix", type=str, required=True,
                    help="Prefix for saving .npy files (prefix_features.npy, prefix_labels.npy)")
    ap.add_argument("--batch-size", type=int, default=64, help="Dataloader batch size")
    ap.add_argument("--workers", type=int, default=4, help="Num workers for dataloader")
    ap.add_argument("--per-class", type=int, default=None,
                    help="If set, sample at most N images per class (class-balanced)")
    ap.add_argument("--device", type=str, default="auto",
                    help="'cuda', 'cpu', or 'auto'")
    return ap.parse_args()


def get_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def load_model_and_transform(model_name: str, device: torch.device):
    """
    Returns (model, transform, forward_fn)
    forward_fn(images_tensor_on_device) -> features_on_cpu_np
    """
    if model_name == "dinov2_vitl14":
        # facebookresearch/dinov2 hub model returns a module that maps (B,3,224,224) -> (B, D)
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
        model.eval()
        transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        @torch.no_grad()
        def forward_fn(x: torch.Tensor) -> np.ndarray:
            feats = model(x)
            # Ensure (B, D)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            return feats.detach().cpu().numpy()

        return model, transform, forward_fn

    elif model_name == "clip_vitl14":
        # OpenAI CLIP
        import clip  # pip install git+https://github.com/openai/CLIP.git
        model, preprocess = clip.load("ViT-L/14", device=device)  # returns (model, preprocess)
        model.eval()
        # We prefer CLIP's own preprocess (resize+center-crop+normalize)
        transform = preprocess

        @torch.no_grad()
        def forward_fn(x: torch.Tensor) -> np.ndarray:
            # Use encode_image for CLIP
            feats = model.encode_image(x)
            return feats.detach().cpu().numpy()

        return model, transform, forward_fn

    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_dataset(name: str, split: str, transform) -> torch.utils.data.Dataset:
    root = "./data"
    train = (split == "train")
    if name == "cifar10":
        return dsets.CIFAR10(root=root, train=train, download=True, transform=transform)
    elif name == "cifar100":
        return dsets.CIFAR100(root=root, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def class_balanced_subset_indices(dataset, per_class: int) -> List[int]:
    # dataset.targets exists for CIFAR-10/100
    targets = np.array(dataset.targets)
    classes = np.unique(targets)
    rng = np.random.default_rng(42)
    indices: List[int] = []
    for c in classes:
        cand = np.where(targets == c)[0]
        rng.shuffle(cand)
        take = cand if per_class is None else cand[:per_class]
        indices.extend(take.tolist())
    rng.shuffle(indices)
    return indices


def main():
    args = build_args()
    device = get_device(args.device)

    print(f"Config: model={args.model}, dataset={args.dataset}, split={args.split}, "
          f"batch={args.batch_size}, per_class={args.per_class}, device={device}")

    # 1) Model + transform
    print("Loading model & transform…")
    model, transform, forward_fn = load_model_and_transform(args.model, device)

    # 2) Dataset (+ optional class-balanced subsampling)
    print(f"Loading dataset {args.dataset} ({args.split})…")
    ds = load_dataset(args.dataset, args.split, transform)

    if args.per_class is not None:
        print(f"Building balanced subset: per_class={args.per_class}")
        idx = class_balanced_subset_indices(ds, args.per_class)
        ds = Subset(ds, idx)

    # 3) DataLoader
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=(device.type == "cuda"))

    # 4) Extract features
    print("Extracting features…")
    features_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in tqdm(dl, desc="Feature Extraction"):
            images = images.to(device, non_blocking=True)
            feats_np = forward_fn(images)  # (B, D)
            features_list.append(feats_np)
            labels_list.append(labels.numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # 5) Save (same naming style as your original script)
    np.save(f"{args.output_prefix}_features.npy", features)
    np.save(f"{args.output_prefix}_labels.npy", labels)

    print("Done.")
    print(f"- {args.output_prefix}_features.npy  shape={features.shape}")
    print(f"- {args.output_prefix}_labels.npy    shape={labels.shape}")


if __name__ == "__main__":
    main()
