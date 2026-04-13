"""Causal ablation of SAE features by classical-feature family.

The correlation result says SAE features match classical features.
This script asks the causal question: does the classifier actually rely
on those SAE features, or do they just happen to encode classical info
without the classifier ever using them?

Method:
  1. Bucket alive SAE features by which classical feature they best match.
  2. For each classical-feature family, zero the SAE features in that
     bucket by subtracting their rank-1 decoder contributions from the
     residual stream, then forward through the classifier head and
     measure overall + per-class accuracy.
  3. Compare to a size-matched random ablation baseline (20 trials).
  4. Call a family "load-bearing" if its ablation hurts more than the
     matched-random baseline by >= 1 sigma.

Also stores the full post-ablation confusion matrix for each family so
we can see *which* modulations become confused. The hypothesis is that
ablating `phase_std` features specifically confuses phase-dependent
schemes (PSK vs FSK), ablating `env_var` features confuses PSK vs QAM,
and so on.

Run:
    python -m src.ablation
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from .classifier import ModClassifier
from .sae import TopKSAE


def load_classifier(ckpt_path: Path, device: str) -> ModClassifier:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = blob.get("config", {"n_classes": 11})
    model = ModClassifier(**cfg).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def load_sae(ckpt_path: Path, device: str):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    sae = TopKSAE(**blob["config"]).to(device)
    sae.load_state_dict(blob["state_dict"])
    sae.eval()
    mean = blob["mean"].to(device)
    return sae, mean


def classify_with_residual(model: ModClassifier, R: torch.Tensor) -> torch.Tensor:
    """Forward a precomputed residual-stream through the classifier head only."""
    with torch.no_grad():
        return model.head(R).argmax(-1)


def per_class_accuracy(preds: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros(n_classes)
    for c in range(n_classes):
        m = labels == c
        if m.any():
            out[c] = (preds[m] == c).mean()
    return out


def reconstruct_from(acts: torch.Tensor, W_dec: torch.Tensor,
                     feature_ids: list[int]) -> torch.Tensor:
    """Sum of rank-1 decoder contributions for the chosen features."""
    if not feature_ids:
        return torch.zeros(acts.shape[0], W_dec.shape[0], device=acts.device)
    idx = torch.tensor(feature_ids, device=acts.device)
    return acts[:, idx] @ W_dec[:, idx].T


def confusion_matrix(preds: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", default="results/classifier.pt")
    ap.add_argument("--sae", default="results/sae.pt")
    ap.add_argument("--activations", default="results/activations.npz")
    ap.add_argument("--analysis", default="results/sae_vs_classical.json",
                    help="Output of analyze_sae_vs_classical — gives us "
                         "each feature's best classical match.")
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--out", default="results/ablation.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    model = load_classifier(Path(args.classifier), args.device)
    sae, mean = load_sae(Path(args.sae), args.device)

    data = np.load(args.activations, allow_pickle=True)
    R_np = data["activations"]                 # [N, 256]
    feats_sae_np = data["sae_features"]        # [N, d_sae]
    classical_names = list(data["classical_names"])
    labels = data["labels"]
    classes = list(data["classes"])
    n_classes = len(classes)

    # Alive features and their best classical match
    analysis = json.loads(Path(args.analysis).read_text())
    alive_best_match = analysis["sae_stats"]["best_classical_per_feature"]
    alive = np.where(feats_sae_np.var(axis=0) > 1e-6)[0]
    assert len(alive) == len(alive_best_match), "alive count mismatch"

    # Bucket alive features by which classical feature is their best match
    by_family: dict[str, list[int]] = defaultdict(list)
    for local_idx, feat_idx in enumerate(alive):
        family = alive_best_match[local_idx]
        by_family[family].append(int(feat_idx))

    print(f"alive features: {len(alive)}")
    print(f"classical families with SAE features: {len(by_family)}")

    R = torch.from_numpy(R_np).to(args.device)
    with torch.no_grad():
        feats = sae.encode(R - mean)     # [N, d_sae]
    W_dec = sae.decoder.weight           # [d_in, d_sae]

    # Baseline
    base_preds = classify_with_residual(model, R).cpu().numpy()
    base_acc = float((base_preds == labels).mean())
    print(f"baseline accuracy: {base_acc:.4f}")

    results = {
        "baseline_accuracy": base_acc,
        "classes": classes,
        "classical_families": {},
        "confusion_matrix_baseline": confusion_matrix(base_preds, labels, n_classes).tolist(),
    }

    print("\n=== per-family ablation ===")
    header = f"  {'family':20s} {'n':>4s} {'abl_acc':>8s} {'rand_mean':>10s} {'rand_std':>9s}  ★"
    print(header)

    for family in sorted(by_family):
        ids = by_family[family]
        n = len(ids)
        rank1 = reconstruct_from(feats, W_dec, ids)
        R_ablated = R - rank1
        preds = classify_with_residual(model, R_ablated).cpu().numpy()
        acc = float((preds == labels).mean())
        pc_acc = per_class_accuracy(preds, labels, n_classes)
        cm = confusion_matrix(preds, labels, n_classes)

        # Matched random baseline — sample n random features from the alive set
        rand_accs = []
        for _ in range(args.n_random):
            rids = list(np.random.choice(alive, size=n, replace=False))
            r1 = reconstruct_from(feats, W_dec, rids)
            preds_r = classify_with_residual(model, R - r1).cpu().numpy()
            rand_accs.append(float((preds_r == labels).mean()))
        rand_accs = np.array(rand_accs)
        exceeds = bool(acc < rand_accs.mean() - rand_accs.std())

        results["classical_families"][family] = {
            "n_features": n,
            "ablated_accuracy": acc,
            "delta_vs_baseline": acc - base_acc,
            "random_mean": float(rand_accs.mean()),
            "random_std": float(rand_accs.std()),
            "random_min": float(rand_accs.min()),
            "exceeds_random": exceeds,
            "per_class_accuracy": {c: float(a) for c, a in zip(classes, pc_acc)},
            "confusion_matrix": cm.tolist(),
        }
        mark = "★" if exceeds else " "
        print(f"  {family:20s} {n:>4d} {acc:>8.4f} {rand_accs.mean():>10.4f} "
              f"{rand_accs.std():>9.4f}  {mark}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
