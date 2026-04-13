"""Compare SAE features to classical modulation-recognition features.

Question: when a TopK SAE is trained on the penultimate activations of a
CNN modulation classifier, do its learned features correspond to classical
features (cumulants, envelope stats, spectral moments) — or are they just
PCA directions in disguise?

The comparison:
  1. For each alive SAE feature, compute Pearson |r| with each classical
     feature. Record the max correlation (the "best classical match").
  2. Run the same analysis on the top-K PCA directions of the same
     activation matrix. This is the baseline: PCA is what any linear
     model could find; the SAE is only interesting if it beats it.
  3. Headline numbers:
        - mean max-correlation across alive SAE features
        - same for PCA
        - count of features with |r| > 0.7 for exactly one classical feature
          (high recovery, high specificity)

Additionally we run a linear probe on SAE features alone and on PCA
features alone: how much class information does each basis carry?
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def pearson_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """|r| matrix: rows index A's columns, cols index B's columns."""
    A = (A - A.mean(0)) / np.maximum(A.std(0), 1e-12)
    B = (B - B.mean(0)) / np.maximum(B.std(0), 1e-12)
    return np.abs(A.T @ B) / A.shape[0]


def linear_probe(X: np.ndarray, y: np.ndarray) -> dict:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    pipe = Pipeline([
        ("scale", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(max_iter=4000)),
    ])
    pipe.fit(X_tr, y_tr)
    return {
        "train_acc": float(pipe.score(X_tr, y_tr)),
        "test_acc": float(pipe.score(X_te, y_te)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default="results/activations.npz")
    ap.add_argument("--out", default="results/sae_vs_classical.json")
    ap.add_argument("--alive-threshold", type=float, default=1e-6)
    ap.add_argument("--strong-r", type=float, default=0.7)
    args = ap.parse_args()

    data = np.load(args.activations, allow_pickle=True)
    acts = data["activations"]                   # [N, 256]
    feats_sae = data["sae_features"]             # [N, d_sae]
    feats_class = data["classical_features"]     # [N, n_classical]
    names_class = list(data["classical_names"])
    labels = data["labels"]
    classes = list(data["classes"])

    # Alive SAE features
    per_feat_var = feats_sae.var(axis=0)
    alive = np.where(per_feat_var > args.alive_threshold)[0]
    print(f"alive SAE features: {len(alive)}/{feats_sae.shape[1]}")

    F_sae = feats_sae[:, alive]
    R_sae = pearson_matrix(F_sae, feats_class)  # [alive, 16]

    # PCA baseline
    n_pca = len(alive)
    pca = PCA(n_components=n_pca).fit(acts)
    F_pca = pca.transform(acts)
    R_pca = pearson_matrix(F_pca, feats_class)

    # Headline stats
    def summary(R: np.ndarray, name: str):
        max_r = R.max(axis=1)
        best_idx = R.argmax(axis=1)
        strong = (R.max(axis=1) > args.strong_r).sum()
        one_hot = sum(
            1 for i in range(R.shape[0])
            if (R[i] > args.strong_r).sum() == 1
        )
        return {
            "n_features": int(R.shape[0]),
            "mean_max_r": float(max_r.mean()),
            "median_max_r": float(np.median(max_r)),
            "fraction_strong": float(strong / R.shape[0]),
            "fraction_cleanly_one_classical": float(one_hot / R.shape[0]),
            "best_classical_per_feature": [names_class[j] for j in best_idx],
            "max_r_per_feature": max_r.tolist(),
        }

    sae_stats = summary(R_sae, "SAE")
    pca_stats = summary(R_pca, "PCA")

    print("\n=== SAE vs PCA interpretability ===")
    print(f"  {'':20s} {'SAE':>10s} {'PCA':>10s}")
    for k in ["mean_max_r", "median_max_r", "fraction_strong",
              "fraction_cleanly_one_classical"]:
        print(f"  {k:20s} {sae_stats[k]:>10.3f} {pca_stats[k]:>10.3f}")

    # Histogram of "best classical match" across SAE features
    best_counts = {}
    for name in sae_stats["best_classical_per_feature"]:
        best_counts[name] = best_counts.get(name, 0) + 1
    print("\n=== which classical features did SAE features lock onto? ===")
    for name in sorted(best_counts, key=lambda n: -best_counts[n]):
        print(f"  {name:20s} {best_counts[name]:3d}")

    # Probes: how much class information does each basis carry?
    probe_sae = linear_probe(F_sae, labels)
    probe_pca = linear_probe(F_pca, labels)
    probe_class = linear_probe(feats_class, labels)
    probe_acts = linear_probe(acts, labels)

    print("\n=== linear probe (test accuracy) ===")
    print(f"  {'raw activations (256)':25s} {probe_acts['test_acc']:.3f}")
    print(f"  {'classical (16)':25s} {probe_class['test_acc']:.3f}")
    print(f"  {'SAE alive ({})'.format(len(alive)):25s} {probe_sae['test_acc']:.3f}")
    print(f"  {'PCA ({})'.format(n_pca):25s} {probe_pca['test_acc']:.3f}")

    out = Path(args.out)
    out.write_text(json.dumps({
        "n_classical": len(names_class),
        "classes": classes,
        "classical_names": names_class,
        "sae_stats": sae_stats,
        "pca_stats": pca_stats,
        "best_counts": best_counts,
        "probes": {
            "raw_activations": probe_acts,
            "classical": probe_class,
            "sae": probe_sae,
            "pca": probe_pca,
        },
        "R_sae": R_sae.tolist(),
        "R_pca": R_pca.tolist(),
    }, indent=2))
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
