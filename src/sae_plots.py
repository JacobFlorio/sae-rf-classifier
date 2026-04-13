"""Headline plots for the SAE-vs-classical interpretability experiment.

Produces:
  - correlation_heatmap.png   rows=alive SAE features, cols=classical,
                              cells = |Pearson r|. Features sorted by
                              best-classical-match for readability.
  - max_r_histogram.png       histogram of max |r| across alive SAE
                              features, overlaid with the same for PCA.
  - per_class_firing.png      heatmap of mean SAE feature activation
                              per modulation class (which features
                              fire for which mods).
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def correlation_heatmap(data_npz, analysis_json, out: Path):
    data = np.load(data_npz, allow_pickle=True)
    feats_sae = data["sae_features"]
    feats_class = data["classical_features"]
    names_class = list(data["classical_names"])

    alive = np.where(feats_sae.var(axis=0) > 1e-6)[0]
    F_sae = feats_sae[:, alive]

    A = (F_sae - F_sae.mean(0)) / np.maximum(F_sae.std(0), 1e-12)
    B = (feats_class - feats_class.mean(0)) / np.maximum(feats_class.std(0), 1e-12)
    R = np.abs(A.T @ B) / A.shape[0]

    # Sort features by which classical they match best, then by strength
    best = R.argmax(axis=1)
    strength = R.max(axis=1)
    order = np.lexsort((-strength, best))
    R_sorted = R[order]

    fig, ax = plt.subplots(figsize=(7, max(5, 0.18 * len(alive))))
    im = ax.imshow(R_sorted, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(names_class)))
    ax.set_xticklabels(names_class, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([])
    ax.set_ylabel(f"alive SAE features ({len(alive)}, sorted by best match)")
    ax.set_xlabel("classical feature")
    ax.set_title("|Pearson r|  —  SAE features vs classical features")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("|r|")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved → {out}")


def max_r_histogram(analysis_json, out: Path):
    data = json.loads(Path(analysis_json).read_text())
    sae = np.array(data["sae_stats"]["max_r_per_feature"])
    pca = np.array(data["pca_stats"]["max_r_per_feature"])

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(pca, bins=bins, alpha=0.55, color="#888",
            label=f"PCA (mean {pca.mean():.2f})", edgecolor="#222")
    ax.hist(sae, bins=bins, alpha=0.65, color="#2ca02c",
            label=f"SAE (mean {sae.mean():.2f})", edgecolor="#222")
    ax.axvline(0.7, ls="--", color="#d62728", alpha=0.6,
               label="|r|=0.7 threshold")
    ax.set_xlabel("max |Pearson r| with any classical feature")
    ax.set_ylabel("# features")
    ax.set_title("Per-feature interpretability: SAE vs PCA")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved → {out}")


def per_class_firing(data_npz, out: Path):
    data = np.load(data_npz, allow_pickle=True)
    feats_sae = data["sae_features"]
    labels = data["labels"]
    classes = list(data["classes"])

    alive = np.where(feats_sae.var(axis=0) > 1e-6)[0]
    F = feats_sae[:, alive]

    per_class = np.zeros((len(classes), len(alive)))
    for c in range(len(classes)):
        m = labels == c
        if m.sum() == 0:
            continue
        per_class[c] = F[m].mean(axis=0)
    # Normalize per feature to [0, 1] so small-magnitude features are visible
    pc_norm = per_class / np.maximum(per_class.max(axis=0, keepdims=True), 1e-12)

    # Sort features by which class they fire most for, for a cleaner display
    top_class = per_class.argmax(axis=0)
    order = np.argsort(top_class)
    pc_sorted = pc_norm[:, order]

    fig, ax = plt.subplots(figsize=(max(7, 0.16 * len(alive)), 4))
    im = ax.imshow(pc_sorted, aspect="auto", cmap="magma")
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel(f"alive SAE features ({len(alive)}, sorted by preferred class)")
    ax.set_title("Per-modulation mean SAE feature firing (column-normalized)")
    fig.colorbar(im, ax=ax, label="normalized activation")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved → {out}")


def main():
    correlation_heatmap("results/activations.npz",
                        "results/sae_vs_classical.json",
                        Path("results/correlation_heatmap.png"))
    max_r_histogram("results/sae_vs_classical.json",
                    Path("results/max_r_histogram.png"))
    per_class_firing("results/activations.npz",
                     Path("results/per_class_firing.png"))


if __name__ == "__main__":
    main()
