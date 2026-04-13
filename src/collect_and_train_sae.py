"""Collect penultimate-layer activations from the classifier and train
a TopK SAE on them. Saves activations + classical features together so
downstream analysis can align them without re-running anything.

Run:
    python -m src.collect_and_train_sae
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .classifier import ModClassifier
from .classical_features import extract as classical_extract, FEATURE_NAMES
from .sae import TopKSAE
from .synth_data import generate, CLASSES


def load_classifier(ckpt_path: Path, device: str) -> ModClassifier:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        cfg = blob.get("config", {"n_classes": 11})
        model = ModClassifier(**cfg).to(device)
        model.load_state_dict(blob["state_dict"])
    else:
        model = ModClassifier(n_classes=11).to(device)
        model.load_state_dict(blob)
    model.eval()
    return model


@torch.no_grad()
def collect_activations(model: ModClassifier, X: torch.Tensor, batch: int, device: str):
    chunks = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(device)
        chunks.append(model.residual_stream(xb).cpu())
    return torch.cat(chunks, dim=0)


def train_sae(X: torch.Tensor, d_sae: int, k: int, epochs: int,
              lr: float, device: str):
    sae = TopKSAE(d_in=X.shape[1], d_sae=d_sae, k=k).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=lr)
    mean = X.mean(dim=0, keepdim=True)
    Xc = (X - mean).to(device)
    base_var = (Xc ** 2).mean().item()

    for step in range(epochs + 1):
        loss, acts = sae.loss(Xc)
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            W = sae.decoder.weight
            W.div_(W.norm(dim=0, keepdim=True).clamp(min=1e-8))
        opt.step()
        if step % 200 == 0:
            var_explained = 1.0 - loss.item() / max(base_var, 1e-12)
            dead = (acts.abs().sum(dim=0) == 0).sum().item()
            print(f"  step {step:5d}  mse {loss.item():.5f}  "
                  f"var_explained {var_explained:.3f}  dead {dead}/{d_sae}")
    return sae, mean.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", default="results/classifier.pt")
    ap.add_argument("--n-per-class", type=int, default=1024)
    ap.add_argument("--eval-snr", type=float, default=10.0)
    ap.add_argument("--d-sae", type=int, default=128)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out-sae", default="results/sae.pt")
    ap.add_argument("--out-activations", default="results/activations.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"loading classifier from {args.classifier}")
    model = load_classifier(Path(args.classifier), args.device)

    print(f"generating {args.n_per_class}/class @ SNR {args.eval_snr} dB")
    X_iq, y = generate(n_per_class=args.n_per_class, snr_db=args.eval_snr,
                       seed=args.seed + 123)
    print(f"  iq shape {tuple(X_iq.shape)}")

    print("collecting penultimate activations")
    acts = collect_activations(model, X_iq, args.batch, args.device)
    print(f"  activation shape {tuple(acts.shape)}  mean {acts.mean():.3f}  std {acts.std():.3f}")

    print("computing classical features")
    feats = classical_extract(X_iq.numpy())
    print(f"  classical feature shape {feats.shape}")

    print(f"training TopK SAE (d_sae={args.d_sae}, k={args.k}, epochs={args.epochs})")
    sae, mean = train_sae(acts, args.d_sae, args.k, args.epochs, args.lr, args.device)

    with torch.no_grad():
        feats_sae = sae.encode((acts - mean).to(args.device)).cpu().numpy()
    print(f"  SAE feature shape {feats_sae.shape}")

    out_sae = Path(args.out_sae)
    out_sae.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": sae.state_dict(),
        "config": {"d_in": acts.shape[1], "d_sae": args.d_sae, "k": args.k},
        "mean": mean,
    }, out_sae)
    print(f"saved → {out_sae}")

    out_acts = Path(args.out_activations)
    np.savez_compressed(
        out_acts,
        activations=acts.numpy(),
        sae_features=feats_sae,
        classical_features=feats,
        classical_names=np.array(FEATURE_NAMES),
        labels=y.numpy(),
        classes=np.array(CLASSES),
    )
    print(f"saved → {out_acts}")


if __name__ == "__main__":
    main()
