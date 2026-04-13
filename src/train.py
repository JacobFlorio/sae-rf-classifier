"""Train the 11-class modulation classifier on synthetic IQ data.

Training uses a mix of SNR levels (5, 10, 15 dB) so the classifier is
robust to noise. Evaluation reports per-class accuracy at each SNR
level on a held-out test set. The resulting checkpoint is the
foundation for the SAE interpretability experiment.

Run:
    python -m src.train
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .classifier import ModClassifier
from .synth_data import generate, CLASSES


def make_mixed_snr(n_per_class: int, snrs: list[float], seed: int):
    """Concatenate synthetic datasets at multiple SNR levels."""
    Xs, ys = [], []
    for i, snr in enumerate(snrs):
        X, y = generate(n_per_class=n_per_class, snr_db=snr, seed=seed + 1000 * i)
        Xs.append(X)
        ys.append(y)
    return torch.cat(Xs, dim=0), torch.cat(ys, dim=0)


def accuracy(model, X, y, batch: int = 256) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i:i + batch].to(next(model.parameters()).device)
            yb = y[i:i + batch].to(xb.device)
            correct += (model(xb).argmax(-1) == yb).sum().item()
            total += len(yb)
    return correct / total


def per_class_accuracy(model, X, y, n_classes: int, batch: int = 256) -> np.ndarray:
    model.eval()
    correct = np.zeros(n_classes)
    total = np.zeros(n_classes)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i:i + batch].to(next(model.parameters()).device)
            pred = model(xb).argmax(-1).cpu().numpy()
            yb = y[i:i + batch].numpy()
            for c in range(n_classes):
                m = yb == c
                total[c] += m.sum()
                correct[c] += (pred[m] == c).sum()
    return correct / np.maximum(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-per-class", type=int, default=1024,
                    help="Training samples per class per SNR.")
    ap.add_argument("--n-test", type=int, default=256,
                    help="Test samples per class per SNR.")
    ap.add_argument("--train-snrs", default="5,10,15")
    ap.add_argument("--eval-snrs", default="0,5,10,15,20")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/classifier.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_snrs = [float(s) for s in args.train_snrs.split(",")]
    eval_snrs = [float(s) for s in args.eval_snrs.split(",")]

    print(f"generating train set: {args.n_per_class}/class × {len(train_snrs)} SNRs × {len(CLASSES)} classes")
    X_tr, y_tr = make_mixed_snr(args.n_per_class, train_snrs, seed=args.seed)
    print(f"  shape {tuple(X_tr.shape)}")

    model = ModClassifier(n_classes=len(CLASSES)).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=args.batch, shuffle=True)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(args.device), yb.to(args.device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        tr_acc = accuracy(model, X_tr, y_tr)
        print(f"epoch {epoch:2d}  loss {np.mean(losses):.4f}  train {tr_acc:.3f}")

    print("\n=== held-out evaluation ===")
    eval_results = {"train_snrs": train_snrs, "eval_snrs": eval_snrs,
                    "classes": CLASSES, "per_snr": {}}
    for snr in eval_snrs:
        X_te, y_te = generate(n_per_class=args.n_test, snr_db=snr,
                              seed=args.seed + 99999 + int(snr * 7))
        overall = accuracy(model, X_te, y_te)
        per_class = per_class_accuracy(model, X_te, y_te, len(CLASSES))
        eval_results["per_snr"][str(snr)] = {
            "overall": float(overall),
            "per_class": {c: float(a) for c, a in zip(CLASSES, per_class)},
        }
        worst = CLASSES[int(per_class.argmin())]
        print(f"  SNR {snr:5.1f} dB  overall {overall:.3f}  worst {worst}={per_class.min():.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"n_classes": len(CLASSES)},
        "eval": eval_results,
    }, out)
    (out.with_suffix(".json")).write_text(json.dumps(eval_results, indent=2))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
