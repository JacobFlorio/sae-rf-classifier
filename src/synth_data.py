"""Synthetic IQ data for 11 modulation classes.

Used as a stand-in for RadioML 2018.01A so the pipeline is runnable
without a 20 GB download. Real data will be wired in for the final run.
"""
from __future__ import annotations
import numpy as np
import torch

CLASSES = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "GFSK", "CPFSK", "PAM4", "WBFM", "AM-SSB", "AM-DSB"]


def _psk(n: int, m: int, rng):
    symbols = rng.integers(0, m, size=n)
    phases = 2 * np.pi * symbols / m
    return np.exp(1j * phases)


def _qam(n: int, m: int, rng):
    side = int(np.sqrt(m))
    i = rng.integers(0, side, size=n) * 2 - (side - 1)
    q = rng.integers(0, side, size=n) * 2 - (side - 1)
    return (i + 1j * q) / (side - 1)


def generate(n_per_class: int = 512, n_samples: int = 1024, snr_db: float = 10.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    sigma = 10 ** (-snr_db / 20)
    for idx, name in enumerate(CLASSES):
        for _ in range(n_per_class):
            if name == "BPSK":
                s = _psk(n_samples, 2, rng)
            elif name == "QPSK":
                s = _psk(n_samples, 4, rng)
            elif name == "8PSK":
                s = _psk(n_samples, 8, rng)
            elif name == "QAM16":
                s = _qam(n_samples, 16, rng)
            elif name == "QAM64":
                s = _qam(n_samples, 64, rng)
            elif name == "PAM4":
                levels = rng.integers(0, 4, size=n_samples) * 2 - 3
                s = levels.astype(complex) / 3
            elif name == "GFSK":
                bits = rng.integers(0, 2, size=n_samples) * 2 - 1
                phase = np.cumsum(bits) * 0.1
                s = np.exp(1j * phase)
            elif name == "CPFSK":
                bits = rng.integers(0, 2, size=n_samples) * 2 - 1
                phase = np.cumsum(bits) * 0.2
                s = np.exp(1j * phase)
            elif name == "WBFM":
                t = np.arange(n_samples) / n_samples
                msg = np.sin(2 * np.pi * 5 * t) + 0.3 * np.sin(2 * np.pi * 13 * t)
                phase = np.cumsum(msg) * 0.5
                s = np.exp(1j * phase)
            elif name == "AM-SSB":
                t = np.arange(n_samples) / n_samples
                msg = np.sin(2 * np.pi * 7 * t)
                s = msg.astype(complex)
            else:  # AM-DSB
                t = np.arange(n_samples) / n_samples
                carrier = np.exp(1j * 2 * np.pi * 0.25 * np.arange(n_samples))
                msg = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)
                s = msg * carrier
            noise = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) * sigma / np.sqrt(2)
            s = s + noise
            iq = np.stack([s.real, s.imag], axis=0).astype(np.float32)
            X.append(iq)
            y.append(idx)
    return torch.tensor(np.array(X)), torch.tensor(y, dtype=torch.long)
