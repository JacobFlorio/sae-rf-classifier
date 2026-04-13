"""Classical modulation-recognition features.

These are the features pre-deep-learning modulation classifiers used.
The headline ones are higher-order cumulants (Swami & Sadler 2000,
"Hierarchical Digital Modulation Classification Using Cumulants"),
which take analytically known values for each digital modulation.

For a complex signal x with n samples, the raw moments are:
    M20 = E[x^2]
    M21 = E[|x|^2]      (= signal power)
    M22 = M21
    M40 = E[x^4]
    M41 = E[x^3 · x*]
    M42 = E[|x|^4]

And the cumulants (after subtracting the lower-order contributions):
    C20 = M20
    C21 = M21
    C40 = M40 - 3 M20^2
    C41 = M41 - 3 M20 M21
    C42 = M42 - |M20|^2 - 2 M21^2

The cumulants are scale-sensitive, so for classification we report
the magnitudes normalized by the appropriate power of C21, which
gives scale-invariant features. E.g. |C40|/|C21|^2 is canonical.

Beyond cumulants we also compute amplitude statistics (mean, std,
kurtosis of |x|), envelope variance, and spectral centroid/bandwidth —
features that show up in almost every classical modulation classifier.

All features return a fixed-length real-valued vector per sample.
"""
from __future__ import annotations
import numpy as np


FEATURE_NAMES = [
    "C20_mag", "C21", "C40_mag", "C41_mag", "C42",
    "C40_norm", "C41_norm", "C42_norm",
    "amp_mean", "amp_std", "amp_kurtosis",
    "env_var", "papr",
    "spec_centroid", "spec_bandwidth",
    "phase_std",
]


def iq_to_complex(iq: np.ndarray) -> np.ndarray:
    """Convert [B, 2, T] real IQ tensor to [B, T] complex array."""
    return iq[:, 0, :] + 1j * iq[:, 1, :]


def cumulants(x: np.ndarray) -> dict[str, np.ndarray]:
    """Per-sample higher-order cumulants of a complex signal.

    x: [B, T] complex
    returns dict of [B]-shaped arrays.
    """
    M20 = np.mean(x ** 2, axis=-1)
    M21 = np.mean(np.abs(x) ** 2, axis=-1).real
    M40 = np.mean(x ** 4, axis=-1)
    M41 = np.mean(x ** 3 * x.conj(), axis=-1)
    M42 = np.mean(np.abs(x) ** 4, axis=-1).real

    C20 = M20
    C21 = M21
    C40 = M40 - 3.0 * M20 ** 2
    C41 = M41 - 3.0 * M20 * M21
    C42 = M42 - np.abs(M20) ** 2 - 2.0 * M21 ** 2
    return {"C20": C20, "C21": C21, "C40": C40, "C41": C41, "C42": C42}


def amplitude_stats(x: np.ndarray) -> dict[str, np.ndarray]:
    """Amplitude moments: mean, std, kurtosis, envelope variance, PAPR."""
    a = np.abs(x)
    m = a.mean(axis=-1)
    s = a.std(axis=-1)
    mu4 = np.mean((a - m[:, None]) ** 4, axis=-1)
    kurt = mu4 / np.maximum(s ** 4, 1e-12)
    env_var = (a ** 2).var(axis=-1)
    peak = (a ** 2).max(axis=-1)
    avg = np.maximum((a ** 2).mean(axis=-1), 1e-12)
    papr = 10.0 * np.log10(peak / avg)
    return {"amp_mean": m, "amp_std": s, "amp_kurtosis": kurt,
            "env_var": env_var, "papr": papr}


def spectral_stats(x: np.ndarray) -> dict[str, np.ndarray]:
    """Spectral centroid and bandwidth in normalized-frequency units."""
    X = np.fft.fftshift(np.fft.fft(x, axis=-1), axes=-1)
    P = np.abs(X) ** 2
    T = x.shape[-1]
    f = np.linspace(-0.5, 0.5, T, endpoint=False)
    Psum = np.maximum(P.sum(axis=-1, keepdims=True), 1e-12)
    p = P / Psum
    centroid = (p * f[None, :]).sum(axis=-1)
    variance = (p * (f[None, :] - centroid[:, None]) ** 2).sum(axis=-1)
    bandwidth = np.sqrt(np.maximum(variance, 0))
    return {"spec_centroid": centroid, "spec_bandwidth": bandwidth}


def phase_stats(x: np.ndarray) -> dict[str, np.ndarray]:
    """Std of instantaneous phase differences — rough angular jitter metric."""
    phases = np.angle(x)
    dph = np.diff(phases, axis=-1)
    dph = (dph + np.pi) % (2 * np.pi) - np.pi
    return {"phase_std": dph.std(axis=-1)}


def extract(iq: np.ndarray) -> np.ndarray:
    """Run every classical feature extractor on a batch of IQ signals.

    iq: either [B, 2, T] real-valued IQ, or [B, T] complex.
    returns: [B, len(FEATURE_NAMES)] float64
    """
    if iq.ndim == 3:
        x = iq_to_complex(iq)
    else:
        x = iq.astype(np.complex128)

    c = cumulants(x)
    C21 = c["C21"].real
    C21_safe = np.maximum(np.abs(C21), 1e-12)

    feats = {
        "C20_mag": np.abs(c["C20"]),
        "C21": C21,
        "C40_mag": np.abs(c["C40"]),
        "C41_mag": np.abs(c["C41"]),
        "C42": c["C42"].real,
        "C40_norm": np.abs(c["C40"]) / C21_safe ** 2,
        "C41_norm": np.abs(c["C41"]) / C21_safe ** 2,
        "C42_norm": c["C42"].real / C21_safe ** 2,
    }
    feats.update(amplitude_stats(x))
    feats.update(spectral_stats(x))
    feats.update(phase_stats(x))

    out = np.stack([feats[name] for name in FEATURE_NAMES], axis=-1).astype(np.float64)
    return out


def linear_probe_score(X: np.ndarray, y: np.ndarray, test_frac: float = 0.2) -> dict:
    """Sanity check: can a logistic regression classify the 11 modulations
    using only these classical features?

    If this scores well above chance (~9%), the feature set has the
    discriminative information the SAE is eventually asked to recover.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_frac, random_state=0, stratify=y
    )
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, multi_class="auto")),
    ])
    pipe.fit(X_tr, y_tr)
    return {
        "train_acc": float(pipe.score(X_tr, y_tr)),
        "test_acc": float(pipe.score(X_te, y_te)),
    }
