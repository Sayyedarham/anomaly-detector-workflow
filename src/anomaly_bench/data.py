"""Deterministic, labelled non-stationary sensor datasets.

Network datasets are intentionally optional: the benchmark is immediately reproducible
offline with these documented synthetic fallbacks, rather than silently changing data.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class SeriesData:
    name: str
    frame: pd.DataFrame
    source: str = "Synthetic fallback (offline reproducible)"
    licence: str = "CC0-style generated data"

def synthetic_series(name: str, n: int = 520, seed: int = 42) -> SeriesData:
    """Generate a drifted sensor stream and point labels from a fixed seed."""
    salt = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed + salt)
    t = np.arange(n); drift = n // 2 + int(rng.integers(-40, 40))
    value = (np.sin(t / (14 + salt % 11)) + .003 * t + rng.normal(0, .12, n))
    value[drift:] += 1.1 + .005 * (t[drift:] - drift)
    value[drift:] += rng.normal(0, .20, n - drift)
    label = np.zeros(n, dtype=int)
    positions = rng.choice(np.r_[np.arange(60, drift-10), np.arange(drift+10, n-5)], 16, replace=False)
    label[positions] = 1; value[positions] += rng.choice([-1, 1], 16) * rng.uniform(2.5, 4.0, 16)
    return SeriesData(name, pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="h"), "value": value, "label": label, "drift": (t >= drift).astype(int)}))

def load_catalog(count: int = 12, seed: int = 42) -> list[SeriesData]:
    return [synthetic_series(f"sensor_{i:02d}", seed=seed) for i in range(1, count + 1)]

def validate_upload(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "value"}
    if not required.issubset(frame.columns): raise ValueError("CSV must contain timestamp and value columns.")
    out = frame[["timestamp", "value"]].copy(); out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce"); out = out.dropna().sort_values("timestamp")
    if len(out) < 30: raise ValueError("At least 30 valid rows are required.")
    if not out["timestamp"].is_monotonic_increasing: raise ValueError("Timestamps must be sortable.")
    return out.reset_index(drop=True)
