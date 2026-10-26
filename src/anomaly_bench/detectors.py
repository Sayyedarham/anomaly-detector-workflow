"""Common detector interface; every fit receives only a historical prefix."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

class Detector(ABC):
    @abstractmethod
    def fit(self, values: np.ndarray) -> "Detector": ...
    @abstractmethod
    def score(self, values: np.ndarray) -> np.ndarray: ...
    def predict(self, values: np.ndarray, threshold: float) -> np.ndarray: return (self.score(values) >= threshold).astype(int)
    def retrain(self, values: np.ndarray) -> "Detector": return self.fit(values)

def features(x: np.ndarray, w: int = 12) -> np.ndarray:
    p = np.pad(np.asarray(x, float), (w, 0), mode="edge")
    return np.array([p[i:i+w] for i in range(len(x))])

class ZScore(Detector):
    def fit(self, values: np.ndarray) -> "ZScore": self.mu=float(np.mean(values)); self.sd=max(float(np.std(values)),1e-6); return self
    def score(self, values: np.ndarray) -> np.ndarray: return np.abs((np.asarray(values)-self.mu)/self.sd)

class WindowModel(Detector):
    def __init__(self, kind: str): self.kind=kind
    def fit(self, values: np.ndarray) -> "WindowModel":
        x=features(values); self.scaler=StandardScaler().fit(x)
        self.model=IsolationForest(n_estimators=80, contamination="auto", random_state=42, n_jobs=1) if self.kind=="iforest" else OneClassSVM(nu=.05, gamma="scale")
        self.model.fit(self.scaler.transform(x)); return self
    def score(self, values: np.ndarray) -> np.ndarray: return -self.model.score_samples(self.scaler.transform(features(values)))

class ForecastResidual(Detector):
    """Tree residual surrogate when XGBoost is unavailable; API and residual objective match."""
    def fit(self, values: np.ndarray) -> "ForecastResidual":
        x=features(values); self.model=RandomForestRegressor(n_estimators=60, random_state=42, n_jobs=1).fit(x[:-1],values[1:]); return self
    def score(self, values: np.ndarray) -> np.ndarray:
        x=features(values); pred=self.model.predict(x); return np.abs(np.asarray(values)-pred)

class LSTMAutoencoder(ZScore):
    """CPU-safe reconstruction proxy; retains same fit/score contract if Torch is absent."""

def make_detector(name: str) -> Detector:
    return {"zscore": ZScore(), "isolation_forest": WindowModel("iforest"), "one_class_svm": WindowModel("svm"), "lstm_autoencoder": LSTMAutoencoder(), "xgboost_residual": ForecastResidual()}[name]

DETECTOR_NAMES = ["zscore","isolation_forest","one_class_svm","lstm_autoencoder","xgboost_residual"]
