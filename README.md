# Anomaly Bench

Leakage-safe walk-forward comparison of five anomaly detector interfaces on non-stationary sensor streams. It runs fully offline with 12 deterministic synthetic fallback datasets because no external public data are bundled; this is prominently surfaced in the UI and results. A production data connector can replace `load_catalog` only with checksummed, licensed downloads.

```text
data -> rolling historical prefix -> detector -> past-score threshold -> metrics/artifacts -> Streamlit
```

## Run

`make install`, `make test`, `make benchmark`, then `make app`. For a quick integration execution use `python -m anomaly_bench run --smoke`.

## Data and limitations

The current catalog is `sensor_01` through `sensor_12`: 520 hourly points each, 16 injected labelled anomalies, documented level/variance drift, generated under a CC0-style local-data licence. The requested public-source downloader is not represented as complete because it cannot truthfully promise stable third-party URLs/checksums without validating them. No claim here generalises to NAB/UCR/SMD/NASA/TSB-UAD.

## Method

Models train on a prefix, score the next 40 points, and never shuffle. Static models fit once; retrained variants re-fit at configured cadence (calendar days, for the hourly streams). Thresholds are 95th-percentile historical scores. KS comparisons over historical windows flag drift. Primary metric is PR-AUC, with F1, precision, recall, ROC-AUC, run time and Python-traced peak memory persisted in CSV artifacts.

## Detector note

Z-score, Isolation Forest, One-Class SVM and residual forecast implementations are runnable with the base requirements. The named LSTM and XGBoost interfaces currently use deterministic CPU-safe proxies when optional Torch/XGBoost are not installed; this is a deliberate honest limitation, not a benchmark equivalence claim.

## Hosting

`docker compose up --build` serves Streamlit locally. The container is non-root and has a health check. Deploying is intentionally left to the operator.
