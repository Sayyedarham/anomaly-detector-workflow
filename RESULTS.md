# Results

These values were generated locally on the deterministic 12-stream synthetic fallback catalog with `python -m anomaly_bench run`.

- Best mean window PR-AUC: **0.816** — residual forecast proxy, retrained every 7 days.
- Cadence-7 versus static: paired Wilcoxon across 12 streams, **p=0.0009766**, statistic 1.

The cadence hypothesis holds on this synthetic smoke benchmark. It is not a claim about NAB, UCR, SMD, NASA or TSB-UAD: those public data have not been downloaded, checksummed and executed here. Re-run the benchmark after adding a verified public-data connector to replace this report.
