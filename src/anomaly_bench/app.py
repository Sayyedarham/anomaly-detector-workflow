"""Interactive UI deliberately reads precomputed artifacts rather than recomputing benchmarks."""
from pathlib import Path
import pandas as pd
import streamlit as st
from anomaly_bench.data import validate_upload
from anomaly_bench.detectors import DETECTOR_NAMES, make_detector

st.set_page_config(page_title="Anomaly Bench", page_icon="📈", layout="wide")
ROOT = Path(__file__).parents[2]
ART = ROOT / "artifacts"

@st.cache_data
def results() -> pd.DataFrame:
    path = ART / "results.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

page = st.sidebar.radio("Page", ["Overview", "Dataset Explorer", "Detector Comparison", "Retraining Study", "Try Your Own", "Methodology"])
df = results()
if page == "Overview":
    st.title("Benchmarking anomaly detectors under drift")
    st.caption("Leakage-safe rolling origins; offline synthetic fallback data.")
    if df.empty: st.info("Run `make benchmark` to generate results.")
    else: st.dataframe(df.groupby(["detector", "regime", "cadence"]).pr_auc.mean().sort_values(ascending=False).reset_index(), use_container_width=True)
elif page in ["Dataset Explorer", "Detector Comparison", "Retraining Study"]:
    st.title(page)
    if df.empty: st.info("No cached results yet.")
    elif page == "Dataset Explorer":
        chosen = st.selectbox("Dataset", sorted(df.dataset.unique()))
        st.line_chart(df[df.dataset == chosen].set_index("window_start")[["pr_auc", "f1"]])
    else: st.dataframe(df.groupby(["detector", "regime", "cadence"])[["pr_auc", "f1", "runtime_s", "memory_mb"]].mean(), use_container_width=True)
elif page == "Try Your Own":
    st.title("Try Your Own")
    uploaded = st.file_uploader("CSV with timestamp,value", type="csv")
    if uploaded:
        try:
            x = validate_upload(pd.read_csv(uploaded)); name = st.selectbox("Detector", DETECTOR_NAMES)
            detector = make_detector(name).fit(x.value.to_numpy()[:max(20, len(x)//2)])
            x["score"] = detector.score(x.value.to_numpy())
            st.dataframe(x); st.download_button("Download scores", x.to_csv(index=False), "scored.csv")
        except (ValueError, pd.errors.ParserError) as error: st.error(str(error))
else:
    st.title("Methodology")
    st.markdown("Each origin fits only on its preceding prefix. Thresholds use past scores only. KS tests on historical adjacent windows mark drift. Metrics are point-level; synthetic fallback data is clearly labelled.")
