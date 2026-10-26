import numpy as np
import pandas as pd
import pytest
from anomaly_bench.data import validate_upload
from anomaly_bench.detectors import DETECTOR_NAMES, make_detector
from anomaly_bench.metrics import compute

def test_detectors_contract():
    x=np.sin(np.arange(80)/5)
    for name in DETECTOR_NAMES:
        d=make_detector(name).fit(x[:50]); assert d.score(x[50:]).shape == (30,)
def test_threshold_uses_past_only():
    m=compute(np.array([0,1]),np.array([.1,.9]),np.array([1.,2.,3.])); assert m["threshold"] < 3
def test_upload_validation():
    assert len(validate_upload(pd.DataFrame({"timestamp":pd.date_range("2024",periods=30),"value":range(30)})))==30
    with pytest.raises(ValueError): validate_upload(pd.DataFrame({"x":[1]}))
