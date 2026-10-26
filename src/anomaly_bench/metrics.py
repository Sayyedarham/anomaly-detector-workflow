from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support

def threshold_from_past(scores: np.ndarray) -> float: return float(np.quantile(scores, .95))
def compute(y: np.ndarray, score: np.ndarray, past_scores: np.ndarray) -> dict[str,float]:
    pred=(score>=threshold_from_past(past_scores)).astype(int); p,r,f,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
    # A window without both classes cannot identify ranking quality; retain it as missing.
    auc = float(average_precision_score(y,score)) if len(np.unique(y)) > 1 else float("nan")
    return {"pr_auc":auc,"roc_auc":float(roc_auc_score(y,score)) if len(np.unique(y))>1 else float("nan"),"precision":float(p),"recall":float(r),"f1":float(f),"threshold":threshold_from_past(past_scores)}
