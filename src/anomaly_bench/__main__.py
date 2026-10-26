from __future__ import annotations
import argparse
from .pipeline import run

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("command",choices=["run","report","app"]); p.add_argument("--smoke",action="store_true"); a=p.parse_args()
    if a.command=="run": run(datasets=2 if a.smoke else 12, cadences=[7] if a.smoke else None)
    elif a.command=="app":
        import os; os.system("streamlit run src/anomaly_bench/app.py")
    else: print("artifacts/RESULTS.generated.md")
if __name__=="__main__": main()
