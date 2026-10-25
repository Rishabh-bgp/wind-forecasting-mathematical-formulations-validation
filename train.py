"""Train + evaluate all models. Usage: python train.py [--config ...] [--quick]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import argparse
from windforecast.experiment import run, load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="few epochs on a short series (smoke test)")
    a = ap.parse_args()
    run(load_config(a.config), a.out, a.quick)
