import argparse
from .experiment import run, load_config

ap = argparse.ArgumentParser(description="Train and evaluate wind speed forecasting models")
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--out", default="results")
ap.add_argument("--quick", action="store_true", help="few epochs, smoke test")
a = ap.parse_args()
run(load_config(a.config), a.out, a.quick)
