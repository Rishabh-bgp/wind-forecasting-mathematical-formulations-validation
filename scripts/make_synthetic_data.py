"""Write a synthetic SCADA-style hourly CSV to data/synthetic/synthetic_scada.csv."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from windforecast.data import generate_synthetic_scada
os.makedirs("data/synthetic", exist_ok=True)
generate_synthetic_scada().to_csv("data/synthetic/synthetic_scada.csv", index=False)
print("wrote data/synthetic/synthetic_scada.csv")
