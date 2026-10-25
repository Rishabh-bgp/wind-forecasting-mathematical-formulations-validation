"""Train a SMALL CNN-BiLSTM-Attention on the synthetic CSV and export weights + demo data
to demo/model.json for the in-browser (pure JS) live demo."""
import json, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd, torch
from windforecast.data import WindowedData, clean
from windforecast.models import CNNBiLSTMAttention, train_net, predict_net
from windforecast.metrics import all_metrics

SMALL = dict(filters=16, kernel_size=3, lstm_units=16, attn_dim=16)


def to_list(t): return t.detach().cpu().numpy().astype(float).round(7).tolist()


def export(model, wd):
    sd = {k: v for k, v in model.state_dict().items()}
    lstm = lambda suf: {"w_ih": to_list(sd["bilstm.weight_ih_l0" + suf]), "w_hh": to_list(sd["bilstm.weight_hh_l0" + suf]),
                        "b_ih": to_list(sd["bilstm.bias_ih_l0" + suf]), "b_hh": to_list(sd["bilstm.bias_hh_l0" + suf])}
    return {"conv_w": to_list(sd["conv.weight"]), "conv_b": to_list(sd["conv.bias"]),
            "lstm_fw": lstm(""), "lstm_bw": lstm("_reverse"),
            "attn_W": to_list(sd["attn.W.weight"]), "attn_b": to_list(sd["attn.W.bias"]),
            "attn_v": to_list(sd["attn.v.weight"]), "head_w": to_list(sd["head.weight"]), "head_b": to_list(sd["head.bias"]),
            "scaler_min": float(wd.scaler.data_min_[0]), "scaler_range": float(wd.scaler.data_range_[0]),
            "target": wd.target}


def main(epochs=15, hours=4000):
    torch.set_num_threads(2)
    df = pd.read_csv(os.path.join(ROOT, "data", "synthetic", "synthetic_scada.csv"), parse_dates=["timestamp"])
    s = clean(df).iloc[:hours]
    wd = WindowedData(s, 24, 1, 0.8, target="delta")
    model = CNNBiLSTMAttention(use_attention=True, **SMALL)
    cfg = {"lr": 1e-3, "batch_size": 32, "epochs": epochs, "patience": 5, "val_frac": 0.1}
    model, _ = train_net(model, wd.X_train, wd.y_train, cfg, seed=42)
    p = wd.to_level(predict_net(model, wd.X_test), wd.X_test); y = wd.inverse(wd.y_test_level)
    pers = wd.inverse(wd.X_test[:, -1])
    m = export(model, wd)
    n = 336  # two weeks of test hours for the demo chart
    m["demo"] = {"timestamps": [str(t) for t in wd.test_index[:n]],
                 "history": s.values[len(wd.train_series) - 24: len(wd.train_series) + n].round(3).tolist(),
                 "actual": y[:n].round(3).tolist(), "python_pred": p[:n].round(4).tolist(), "persistence": pers[:n].round(3).tolist(),
                 "metrics_model": all_metrics(y, p), "metrics_persistence": all_metrics(y, pers),
                 "note": f"Small demo model ({SMALL}), {epochs} epochs max, first {hours} h of SYNTHETIC data. Not the paper's model or data."}
    os.makedirs(os.path.join(ROOT, "demo"), exist_ok=True)
    json.dump(m, open(os.path.join(ROOT, "demo", "model.json"), "w"))
    print("model:", m["demo"]["metrics_model"]); print("persistence:", m["demo"]["metrics_persistence"])


if __name__ == "__main__":
    main()
