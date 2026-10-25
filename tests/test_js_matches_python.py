import json, os, shutil, subprocess, sys, tempfile
import numpy as np, pytest, torch
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from windforecast.models import CNNBiLSTMAttention

MODEL = os.path.join(ROOT, "demo", "model.json")


def load_torch(m):
    F, K, H = len(m["conv_w"]), len(m["conv_w"][0][0]), len(m["lstm_fw"]["w_hh"][0])
    net = CNNBiLSTMAttention(filters=F, kernel_size=K, lstm_units=H, attn_dim=len(m["attn_W"]))
    T = lambda a: torch.tensor(np.array(a), dtype=torch.float32)
    sd = {"conv.weight": T(m["conv_w"]), "conv.bias": T(m["conv_b"]), "attn.W.weight": T(m["attn_W"]),
          "attn.W.bias": T(m["attn_b"]), "attn.v.weight": T(m["attn_v"]), "head.weight": T(m["head_w"]), "head.bias": T(m["head_b"])}
    for suf, d in (("", "lstm_fw"), ("_reverse", "lstm_bw")):
        for k in ("w_ih", "w_hh", "b_ih", "b_hh"):
            name = {"w_ih": "weight_ih", "w_hh": "weight_hh", "b_ih": "bias_ih", "b_hh": "bias_hh"}[k]
            sd[f"bilstm.{name}_l0{suf}"] = T(m[d][k])
    net.load_state_dict(sd); net.eval()
    return net


@pytest.mark.skipif(shutil.which("node") is None or not os.path.exists(MODEL), reason="needs node and demo/model.json")
def test_js_and_python_predictions_match():
    m = json.load(open(MODEL))
    rng = np.random.default_rng(0)
    windows = rng.uniform(2, 14, size=(20, 24)).round(3)
    net = load_torch(m)
    x = (windows - m["scaler_min"]) / m["scaler_range"]
    with torch.no_grad():
        out = net(torch.tensor(x, dtype=torch.float32)).numpy()
        if m.get("target") == "delta":
            out = out + x[:, -1]
        py = out * m["scaler_range"] + m["scaler_min"]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(windows.tolist(), f)
    out = subprocess.run(["node", os.path.join(ROOT, "demo", "cli.js"), MODEL, f.name], capture_output=True, text=True, check=True)
    js = np.array(json.loads(out.stdout))
    assert np.allclose(js, py, atol=1e-3), np.abs(js - py).max()
