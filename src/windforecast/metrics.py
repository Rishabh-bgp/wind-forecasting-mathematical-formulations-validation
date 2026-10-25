import numpy as np


def mae(y, p): return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def mse(y, p): return float(np.mean((y - p) ** 2))


def mape(y, p, eps: float = 0.1):
    """MAPE in %, ignoring targets with |y| < eps m/s (near-calm hours would
    blow MAPE up; the papers do not say how zeros were handled - assumption)."""
    m = np.abs(y) >= eps
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100)


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2))


def all_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE": mape(y, p), "R2": r2(y, p)}


def stratified_rmse(y, p, low: float = 5.0, high: float = 10.0):
    y, p = np.asarray(y), np.asarray(p)
    out = {}
    for name, m in {"low(<%g)" % low: y < low, "medium(%g-%g)" % (low, high): (y >= low) & (y <= high),
                    "high(>%g)" % high: y > high}.items():
        out[name] = rmse(y[m], p[m]) if m.sum() else float("nan")
    return out
