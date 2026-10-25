"""Residual diagnostics and Diebold-Mariano test."""
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
import statsmodels.api as sm


def diebold_mariano(y, p1, p2, h: int = 1, power: int = 2, hln: bool = True):
    """DM test (Diebold & Mariano, 1995) with optional Harvey-Leybourne-Newbold
    small-sample correction. Negative statistic => model 1 more accurate."""
    y, p1, p2 = map(lambda a: np.asarray(a, float), (y, p1, p2))
    d = np.abs(y - p1) ** power - np.abs(y - p2) ** power
    n, dbar = len(d), d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    dm = dbar / np.sqrt(var)
    if hln:
        dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        pval = 2 * stats.t.sf(abs(dm), df=n - 1)
    else:
        pval = 2 * stats.norm.sf(abs(dm))
    return float(dm), float(pval)


def residual_diagnostics(y, p, lb_lag: int = 24, max_shapiro: int = 5000):
    y, p = np.asarray(y, float), np.asarray(p, float)
    r = y - p
    lb = acorr_ljungbox(r, lags=[lb_lag], return_df=True)
    sw = stats.shapiro(r[:max_shapiro])
    X = sm.add_constant(p)
    bp = het_breuschpagan(r, X)
    return {"ljung_box_lag": lb_lag, "ljung_box_p": float(lb["lb_pvalue"].iloc[0]),
            "shapiro_W": float(sw.statistic), "shapiro_p": float(sw.pvalue),
            "breusch_pagan_LM": float(bp[0]), "breusch_pagan_p": float(bp[1])}


def ljung_box_curve(r, max_lag: int = 24):
    lb = acorr_ljungbox(np.asarray(r, float), lags=list(range(1, max_lag + 1)), return_df=True)
    return lb["lb_pvalue"].tolist()
