"""End-to-end experiment runner: data -> models -> metrics -> statistical tests."""
from __future__ import annotations
import json
import os
import time
import numpy as np
import pandas as pd
import yaml

from . import data as D, metrics as M, stats as S, models as Mo

DEEP = {"cnn_bilstm_attention", "cnn_bilstm", "lstm", "ann"}
LABELS = {"cnn_bilstm_attention": "CNN-BiLSTM-Attention", "cnn_bilstm": "CNN-BiLSTM",
          "lstm": "LSTM", "ann": "ANN", "svr": "SVR", "svm": "SVM (SVR-RBF)",
          "rf": "Random Forest", "xgb": "XGBoost", "arima": "ARIMA", "arima_ann": "ARIMA-ANN", "persistence": "Persistence"}


def load_series(cfg: dict) -> pd.Series:
    dc = cfg["data"]
    if dc.get("csv") and os.path.exists(dc["csv"]):
        df = D.load_scada_csv(dc["csv"], dc.get("timestamp_col", "timestamp"),
                              dc.get("wind_col", "wind_speed"), dc.get("resample", "1h"),
                              dc.get("dayfirst", False))
        print(f"Loaded {len(df)} rows from {dc['csv']}")
    else:
        if dc.get("csv"):
            print(f"[warn] {dc['csv']} not found -> falling back to synthetic SCADA-style data")
        df = D.generate_synthetic_scada(dc.get("n_hours", 8760), seed=cfg.get("seed", 42))
        print(f"Generated synthetic SCADA-style series: {len(df)} hourly rows")
    return D.clean(df)


def run(cfg: dict, out_dir: str = "results", quick: bool = False) -> pd.DataFrame:
    seed = cfg.get("seed", 42)
    tc = dict(cfg["training"])
    if quick:
        tc["epochs"] = tc.get("quick_epochs", 3)
        # smoke-test mode: tiny grids so the whole pipeline runs in minutes
        tc.update(cv_folds=2, rf_trees=50, svr_grid={"C": [10], "gamma": ["scale"], "epsilon": [0.05]},
                  xgb_grid={"n_estimators": [100], "max_depth": [4], "learning_rate": [0.1]})
    s = load_series(cfg)
    if quick and cfg["data"].get("quick_hours"):
        s = s.iloc[: cfg["data"]["quick_hours"]]
    w = cfg["window"]
    wd = D.WindowedData(s, w, cfg.get("horizon", 1), cfg.get("train_frac", 0.8), cfg.get("target", "delta"))
    y_true = wd.inverse(wd.y_test_level)
    train_raw, test_raw = wd.train_series.values, wd.test_series.values[: len(y_true)]
    os.makedirs(out_dir, exist_ok=True)
    preds, hists, rows = {}, {}, []
    order = None
    names = list(cfg["models"]) + (["persistence"] if "persistence" not in cfg["models"] else [])
    for name in names:
        t0 = time.time()
        print(f"Training {LABELS.get(name, name)} ...")
        if name == "persistence":  # naive baseline: next hour = last observed hour
            p = wd.inverse(wd.X_test[:, -1])
        elif name in DEEP:
            net = Mo.build_net(name, w, tc)
            net, hist = Mo.train_net(net, wd.X_train, wd.y_train, tc, seed)
            hists[name] = hist
            p = wd.to_level(Mo.predict_net(net, wd.X_test), wd.X_test)
        elif name in ("svr", "svm"):
            p = wd.to_level(Mo.fit_svr(wd.X_train, wd.y_train, tc, seed).predict(wd.X_test), wd.X_test)
        elif name == "rf":
            p = wd.to_level(Mo.fit_rf(wd.X_train, wd.y_train, tc, seed).predict(wd.X_test), wd.X_test)
        elif name == "xgb":
            p = wd.to_level(Mo.fit_xgb(wd.X_train, wd.y_train, tc, seed).predict(wd.X_test), wd.X_test)
        elif name in ("arima", "arima_ann"):
            if order is None:
                search = {"max_p": 2, "max_d": 0, "max_q": 2} if quick else cfg.get("arima_search", {})
                order = tuple(cfg.get("arima_order") or Mo.select_arima_order(train_raw, **search))
                print(f"  ARIMA order = {order}")
            if name == "arima":
                p, _ = Mo.arima_rolling_one_step(train_raw, test_raw, order)
            else:
                p = Mo.arima_ann(train_raw, test_raw, order, w, tc, seed)
        else:
            raise ValueError(name)
        preds[name] = np.asarray(p)
        r = {"model": LABELS.get(name, name), **M.all_metrics(y_true, p), "train_s": round(time.time() - t0, 1)}
        rows.append(r)
        print("  " + ", ".join(f"{k}={v:.4f}" for k, v in r.items() if isinstance(v, float)))
    res = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    res.insert(0, "rank", range(1, len(res) + 1))
    res.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    extra = {"target": wd.target, "n_train_points": len(wd.train_series), "n_test_points": len(wd.test_series),
             "arima_order": order}
    ref = cfg.get("reference_model", "cnn_bilstm_attention")
    if ref in preds:
        extra["diebold_mariano_vs_" + ref] = {
            LABELS.get(k, k): dict(zip(("DM", "p"), S.diebold_mariano(y_true, preds[ref], v)))
            for k, v in preds.items() if k != ref}
        extra["residual_diagnostics_" + ref] = S.residual_diagnostics(y_true, preds[ref])
        extra["stratified_rmse"] = {LABELS.get(k, k): M.stratified_rmse(y_true, v) for k, v in preds.items()}
        rr = M.rmse(y_true, preds[ref])
        extra["rmse_reduction_pct_vs_baselines"] = {
            LABELS.get(k, k): 100 * (M.rmse(y_true, v) - rr) / M.rmse(y_true, v)
            for k, v in preds.items() if k != ref}
        if cfg.get("ljung_box_curve"):
            extra["ljung_box_pvalues_lags_1_24"] = {
                LABELS.get(k, k): S.ljung_box_curve(y_true - preds[k]) for k in preds if k in (ref, "lstm")}
    extra["loss_history"] = hists
    for extra_exp in cfg.get("extra_experiments", []):
        if extra_exp == "lr_sensitivity":
            extra["lr_sensitivity_rmse"] = {}
            for lr in cfg.get("lr_grid", [1e-4, 1e-3, 1e-2]):
                net, _ = Mo.train_net(Mo.build_net(ref, w, tc), wd.X_train, wd.y_train, tc, seed, lr=lr)
                extra["lr_sensitivity_rmse"][str(lr)] = M.rmse(y_true, wd.to_level(Mo.predict_net(net, wd.X_test), wd.X_test))
                print(f"  lr={lr}: RMSE={extra['lr_sensitivity_rmse'][str(lr)]:.4f}")
        if extra_exp == "optimizer_comparison":
            extra["optimizer_comparison"] = {}
            for opt in ("adam", "rmsprop", "sgd_momentum"):
                lr = tc.get("lr", 1e-3) if opt != "sgd_momentum" else tc.get("sgd_lr", 1e-2)
                net, h = Mo.train_net(Mo.build_net(ref, w, tc), wd.X_train, wd.y_train, tc, seed, opt, lr)
                extra["optimizer_comparison"][opt] = {
                    "test_rmse": M.rmse(y_true, wd.to_level(Mo.predict_net(net, wd.X_test), wd.X_test)),
                    "epochs_run": len(h["train_loss"]), "final_train_loss": h["train_loss"][-1]}
                print(f"  {opt}: {extra['optimizer_comparison'][opt]}")
    with open(os.path.join(out_dir, "details.json"), "w") as f:
        json.dump(extra, f, indent=2, default=float)
    pd.DataFrame({"timestamp": wd.test_index, "actual": y_true,
                  **{LABELS.get(k, k): v for k, v in preds.items()}}).to_csv(
        os.path.join(out_dir, "predictions.csv"), index=False)
    print("\n" + res.to_string(index=False))
    if "diebold_mariano_vs_" + ref in extra:
        print("\nDiebold-Mariano (negative => %s more accurate):" % LABELS.get(ref, ref))
        for k, v in extra["diebold_mariano_vs_" + ref].items():
            print(f"  vs {k:22s} DM={v['DM']:+.3f} p={v['p']:.4f}")
        print("Residual diagnostics:", {k: round(v, 4) for k, v in extra["residual_diagnostics_" + ref].items()})
    print(f"\nResults written to {out_dir}/")
    return res


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
