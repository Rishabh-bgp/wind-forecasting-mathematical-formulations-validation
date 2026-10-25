"""Model zoo. Deep models in PyTorch (the papers used TensorFlow/Keras)."""
from __future__ import annotations
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


# ---------------------------------------------------------------- deep nets
class SoftAttention(nn.Module):
    """e_t = v^T tanh(W_a h_t + b_a); alpha = softmax(e); c = sum alpha_t h_t."""

    def __init__(self, in_dim: int, attn_dim: int = 64):
        super().__init__()
        self.W = nn.Linear(in_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, H):
        e = self.v(torch.tanh(self.W(H))).squeeze(-1)
        alpha = torch.softmax(e, dim=1)
        return torch.sum(alpha.unsqueeze(-1) * H, dim=1), alpha


class CNNBiLSTMAttention(nn.Module):
    def __init__(self, filters=64, kernel_size=3, lstm_units=64, attn_dim=64,
                 dropout=0.0, use_attention=True):
        super().__init__()
        self.conv = nn.Conv1d(1, filters, kernel_size)  # 'valid' -> T-k+1 steps
        self.act = nn.ReLU()
        self.bilstm = nn.LSTM(filters, lstm_units, batch_first=True, bidirectional=True)
        self.use_attention = use_attention
        self.attn = SoftAttention(2 * lstm_units, attn_dim) if use_attention else None
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * lstm_units, 1)
        self.last_alpha = None

    def forward(self, x):  # x: (B, T)
        z = self.act(self.conv(x.unsqueeze(1))).transpose(1, 2)  # (B, T', F)
        H, _ = self.bilstm(z)
        if self.use_attention:
            c, self.last_alpha = self.attn(H)
        else:
            c = H[:, -1, :]
        return self.head(self.drop(c)).squeeze(-1)


class LSTMNet(nn.Module):
    def __init__(self, units=64):
        super().__init__()
        self.lstm = nn.LSTM(1, units, batch_first=True)
        self.head = nn.Linear(units, 1)

    def forward(self, x):
        H, _ = self.lstm(x.unsqueeze(-1))
        return self.head(H[:, -1, :]).squeeze(-1)


class ANNNet(nn.Module):
    def __init__(self, in_dim=24, hidden=(64, 32)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_net(name: str, window: int, cfg: dict) -> nn.Module:
    if name == "cnn_bilstm_attention":
        return CNNBiLSTMAttention(cfg.get("filters", 64), cfg.get("kernel_size", 3),
                                  cfg.get("lstm_units", 64), cfg.get("attn_dim", 64),
                                  cfg.get("dropout", 0.0), True)
    if name == "cnn_bilstm":
        return CNNBiLSTMAttention(cfg.get("filters", 64), cfg.get("kernel_size", 3),
                                  cfg.get("lstm_units", 64), dropout=cfg.get("dropout", 0.0),
                                  use_attention=False)
    if name == "lstm":
        return LSTMNet(cfg.get("lstm_units_baseline", 64))
    if name == "ann":
        return ANNNet(window, tuple(cfg.get("ann_hidden", [64, 32])))
    raise ValueError(name)


def train_net(model, X_tr, y_tr, cfg: dict, seed: int = 42, optimizer: str = "adam",
              lr: float | None = None, verbose: bool = False):
    """Adam(lr) + MSE, batch 32, max epochs, early stopping on a chronological
    validation tail of the training set (the papers do not state the validation
    fraction; 10% is an assumption). Returns model and loss history."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    lr = lr or cfg.get("lr", 1e-3)
    n_val = max(1, int(len(X_tr) * cfg.get("val_frac", 0.1)))
    Xt, yt = torch.tensor(X_tr[:-n_val]), torch.tensor(y_tr[:-n_val])
    Xv, yv = torch.tensor(X_tr[-n_val:]), torch.tensor(y_tr[-n_val:])
    opt = {"adam": lambda p: torch.optim.Adam(p, lr=lr),
           "rmsprop": lambda p: torch.optim.RMSprop(p, lr=lr),
           "sgd_momentum": lambda p: torch.optim.SGD(p, lr=lr, momentum=0.9)}[optimizer](model.parameters())
    lossf = nn.MSELoss()
    bs, best, best_state, wait = cfg.get("batch_size", 32), float("inf"), None, 0
    hist = {"train_loss": [], "val_loss": []}
    g = torch.Generator().manual_seed(seed)
    for ep in range(cfg.get("epochs", 100)):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xv), yv).item()
        hist["train_loss"].append(tot / len(Xt))
        hist["val_loss"].append(vl)
        if verbose:
            print(f"  epoch {ep + 1}: train {tot / len(Xt):.5f} val {vl:.5f}")
        if vl < best - 1e-7:
            best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.get("patience", 10):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


def predict_net(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X)).numpy()


# ---------------------------------------------------------------- shallow
def fit_svr(X, y, cfg: dict, seed=42):
    grid = cfg.get("svr_grid", {"C": [1, 10], "gamma": ["scale", 0.1], "epsilon": [0.01, 0.05]})
    gs = GridSearchCV(SVR(kernel="rbf"), grid, cv=TimeSeriesSplit(cfg.get("cv_folds", 5)),
                      scoring="neg_mean_squared_error", n_jobs=-1)
    max_n = cfg.get("svr_max_train", None)
    Xs, ys = (X[-max_n:], y[-max_n:]) if max_n else (X, y)
    gs.fit(Xs, ys)
    return gs.best_estimator_


def fit_rf(X, y, cfg: dict, seed=42):
    return RandomForestRegressor(n_estimators=cfg.get("rf_trees", 300), random_state=seed,
                                 n_jobs=-1).fit(X, y)


def fit_xgb(X, y, cfg: dict, seed=42):
    grid = cfg.get("xgb_grid", {"n_estimators": [200, 400], "max_depth": [3, 6],
                                "learning_rate": [0.05, 0.1]})
    try:
        from xgboost import XGBRegressor
        est = XGBRegressor(random_state=seed, n_jobs=-1, verbosity=0)
    except ImportError:  # fallback keeps the repo runnable without xgboost
        from sklearn.ensemble import HistGradientBoostingRegressor
        warnings.warn("xgboost not installed; using HistGradientBoostingRegressor")
        est = HistGradientBoostingRegressor(random_state=seed)
        grid = {"max_iter": grid["n_estimators"], "max_depth": grid["max_depth"],
                "learning_rate": grid["learning_rate"]}
    gs = GridSearchCV(est, grid, cv=TimeSeriesSplit(cfg.get("cv_folds", 5)),
                      scoring="neg_mean_squared_error", n_jobs=1)
    return gs.fit(X, y).best_estimator_


# ---------------------------------------------------------------- ARIMA family
def select_arima_order(train, max_p=3, max_d=1, max_q=3):
    """AIC grid search over (p,d,q)."""
    from statsmodels.tsa.arima.model import ARIMA
    best, order = np.inf, (1, 0, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    if p == q == 0:
                        continue
                    try:
                        aic = ARIMA(train, order=(p, d, q)).fit().aic
                    except Exception:
                        continue
                    if aic < best:
                        best, order = aic, (p, d, q)
    return order


def arima_rolling_one_step(train, test, order):
    """One-step-ahead ARIMA forecasts over the test period: parameters are fit
    on train only, then the state is updated with each observed test value
    (append with refit=False). This mirrors the 24->1 one-step protocol of
    the deep models; a multi-step static forecast would be much worse."""
    from statsmodels.tsa.arima.model import ARIMA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(np.asarray(train, float), order=order).fit()
        preds = []
        for v in np.asarray(test, float):
            preds.append(float(res.forecast(1)[0]))
            res = res.append([v], refit=False)
    return np.asarray(preds), res


def arima_ann(train, test, order, window, cfg, seed=42):
    """Zhang (2003)-style hybrid: ARIMA for the linear part L_t, ANN on lagged
    ARIMA residuals for N_t; forecast = L^_t + N^_t."""
    from statsmodels.tsa.arima.model import ARIMA
    from .data import make_windows
    train = np.asarray(train, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(train, order=order).fit()
    resid_tr = train - res.fittedvalues
    lin_test, _ = arima_rolling_one_step(train, test, order)
    resid_test = np.asarray(test, float) - lin_test
    sc = float(np.std(resid_tr)) or 1.0
    Xr, yr = make_windows(resid_tr / sc, window)
    net, _ = train_net(ANNNet(window, tuple(cfg.get("ann_hidden", [64, 32]))), Xr, yr, cfg, seed)
    allr = np.concatenate([resid_tr, resid_test]) / sc
    ctx = allr[len(resid_tr) - window:]
    Xt, _ = make_windows(np.append(ctx, 0.0), window)  # windows end before each test point
    nl = predict_net(net, Xt[:len(test)]) * sc
    return lin_test + nl
