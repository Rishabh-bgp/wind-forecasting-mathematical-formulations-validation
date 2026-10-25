"""Data loading, synthetic SCADA-style generation, windowing and scaling."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def generate_synthetic_scada(n_hours: int = 8760, start: str = "2018-01-01",
                             missing_rate: float = 0.002, seed: int = 42) -> pd.DataFrame:
    """Synthetic hourly SCADA-style wind speed series.

    Follows the three additive components described in the papers: (i) a
    low-frequency seasonal cycle peaking around March/December with a monsoon
    minimum in July-August, (ii) a 24-h diurnal sinusoid, (iii) a stochastic
    AR(1) term with Weibull-like right skew; plus ~0.2% missing values and a
    slow sensor drift. Exact generator parameters are not given in the paper,
    so the constants below are assumptions tuned to give mostly 6-9 m/s.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_hours, freq="h")
    doy = idx.dayofyear.values
    hour = idx.hour.values
    # two maxima (~day 75 and ~day 350) and a minimum around day 200
    seasonal = 1.2 * np.cos(2 * np.pi * (doy - 350) / 365.0 * 2) * 0.5 \
        + 0.9 * np.cos(2 * np.pi * (doy - 20) / 365.0)
    seasonal -= 0.6 * np.exp(-((doy - 205) / 30.0) ** 2)
    diurnal = 0.8 * np.sin(2 * np.pi * (hour - 9) / 24.0)
    noise = np.zeros(n_hours)
    eps = rng.gamma(shape=2.0, scale=0.55, size=n_hours) - 1.1
    for t in range(1, n_hours):
        noise[t] = 0.85 * noise[t - 1] + eps[t]
    drift = np.linspace(0, 0.15, n_hours)
    ws = np.clip(7.5 + seasonal + diurnal + noise + drift, 0.0, None)
    ws[rng.random(n_hours) < missing_rate] = np.nan
    return pd.DataFrame({"timestamp": idx, "wind_speed": ws})


def load_scada_csv(path: str, timestamp_col: str = "timestamp",
                   wind_col: str = "wind_speed", resample: str | None = "1h",
                   dayfirst: bool = False) -> pd.DataFrame:
    """Load a SCADA-style CSV (e.g. 10-minute records) and mean-resample.

    Column names are configurable, e.g. for the public Kaggle "Wind Turbine
    Scada Dataset" use timestamp_col="Date/Time", wind_col="Wind Speed (m/s)",
    dayfirst=True.
    """
    df = pd.read_csv(path)
    ts = pd.to_datetime(df[timestamp_col], dayfirst=dayfirst)
    s = pd.Series(pd.to_numeric(df[wind_col], errors="coerce").values, index=ts).sort_index()
    if resample:
        s = s.resample(resample).mean()
    return pd.DataFrame({"timestamp": s.index, "wind_speed": s.values})


def clean(df: pd.DataFrame) -> pd.Series:
    """Linear interpolation of gaps (as in the papers)."""
    s = df.set_index("timestamp")["wind_speed"].astype(float)
    return s.interpolate(method="linear", limit_direction="both")


def chrono_split(s: pd.Series, train_frac: float = 0.8):
    n = int(round(len(s) * train_frac))
    return s.iloc[:n], s.iloc[n:]


def make_windows(values: np.ndarray, window: int = 24, horizon: int = 1):
    X, y = [], []
    for i in range(len(values) - window - horizon + 1):
        X.append(values[i:i + window])
        y.append(values[i + window + horizon - 1])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


class WindowedData:
    """Scaler fitted on TRAINING data only; test windows may use the last
    `window` training points as context (no test targets leak into training)."""

    def __init__(self, series: pd.Series, window: int = 24, horizon: int = 1,
                 train_frac: float = 0.8, target: str = "delta"):
        """target="delta": models learn the CHANGE from the last reading
        (y_{t+h} - y_t, in scaled units) and forecasts are last reading + change.
        target="level": models learn the next value directly."""
        self.window, self.horizon, self.target = window, horizon, target
        train, test = chrono_split(series, train_frac)
        self.train_series, self.test_series = train, test
        self.scaler = MinMaxScaler()
        self.scaler.fit(train.values.reshape(-1, 1))
        tr = self.scaler.transform(train.values.reshape(-1, 1)).ravel()
        full = self.scaler.transform(series.values.reshape(-1, 1)).ravel()
        self.X_train, self.y_train = make_windows(tr, window, horizon)
        ctx = full[len(train) - window - horizon + 1:]
        self.X_test, self.y_test = make_windows(ctx, window, horizon)
        self.test_index = test.index[:len(self.y_test)]
        self.y_test_level = self.y_test.copy()
        if target == "delta":
            self.y_train = (self.y_train - self.X_train[:, -1]).astype(np.float32)
            self.y_test = (self.y_test - self.X_test[:, -1]).astype(np.float32)

    def to_level(self, pred, X):
        """Model output (scaled delta or level) -> forecast in m/s."""
        pred = np.asarray(pred).ravel()
        if self.target == "delta":
            pred = pred + X[:, -1]
        return self.inverse(pred)

    def inverse(self, a):
        return self.scaler.inverse_transform(np.asarray(a).reshape(-1, 1)).ravel()
