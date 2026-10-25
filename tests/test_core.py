import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from windforecast.data import generate_synthetic_scada, clean, WindowedData, make_windows
from windforecast.metrics import all_metrics


def test_windows_and_no_leak():
    X, y = make_windows(np.arange(30, dtype=float), 24, 1)
    assert X.shape == (6, 24) and y[0] == 24
    s = clean(generate_synthetic_scada(1000))
    wd = WindowedData(s, 24, 1, 0.8)
    assert len(wd.y_test) == len(wd.test_series)
    # delta target: last reading + change reproduces the level
    assert np.allclose(wd.to_level(wd.y_test, wd.X_test), wd.inverse(wd.y_test_level), atol=1e-5)
    # scaler is fitted on training data only
    assert np.isclose(wd.scaler.data_max_[0], wd.train_series.max())


def test_metrics_perfect():
    m = all_metrics(np.array([1.0, 2, 3]), np.array([1.0, 2, 3]))
    assert m["RMSE"] == 0 and m["R2"] == 1
