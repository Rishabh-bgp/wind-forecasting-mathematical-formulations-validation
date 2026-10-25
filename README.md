# Mathematical Formulations and Statistical Validation of Hybrid Deep Learning Models for Wind Speed Forecasting (PyTorch)

<a href="https://rishabh-bgp.github.io/wind-forecasting-mathematical-formulations-validation/"><img src="docs/img/banner.png" width="100%" alt="Banner: Mathematical Formulations and Statistical Validation of Hybrid Deep Learning Models for Wind Speed Forecasting, with a red seal stamp and a Bagua trigram"></a>

[![Live demo](https://img.shields.io/badge/Live%20demo-GitHub%20Pages-B3121B)](https://rishabh-bgp.github.io/wind-forecasting-mathematical-formulations-validation/) [![CI](https://img.shields.io/badge/CI-GitHub%20Actions-B3121B)](https://github.com/Rishabh-bgp/wind-forecasting-mathematical-formulations-validation/actions/workflows/ci.yml) [![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-D4A64A)](https://colab.research.google.com/github/Rishabh-bgp/wind-forecasting-mathematical-formulations-validation/blob/main/notebooks/walkthrough.ipynb) [![License: MIT](https://img.shields.io/badge/License-MIT-D4A64A.svg)](LICENSE) ![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-111214) ![Data: synthetic](https://img.shields.io/badge/data-synthetic-111214)

Equation-by-equation implementation of ANN, LSTM/BiLSTM, SVR, ARIMA-ANN, CNN-BiLSTM and CNN-BiLSTM-Attention for short-term wind speed forecasting, with Adam/RMSprop/SGD optimizer comparison, learning-rate sensitivity, loss curves, Ljung-Box, Shapiro-Wilk, Breusch-Pagan and Diebold-Mariano tests. Renewable energy time-series forecasting, M.Tech thesis work, IIIT Bhagalpur. The paper proposes a unified **mathematical formulation** of six wind speed forecasting models (ANN, LSTM, SVM/SVR, ARIMA–ANN, CNN–BiLSTM and CNN–BiLSTM–Attention). It covers layer equations, LSTM gates, soft attention, MSE loss, backpropagation through time, Adam updates, and optimizer and learning-rate studies. It also sets out a full **statistical validation** toolkit: Ljung–Box, Shapiro–Wilk, Breusch–Pagan and Diebold–Mariano.

<a href="https://rishabh-bgp.github.io/wind-forecasting-mathematical-formulations-validation/"><img src="docs/img/demo.png" width="100%" alt="Screenshot of the Daoist-style live demo for wind-forecasting-mathematical-formulations-validation: red and gold page with the 離 (Lí, fire) trigram, a forecast chart on synthetic data compared with persistence, and input sliders"></a>

**Live demo:** https://rishabh-bgp.github.io/wind-forecasting-mathematical-formulations-validation/ (runs entirely in your browser on synthetic data)

**About the demo design.** The page uses a Daoist, Chinese-ink style: crimson, charcoal and navy with gold accents, brush-style headings in the Ma Shan Zheng font, and charts drawn in gold, red and ink. Each repository has its own trigram. This one is **離 ☲ (Lí, Fire)**. 離 (Lí) is fire: light and clarity. It suits a paper that makes each equation visible and checks it with statistical tests. The red seal stamp (印) carries the initials RA, the faint taiji and the row of eight Bagua trigrams stand for balance and change, the cloud scrolls stand for moving air, and the vertical motto 明照四方 means "Brightness illuminates the four quarters". The design is purely decorative. Everything is computed by the model shown, on synthetic data.

## What this repository implements

Code that maps one-to-one onto the paper's equations:
- **ANN** (Sec. 3.1, x in R^24 -> affine + ReLU layers), **LSTM/BiLSTM** gates (3.2-3.3, `torch.nn.LSTM`), **1-D convolution** ((T-k+1) x F output, 3.4), **soft attention** e_t = v^T tanh(W_a h_t + b_a), alpha = softmax(e), c = sum alpha_t h_t (3.5, `models.SoftAttention`), **SVR** with RBF kernel (3.6), **ARIMA-ANN** y = L + N (3.7, `models.arima_ann`), **CNN-BiLSTM-Attention** (3.8).
- **MSE loss + Adam** (Sec. 4), training/validation **loss curves** for all deep models (`details.json -> loss_history`).
- **Learning-rate sensitivity** of the hybrid (1e-4, 5e-4, 1e-3, 5e-3, 1e-2) and **optimizer comparison** (Adam vs RMSprop vs SGD+Momentum), Sec. 4.4-4.5.
- **Statistical validation** (Sec. 5): MAE/RMSE/MAPE/R2, Shapiro-Wilk normality, **Ljung-Box p-values at lags 1-24** for the hybrid and LSTM, Breusch-Pagan heteroskedasticity, **Diebold-Mariano** (with Harvey-Leybourne-Newbold correction, `stats.diebold_mariano`).

## 架構 · Architecture

```
input  : last 24 hourly wind speeds (min-max scaled, scaler fit on train only)
Conv1D : 64 filters, kernel 3, ReLU            -> (22 x 64)
BiLSTM : 64 units per direction, full sequence -> (22 x 128)
Soft attention: e_t = v^T tanh(W_a h_t + b_a), alpha = softmax(e), c = sum alpha_t h_t
Dense  : 1 unit = predicted CHANGE from the last reading
output : forecast = last reading + predicted change
```
Training: Adam (lr 0.001), MSE loss, batch 32, up to 100 epochs, early stopping (patience 10), fixed seed 42.
Data split: chronological 80 % train / 20 % test; 24 -> 1 sliding window.
Target: every learned model (deep and shallow) predicts the **change from the last reading** (`target: delta` in the config; set `target: level` to predict the value directly). Persistence is the special case where the predicted change is zero.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py --quick      # smoke test: 3 epochs, short series, small grids
python train.py              # full run with the paper's settings
python -m pytest -q          # tests (incl. JS-vs-Python demo check; needs Node.js)
```

Outputs in `results/`: `metrics.csv` (MAE, RMSE, MAPE, R2 per model), `predictions.csv`,
`details.json` (Diebold-Mariano tests vs the hybrid, Ljung-Box / Shapiro-Wilk / Breusch-Pagan
residual diagnostics, wind-regime stratified RMSE, RMSE reduction %, loss curves).

Repository layout:

```
src/windforecast/  data.py (loader, synthetic generator, windowing, scaling)
                   models.py (CNN-BiLSTM-Attention and baselines)
                   metrics.py, stats.py (DM test, residual diagnostics)
                   experiment.py (end-to-end runner)
configs/default.yaml   train.py   tests/   notebooks/walkthrough.ipynb
scripts/make_synthetic_data.py   scripts/export_demo_model.py   demo/ (GitHub Pages source)
data/synthetic/synthetic_scada.csv (SYNTHETIC)
```

## Notebook

[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-D4A64A)](https://colab.research.google.com/github/Rishabh-bgp/wind-forecasting-mathematical-formulations-validation/blob/main/notebooks/walkthrough.ipynb)

[`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb) is a beginner-friendly, self-contained version of the code, with explanations in plain words. It runs in Colab or in Anaconda Jupyter on Windows, macOS or Linux:

1. Load the synthetic SCADA-style CSV (local file, else download from GitHub, else generate it).
2. Fill gaps and split 80/20 by time.
3. Scale to [0, 1], fitting the scaler on training data only.
4. Turn the series into 24-hour input windows with a 1-hour target.
5. Build the CNN–BiLSTM–Attention model.
6. Train it briefly (up to 20 epochs with early stopping).
7. Compare MAE, RMSE, MAPE and R² with a persistence baseline.
8. Plot actual vs predicted.

## 數據 · Data

**The original SCADA dataset used in the paper is not publicly available** (it is confidential and
is not, and will never be, included in this repository).

- `data/synthetic/synthetic_scada.csv` is a **SYNTHETIC** hourly SCADA-style wind speed series generated by
  `scripts/make_synthetic_data.py` (seed 42). It is committed only so the code runs out of the box.
  It is not measured data, and results on it say nothing about the published results.
- To use your own data locally, put a CSV anywhere under `data/` (for example `data/scada.csv`)
  with columns `timestamp,wind_speed` (other names: set `timestamp_col` / `wind_col`; 10-minute data is mean-resampled to hourly), and point `data.csv` in `configs/default.yaml` at it. Everything in `data/` except the
  committed synthetic CSV is git-ignored, so your file will not be committed.

## 結果 · Results (honest, synthetic data)

These are the real numbers from runs on the **synthetic** CSV. They are not the paper's results, and they are not tuned to match them. Every learned model predicts the **change from the last reading**, so persistence (change = 0) is the honest baseline to beat.

**1. Smoke test** (`python train.py --quick`: first 2,000 hours, 3 epochs, small hyper-parameter grids; test = last 400 hours):

| Rank | Model | MAE (m/s) | RMSE (m/s) | MAPE (%) | R² |
|---:|---|---:|---:|---:|---:|
| 1 | LSTM | 0.595 | 0.751 | 7.99 | 0.733 |
| 2 | ARIMA-ANN | 0.604 | 0.752 | 8.19 | 0.733 |
| 3 | ANN | 0.601 | 0.753 | 8.07 | 0.732 |
| 4 | CNN-BiLSTM | 0.601 | 0.754 | 8.09 | 0.731 |
| 5 | CNN-BiLSTM-Attention | 0.602 | 0.762 | 7.97 | 0.725 |
| 6 | **Persistence** | 0.597 | 0.764 | 7.87 | 0.723 |
| 7 | SVM (SVR-RBF) | 0.768 | 0.970 | 10.31 | 0.555 |

Does CNN–BiLSTM–Attention beat persistence? RMSE 0.762 vs 0.764 m/s (beats persistence), MAE 0.602 vs 0.597 m/s (does not beat persistence). Models with lower RMSE than persistence in this run: LSTM, ARIMA-ANN, ANN, CNN-BiLSTM, CNN-BiLSTM-Attention. The margins are small, and three epochs are a smoke test, not a tuned result.

**2. Small demo model** (the one on the live demo: 16 filters, 16 LSTM units per direction, up to 15 epochs, first 4,000 hours; test = last 800 hours):

| Model | MAE (m/s) | RMSE (m/s) | MAPE (%) | R² |
|---|---:|---:|---:|---:|
| CNN–BiLSTM–Attention (small) | 0.614 | 0.771 | 8.49 | 0.750 |
| **Persistence** | 0.617 | 0.795 | 8.35 | 0.734 |

Does it beat persistence? RMSE 0.771 vs 0.795 m/s (beats persistence), MAE 0.614 vs 0.617 m/s (beats persistence). On this smooth synthetic series, persistence is a strong baseline, so any gain is modest.

## Reference implementation note

> **Reference implementation.** This repository is a reference implementation of my own paper (cited below), rebuilt in Python and run on synthetic data because the original SCADA data is confidential. Results from this code will differ from the numbers published in the paper, and this repository does not claim to reproduce them.

### Assumptions and deviations from the paper

- SGD+Momentum uses lr = 0.01, momentum 0.9 (the paper does not give its settings); RMSprop uses lr = 0.001.
- The learning-rate grid is an assumption (the paper only names 0.0001 and 0.001 explicitly).
- The DM test uses squared-error loss, h = 1, and the HLN small-sample correction.
- Deep models are implemented in **PyTorch**; the paper used TensorFlow/Keras.
- The early-stopping validation set is the last 10 % of the training windows (not stated in the paper).
- Attention hidden size d_a = 64 and no dropout (not stated in the paper).
- MAPE ignores hours with |y| < 0.1 m/s to avoid division by ~0 (handling not stated).
- ARIMA is evaluated as rolling one-step-ahead forecasts (parameters fit on train only), matching the 1-step horizon of the other models.

## Citation

R. Aryan and Manimozhi I, "Mathematical formulations, optimization, and statistical validation of hybrid deep learning models for short-term wind speed forecasting: A comparative analysis of ANN, LSTM, SVM, ARIMA-ANN, CNN-BiLSTM, and CNN-BiLSTM-Attention architectures," *Int. J. Res. Adv. Comput. Sci. Eng. (IJRACSE)*, vol. 12, no. 1, pp. 46–53, May 2026. [Online]. Available: https://gnpublication.org/index.php/cse

## License

MIT License, © 2026 Rishabh Aryan. See [`LICENSE`](LICENSE).

## M.Tech Thesis Work in AI and Data Science, IIIT Bhagalpur

This repository is part of my M.Tech thesis work, *An Investigation into Advanced Techniques for Wind Speed Forecasting with Applications to Renewable Energy Grid Integration*.

**1. Quantum and hybrid quantum ML**

<!-- paper 08: add link -->
<!-- paper 10: add link -->
<!-- paper 12: add link -->

**2. Metaheuristic-optimized hybrids**

<!-- paper 04: add link -->

**3. Deep learning with attention and gating**

- **01** [CNN–BiLSTM–Attention on synthetic SCADA-style data](https://github.com/Rishabh-bgp/cnn-bilstm-attention-synthetic-scada)
- **02** [CNN–BiLSTM–Attention on real SCADA data](https://github.com/Rishabh-bgp/cnn-bilstm-attention-real-scada)
- **03** [Benchmarking CNN–BiLSTM–Attention on Indian SCADA data](https://github.com/Rishabh-bgp/cnn-bilstm-attention-benchmarking-indian-scada)
<!-- paper 05: add link -->
- **06** [Mathematical formulations and statistical validation](https://github.com/Rishabh-bgp/wind-forecasting-mathematical-formulations-validation)
<!-- paper 09+13: add link -->
<!-- paper 16: add link -->

**4. IoT, automation and multi-agent systems**

<!-- paper 14: add link -->
<!-- paper 15: add link -->

**5. Statistical time-series models**

- **07** [ARIMA and SARIMA wind speed forecasting for Chennai](https://github.com/Rishabh-bgp/arima-sarima-wind-speed-chennai)

## Other repositories by Rishabh Aryan

<!--OTHER_REPOS-->
