# Stock Return Prediction with Supervised CNN Variational Autoencoders

## Overview

This project investigates whether representation learning can improve stock return prediction and long-short portfolio construction compared to traditional machine learning approaches.

Starting from daily stock market data, we construct rolling feature windows and train a series of models ranging from linear baselines to supervised Variational Autoencoders (VAEs). The primary goal is to learn latent representations that capture predictive information about future returns and can be used to build profitable trading strategies.

The project was developed as part of the **Advanced Machine Learning** course.

---

## Research Question

Can a supervised latent representation learned by a CNN-based Variational Autoencoder outperform traditional feature-based models in stock return prediction and long-short portfolio construction?

---

## Data

The dataset consists of daily stock data for a universe of large-cap US equities.

Features include:

* 1-day returns
* 20-day momentum
* 60-day momentum
* 120-day momentum
* 20-day volatility
* 60-day volatility

For each stock, rolling windows of the previous 60 trading days are used as model inputs.

---

## Project Pipeline

Raw stock features

→ Rolling time-series windows

→ CNN-VAE encoder

→ Latent representation

→ Future return prediction

→ Long-short portfolio construction

→ Backtesting and evaluation

---

## Models Investigated

### Baseline Models

* Ridge Regression
* Feature-based ranking strategies
* Volatility-based ranking strategies
* Contrarian momentum strategies

### Representation Learning Models

* Fully Connected VAE
* CNN-VAE
* Supervised CNN-VAE
* CNN-VAE with latent-space analysis

### Advanced Architectures

* Stock-level latent attention
* Temporal stock attention models
* Latent-space regime discovery

---

## Key Result

The strongest model was the **Supervised CNN-VAE**, which significantly outperformed the raw feature baseline.

| Model                      | Sharpe Ratio |
| -------------------------- | ------------ |
| Raw Feature Ridge          | 0.95         |
| Supervised CNN-VAE         | 1.57         |
| VAE Latent Stock Attention | 1.03         |

The supervised CNN-VAE achieved:

* Higher total return
* Higher risk-adjusted performance
* Lower volatility
* Lower maximum drawdown

compared to the baseline model.

---

## Walk-Forward Validation

To reduce the risk of overfitting, the models were evaluated using rolling walk-forward validation.

Training and testing periods:

| Train Period | Test Period |
| ------------ | ----------- |
| 2015-2019    | 2020        |
| 2016-2020    | 2021        |
| 2017-2021    | 2022        |
| 2018-2022    | 2023        |
| 2019-2023    | 2024        |

The supervised CNN-VAE outperformed the baseline in 3 out of 5 test periods and produced superior aggregate performance across all periods.

---

## Latent Space Analysis

A major objective of the project was to understand what the learned latent space represents.

Using the latent vectors produced by the supervised CNN-VAE:

1. Daily market states were constructed by averaging stock-level latent representations.
2. PCA was used for visualization.
3. KMeans clustering was applied to identify latent market regimes.

The discovered clusters exhibited:

* Different future return characteristics
* Different volatility characteristics
* Long-term temporal persistence

These findings suggest that the supervised CNN-VAE learns a **regime-sensitive latent representation** of the market.

---

## Main Findings

* Representation learning improves stock ranking quality.
* A supervised latent space is more predictive than an unsupervised latent space.
* CNN-based latent representations outperform raw feature models.
* The learned latent space appears to encode market regimes.
* The benefit of the VAE is regime-dependent and strongest during more complex market environments.

---

## Repository Structure

```text
notebooks/
│
├── 01_data_load_and_eda.ipynb
├── 02_feature_engineering.ipynb
├── ...
├── 14_train_supervised_cnn_vae.ipynb
├── 15_supervised_vae_backtest.ipynb
├── 16_walk_forward_validation.ipynb
├── 22_best_seed_selection.ipynb
├── 24_vae_latent_stock_attention.ipynb
└── 25_latent_market_regimes.ipynb
```

---

## Future Work

Potential extensions include:

* Transformer-based temporal encoders
* Graph neural networks for stock relationships
* Regime-aware portfolio allocation
* Multi-horizon return prediction
* Attention-based latent representations

---

## Authors

Áron Holló
Árpád Takács
Regina Fiam

Advanced Machine Learning Project

Eötvös Loránd University (ELTE)
