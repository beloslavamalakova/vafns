# Towards Variatonal Arbitrage-Free Nelson-Siegel Model

Hyperparameter Tuning for Ornstein-Uhlenbeck Transition Model with DVBF

## Overview
This project implements a Deep Variational Bayes Filter (DVBF) with an Ornstein-Uhlenbeck (OU) transition model to predict and evaluate time-series data. It includes a hyperparameter tuning pipeline using grid search to optimize the model's performance on a given dataset.

## Features
- **OU Transition Model**: Implements an Ornstein-Uhlenbeck process for latent state transitions.
- **Hyperparameter Tuning**: Automatically explores combinations of parameters like `theta`, `mu`, `sigma`, `latent_dim`, and `learning_rate`.
- **Performance Metrics**: Evaluates predictions using metrics such as MSE, MAE, R², and explained variance.
- **Grid Search Optimization**: Finds the best set of hyperparameters that minimizes the loss.
