# Bayesian Regression Inference API

A production-ready Bayesian linear regression service exposing posterior predictive uncertainty, coefficient credible intervals, posterior predictive checks (PPC), and effect curves.

Built with FastAPI, deployed via Docker on Railway, and structured around artifact-based inference.

---

## Overview

This project implements a conjugate Bayesian linear regression model using a Normal–Inverse-Gamma posterior. The service provides:

- Posterior predictive mean and 95% credible intervals
- Posterior coefficient summaries with uncertainty
- Posterior predictive checks (Bayesian p-values)
- Feature sensitivity analysis via predictive curves
- Artifact-based inference-only deployment

The model is trained offline and deployed as a lightweight inference API.

---

## Model Details

- Model: Conjugate Bayesian Linear Regression
- Prior: Normal–Inverse-Gamma
- Posterior predictive distribution: Student-t
- Default credible interval: 95%
- Inference method: Closed-form (no MCMC)
- Deployment: FastAPI + Docker + Railway

---

## Live Deployment

Interactive API documentation:
https://bayesian-regression-inference-api-production.up.railway.app/docs

Health check:
https://bayesian-regression-inference-api-production.up.railway.app/health

---

## API Endpoints

### Health
`GET /health`  
Returns service status and model version.

### Features
`GET /features`  
Returns ordered feature names and expected input dimension.

### Predict
`POST /predict`  
Returns:
- Posterior predictive mean
- Credible interval
- Degrees of freedom
- Predictive scale

Example:

```json
{
  "x": [4.0, 7.5, 3.0, 45.0, 180.0, 0.9, 3.4, 0.2, 20.0, 0.3, -0.2, 0.1],
  "level": 0.95,
  "n_samples": 0
}
Posterior Coefficients

GET /posterior/coefficients

Returns posterior mean, standard deviation, and credible interval for each coefficient.

Posterior Predictive Check (PPC)

POST /ppc

Returns Bayesian p-values (mean and variance) and residual metrics (MAE, RMSE).

Predictive Curve

POST /posterior/predictive_curve

Generates feature sensitivity curves with uncertainty bands.

Architecture
Training Pipeline

Synthetic dataset generation

Z-score feature standardization

Closed-form posterior computation

Artifact serialization

Deployment Pipeline

Load trained posterior artifacts

Serve inference-only FastAPI endpoints

Containerized via Docker

Publicly deployed on Railway

Why This Project

Most ML deployments expose only point predictions.

This service exposes:

Full predictive uncertainty

Parameter uncertainty

Posterior diagnostics

Sensitivity analysis

It demonstrates a complete statistical modeling lifecycle:
training → validation → artifact management → containerization → cloud deployment.

Future Extensions

Hierarchical Bayesian regression

Robust heavy-tailed likelihoods

Drift detection and monitoring

Conformal prediction overlay

Decision-aware uncertainty thresholds


