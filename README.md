Bayesian Regression Inference API

Production-ready Bayesian linear regression service exposing posterior predictive uncertainty, coefficient credible intervals, posterior predictive checks (PPC), and effect curves.

Deployed with FastAPI and Docker on Railway using artifact-based model loading.

Overview

This project implements a conjugate Bayesian linear regression model using a Normal–Inverse-Gamma posterior. The service provides:

Posterior predictive mean and credible intervals

Posterior coefficient summaries with uncertainty

Posterior predictive checks (PPC diagnostics)

Predictive effect curves for sensitivity analysis

Artifact-based inference deployment

The model is trained offline and deployed as an inference-only API.

Model Details

Model: Conjugate Bayesian Linear Regression

Prior: Normal–Inverse-Gamma

Posterior predictive distribution: Student-t

Interval default: 95% posterior predictive interval

Inference: Closed-form (no MCMC)

Deployment: FastAPI + Docker + Railway

Endpoints
Health

GET /health
Returns service status and model version.

Features

GET /features
Returns ordered feature names and expected input dimension.

Predict

POST /predict
Returns:

Posterior predictive mean

Credible interval

Degrees of freedom

Predictive scale

Posterior Coefficients

GET /posterior/coefficients
Returns:

Posterior mean

Standard deviation

Credible interval per coefficient

Posterior Predictive Check

POST /ppc
Returns:

Bayesian p-values (mean & variance)

Residual metrics (MAE, RMSE)

Predictive Curve

POST /posterior/predictive_curve
Generates effect curves for individual features.

Architecture

Training pipeline:

Synthetic dataset generation

Z-score standardization

Conjugate posterior computation

Artifact serialization

Deployment pipeline:

Load trained posterior artifacts

Serve inference-only FastAPI endpoints

Containerized via Docker

Publicly deployed via Railway

Why This Project

Most ML deployments expose only point predictions.

This API exposes:

Full predictive uncertainty

Parameter uncertainty

Model diagnostics

Sensitivity analysis

It demonstrates a complete statistical modeling lifecycle:
training → validation → artifact management → containerization → cloud deployment.

Live Deployment

Interactive docs:
https://bayesian-regression-inference-api-production.up.railway.app/docs

Health check:
https://bayesian-regression-inference-api-production.up.railway.app/health

Future Extensions

Hierarchical Bayesian regression

Robust regression (heavy-tailed noise)

Drift monitoring

Conformal prediction overlay

Probabilistic decision rules
