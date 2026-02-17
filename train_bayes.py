# train_bayes.py
# Generates artifacts for app.py (Bayesian Regression API)
#
# Input: CSV with numeric features + continuous target
# Output (in ./artifacts):
#   posterior.json       { "mn": [...], "Vn": [[...]], "an": float, "bn": float }
#   feature_stats.json   { "mu": [...], "sd": [...], "feature_names": [...] }  # RAW features only
#   ppc_report.json      { "summary": {...}, "bayes_p_values": {...}, "residuals": {...} }
#   model_card.json      metadata including raw feature names and notes
#
# Usage:
#   python train_bayes.py --csv data.csv --target y
#
# Notes:
# - Keeps numeric columns only (safe default).
# - Standardizes features (z-score) using TRAIN split stats.
# - Adds intercept internally (design = [1, z-scored features]).
# - Uses conjugate Normal–Inverse-Gamma prior (closed-form posterior).
# - PPC uses posterior predictive draws on test split to compute Bayes p-values.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.model_selection import train_test_split


ART = Path("artifacts")
ART.mkdir(exist_ok=True)


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def fit_bayes_linreg_conjugate(X_design: np.ndarray,
                               y: np.ndarray,
                               m0: np.ndarray,
                               V0: np.ndarray,
                               a0: float,
                               b0: float):
    """
    Conjugate Bayesian linear regression with Normal-Inverse-Gamma prior:
      beta | sigma^2 ~ N(m0, sigma^2 V0)
      sigma^2 ~ Inv-Gamma(a0, b0)

    Posterior:
      Vn = (V0^-1 + X'X)^-1
      mn = Vn (V0^-1 m0 + X'y)
      an = a0 + n/2
      bn = b0 + 0.5 (y'y + m0'V0^-1 m0 - mn'Vn^-1 mn)

    Returns mn, Vn, an, bn.
    """
    X = np.asarray(X_design, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)

    m0 = np.asarray(m0, dtype=float).reshape(-1, 1)
    V0 = np.asarray(V0, dtype=float)

    XtX = X.T @ X
    Xty = X.T @ y

    V0_inv = np.linalg.inv(V0)
    Vn_inv = V0_inv + XtX
    Vn = np.linalg.inv(Vn_inv)

    mn = Vn @ (V0_inv @ m0 + Xty)

    n = X.shape[0]
    an = float(a0 + n / 2.0)

    yty = float((y.T @ y)[0, 0])
    m0_term = float((m0.T @ V0_inv @ m0)[0, 0])
    mn_term = float((mn.T @ Vn_inv @ mn)[0, 0])
    bn = float(b0 + 0.5 * (yty + m0_term - mn_term))

    return mn, Vn, an, bn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV dataset.")
    ap.add_argument("--target", required=True, help="Target column name (continuous).")
    ap.add_argument("--test_size", type=float, default=0.2, help="Holdout fraction for PPC/test.")
    ap.add_argument("--random_state", type=int, default=42)

    # Prior hyperparameters (reasonable defaults)
    ap.add_argument("--prior_strength", type=float, default=1.0,
                    help="Larger => tighter prior on beta (smaller V0).")
    ap.add_argument("--a0", type=float, default=2.0, help="Inv-Gamma shape (>=1 recommended).")
    ap.add_argument("--b0", type=float, default=2.0, help="Inv-Gamma scale (>0).")

    # PPC settings
    ap.add_argument("--ppc_draws", type=int, default=800, help="Posterior predictive draws for PPC.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in CSV columns.")

    # y
    y = df[args.target].astype(float).to_numpy()

    # X (numeric only)
    Xdf = df.drop(columns=[args.target]).select_dtypes(include=[np.number]).copy()
    feature_names = Xdf.columns.tolist()
    if len(feature_names) == 0:
        raise ValueError("No numeric features found after dropping target. Provide numeric columns.")

    X = Xdf.to_numpy(dtype=float)

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # Standardize using TRAIN stats (important!)
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0) + 1e-12

    X_trs = (X_tr - mu) / sd
    X_tes = (X_te - mu) / sd

    # Add intercept to make design matrix
    X_tr_design = np.concatenate([np.ones((X_trs.shape[0], 1)), X_trs], axis=1)
    X_te_design = np.concatenate([np.ones((X_tes.shape[0], 1)), X_tes], axis=1)

    d = X_tr_design.shape[1]

    # Prior
    # beta|sigma2 ~ N(m0, sigma2 V0)
    m0 = np.zeros(d)
    V0 = (1.0 / max(args.prior_strength, 1e-12)) * np.eye(d)

    # Fit posterior (closed form)
    mn, Vn, an, bn = fit_bayes_linreg_conjugate(
        X_tr_design, y_tr, m0=m0, V0=V0, a0=float(args.a0), b0=float(args.b0)
    )

    # Posterior predictive PPC on test
    rng = np.random.default_rng(2026)
    draws = int(args.ppc_draws)
    df_pred = 2.0 * an

    # Predictive mean for each test row
    locs = (X_te_design @ mn).reshape(-1)

    # Predictive scale per row: sqrt((bn/an) * (1 + x'Vn x))
    scales = np.empty(X_te_design.shape[0], dtype=float)
    s2 = (bn / an)
    for i in range(X_te_design.shape[0]):
        xi = X_te_design[i, :].reshape(-1, 1)
        scales[i] = np.sqrt(max(s2 * (1.0 + float((xi.T @ Vn @ xi)[0, 0])), 1e-12))

    # Draw posterior predictive replicates y_rep: (draws, n_test)
    y_rep = student_t.rvs(
        df=df_pred,
        loc=locs,
        scale=scales,
        size=(draws, X_te_design.shape[0]),
        random_state=rng
    )

    # Simple Bayesian p-values for mean & variance (model criticism)
    obs_mean = float(np.mean(y_te))
    rep_means = np.mean(y_rep, axis=1)
    p_mean = float(np.mean(rep_means >= obs_mean))

    if len(y_te) > 1:
        obs_var = float(np.var(y_te, ddof=1))
        rep_vars = np.var(y_rep, axis=1, ddof=1)
        p_var = float(np.mean(rep_vars >= obs_var))
    else:
        obs_var = 0.0
        p_var = 0.5  # neutral when variance not meaningful

    # Residual metrics using predictive mean
    resid = y_te - locs
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))

    # Save artifacts
    save_json(
        {"mn": mn.reshape(-1).tolist(), "Vn": Vn.tolist(), "an": an, "bn": bn},
        ART / "posterior.json",
    )

    save_json(
        {"mu": mu.tolist(), "sd": sd.tolist(), "feature_names": feature_names},
        ART / "feature_stats.json",
    )

    save_json(
        {
            "summary": {
                "n_train": int(X_tr_design.shape[0]),
                "n_test": int(X_te_design.shape[0]),
                "ppc_draws": draws,
                "df_predictive": float(df_pred),
                "prior_strength": float(args.prior_strength),
                "a0": float(args.a0),
                "b0": float(args.b0),
            },
            "bayes_p_values": {"mean": p_mean, "variance": p_var},
            "residuals": {"mae": mae, "rmse": rmse},
        },
        ART / "ppc_report.json",
    )

    save_json(
        {
            "model_name": "Conjugate Bayesian Linear Regression (Normal–Inverse-Gamma)",
            "model_version": "bayes-linreg-v1.0.0",
            "target": args.target,
            "feature_names": feature_names,  # RAW features only
            "server_transforms": "z-score standardization using training mu/sd + intercept added server-side",
            "posterior": "Closed-form posterior (no MCMC).",
            "posterior_predictive": "Student-t with df=2*a_n; scale uses (b_n/a_n)*(1 + x'V_n x).",
            "ppc": "Bayesian p-values for mean/variance computed via posterior predictive draws on holdout set.",
            "notes": [
                "Send RAW features in the exact order returned by GET /features.",
                "If you retrain (new data), regenerate artifacts and redeploy."
            ],
        },
        ART / "model_card.json",
    )

    print("✅ Saved artifacts to ./artifacts")
    print(f"- features: {len(feature_names)}")
    print(f"- posterior df (predictive): {df_pred:.2f}")
    print(f"- PPC Bayes p-values: mean={p_mean:.3f}, var={p_var:.3f}")
    print(f"- Residuals: MAE={mae:.4f}, RMSE={rmse:.4f}")


if __name__ == "__main__":
    main()
