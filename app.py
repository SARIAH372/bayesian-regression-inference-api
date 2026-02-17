# app.py
# Bayesian Regression API (Conjugate Normal–Inverse-Gamma) with:
# - Posterior predictive intervals (/predict)
# - Posterior Predictive Checks (/ppc)
# - Posterior coefficient summaries (/posterior/coefficients)
# - Posterior coefficient simulation (/posterior/simulate_coefficients)
# - Posterior predictive effect curve (/posterior/predictive_curve)
#
# Expects these files in ./artifacts:
#   posterior.json        { "mn": [...], "Vn": [[...]], "an": float, "bn": float }
#   feature_stats.json    { "mu": [...], "sd": [...], "feature_names": [...] }   # RAW feature names
#   ppc_report.json       stored PPC report (any JSON dict with keys used below)
#   model_card.json       metadata (any JSON dict; model_version optional)

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Optional, Dict, Any

import json
import numpy as np
from scipy.stats import t as student_t


# -------------------------
# Utils
# -------------------------
def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# Bayesian linreg core
# -------------------------
def posterior_predictive_params(x_design: np.ndarray, mn: np.ndarray, Vn: np.ndarray, an: float, bn: float):
    """
    Predictive distribution for y* at design vector x:
      y* ~ Student-t(df=2an, loc = x'mn, scale = sqrt( (bn/an) * (1 + x'Vn x) ))
    """
    x = np.asarray(x_design, dtype=float).reshape(-1, 1)
    loc = float((x.T @ mn)[0, 0])
    df = 2.0 * float(an)
    scale2 = (float(bn) / float(an)) * (1.0 + float((x.T @ Vn @ x)[0, 0]))
    scale = float(np.sqrt(max(scale2, 1e-12)))
    return df, loc, scale


def predictive_interval(df: float, loc: float, scale: float, level: float):
    alpha = 1.0 - float(level)
    lo = student_t.ppf(alpha / 2.0, df=df, loc=loc, scale=scale)
    hi = student_t.ppf(1.0 - alpha / 2.0, df=df, loc=loc, scale=scale)
    return float(lo), float(hi)


def predictive_samples(df: float, loc: float, scale: float, n: int, rng: np.random.Generator):
    return student_t.rvs(df=df, loc=loc, scale=scale, size=int(n), random_state=rng).astype(float).tolist()


def beta_posterior_scale(an: float, bn: float) -> float:
    # beta posterior scale scalar multiplier sqrt(bn/an)
    return float(np.sqrt(float(bn) / float(an)))


def coef_intervals(mn: np.ndarray, Vn: np.ndarray, an: float, bn: float, level: float):
    """
    Each coefficient marginal:
      beta_j ~ Student-t(df=2an, loc=mn_j, scale=sqrt((bn/an)*Vn_jj))
    """
    mn_ = np.asarray(mn, dtype=float).reshape(-1)
    Vn_ = np.asarray(Vn, dtype=float)
    df = 2.0 * float(an)
    s2 = float(bn) / float(an)
    alpha = 1.0 - float(level)
    q = student_t.ppf(1.0 - alpha / 2.0, df=df)

    rows = []
    for j in range(mn_.shape[0]):
        sd_j = float(np.sqrt(max(s2 * Vn_[j, j], 1e-12)))
        lo = float(mn_[j] - q * sd_j)
        hi = float(mn_[j] + q * sd_j)
        rows.append((float(mn_[j]), sd_j, lo, hi))
    return df, rows


def sample_beta(mn: np.ndarray, Vn: np.ndarray, an: float, bn: float, n_draws: int, rng: np.random.Generator):
    """
    Draw samples from multivariate t posterior of beta:
      beta = mn + sqrt(s2) * L z * sqrt(df/u)
    """
    mn_ = np.asarray(mn, dtype=float).reshape(-1)
    Vn_ = np.asarray(Vn, dtype=float)
    df = 2.0 * float(an)
    s2 = float(bn) / float(an)

    d = mn_.shape[0]
    L = np.linalg.cholesky(Vn_ + 1e-12 * np.eye(d))

    Z = rng.normal(size=(int(n_draws), d))
    u = rng.chisquare(df, size=(int(n_draws), 1))
    t_scale = np.sqrt(df / np.clip(u, 1e-12, None))  # (n,1)

    draws = mn_.reshape(1, -1) + (np.sqrt(s2) * (Z @ L.T)) * t_scale
    return draws.astype(float), float(df)


# -------------------------
# Schemas
# -------------------------
class PredictRequest(BaseModel):
    x: List[float] = Field(..., description="RAW feature vector aligned with /features ordering.")
    level: float = Field(0.90, ge=0.5, le=0.99, description="Posterior predictive interval level.")
    n_samples: int = Field(0, ge=0, le=5000, description="Optional posterior predictive samples to return.")


class PredictResponse(BaseModel):
    mean: float
    interval: Dict[str, float]  # {level, lower, upper}
    df: float
    scale: float
    samples: Optional[List[float]] = None
    model_version: str


class PPCRequest(BaseModel):
    X: Optional[List[List[float]]] = Field(None, description="Optional RAW feature matrix (rows).")
    y: Optional[List[float]] = Field(None, description="Optional targets aligned with X.")
    n_draws: int = Field(500, ge=50, le=5000, description="Posterior predictive draws for PPC.")


class PPCResponse(BaseModel):
    summary: Dict[str, Any]
    bayes_p_values: Dict[str, float]
    residuals: Dict[str, float]
    model_version: str


class CoefSummary(BaseModel):
    name: str
    mean: float
    sd: float
    ci: Dict[str, float]  # {level, lower, upper}


class CoefPosteriorResponse(BaseModel):
    df: float
    scale_beta: float
    level: float
    coefficients: List[CoefSummary]
    model_version: str


class CoefSimRequest(BaseModel):
    n_draws: int = Field(500, ge=50, le=20000)
    seed: int = Field(2026, ge=0, le=10_000_000)


class CoefSimResponse(BaseModel):
    coefficient_names: List[str]
    draws: List[List[float]]  # (n_draws, d)
    model_version: str


class PredictiveCurveRequest(BaseModel):
    feature: str = Field(..., description="RAW feature name to sweep (must be in /features).")
    x_min: float = Field(..., description="Minimum RAW value for sweep.")
    x_max: float = Field(..., description="Maximum RAW value for sweep.")
    n_points: int = Field(50, ge=10, le=400)
    level: float = Field(0.95, ge=0.5, le=0.99)
    baseline: Optional[List[float]] = Field(
        None,
        description="Optional baseline RAW vector (len = expected_dim). If omitted, uses training means.",
    )
    n_samples_per_point: int = Field(0, ge=0, le=1000)
    seed: int = Field(2026, ge=0, le=10_000_000)


class PredictiveCurveResponse(BaseModel):
    feature: str
    grid: List[float]
    mean: List[float]
    lower: List[float]
    upper: List[float]
    level: float
    df: float
    model_version: str
    samples: Optional[List[List[float]]] = None


# -------------------------
# Load artifacts
# -------------------------
ART = Path("artifacts")
POST = read_json(ART / "posterior.json")
STATS = read_json(ART / "feature_stats.json")
PPC_STORED = read_json(ART / "ppc_report.json")
CARD = read_json(ART / "model_card.json")

mn = np.array(POST["mn"], dtype=float).reshape(-1, 1)       # includes intercept + standardized betas
Vn = np.array(POST["Vn"], dtype=float)
an = float(POST["an"])
bn = float(POST["bn"])

mu = np.array(STATS["mu"], dtype=float)                     # RAW feature means
sd = np.array(STATS["sd"], dtype=float)                     # RAW feature stds
RAW_FEATURE_NAMES = list(STATS["feature_names"])
RAW_DIM = len(RAW_FEATURE_NAMES)

MODEL_VERSION = CARD.get("model_version", "bayes-linreg-v1.0.0")


def to_design_vector(x_raw: np.ndarray) -> np.ndarray:
    """
    RAW -> standardized -> add intercept.
    design = [1, (x-mu)/sd]
    """
    x_raw = np.asarray(x_raw, dtype=float).reshape(-1)
    if x_raw.shape[0] != RAW_DIM:
        raise ValueError(f"Expected {RAW_DIM} raw features, got {x_raw.shape[0]}.")
    x_std = (x_raw - mu) / np.clip(sd, 1e-12, None)
    return np.concatenate([[1.0], x_std], axis=0)


# -------------------------
# App
# -------------------------
app = FastAPI(title="Bayesian Regression API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/model_card")
def model_card():
    return CARD


@app.get("/features")
def features():
    return {"feature_names": RAW_FEATURE_NAMES, "expected_dim": RAW_DIM}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    x_raw = np.asarray(req.x, dtype=float)
    if x_raw.ndim != 1:
        raise HTTPException(status_code=400, detail="x must be a 1D list of floats.")

    try:
        x_design = to_design_vector(x_raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    df, loc, scale = posterior_predictive_params(x_design, mn, Vn, an, bn)
    lo, hi = predictive_interval(df, loc, scale, req.level)

    samples = None
    if req.n_samples and req.n_samples > 0:
        rng = np.random.default_rng(12345)
        samples = predictive_samples(df, loc, scale, int(req.n_samples), rng)

    return PredictResponse(
        mean=float(loc),
        interval={"level": float(req.level), "lower": float(lo), "upper": float(hi)},
        df=float(df),
        scale=float(scale),
        samples=samples,
        model_version=MODEL_VERSION,
    )


@app.post("/ppc", response_model=PPCResponse)
def ppc(req: PPCRequest):
    # If no data provided, return stored PPC report
    if req.X is None or req.y is None:
        # be forgiving if the stored JSON has extra keys
        return PPCResponse(
            summary=PPC_STORED.get("summary", {}),
            bayes_p_values=PPC_STORED.get("bayes_p_values", {}),
            residuals=PPC_STORED.get("residuals", {}),
            model_version=MODEL_VERSION,
        )

    X = np.asarray(req.X, dtype=float)
    y = np.asarray(req.y, dtype=float)

    if X.ndim != 2:
        raise HTTPException(status_code=400, detail="X must be a 2D list.")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise HTTPException(status_code=400, detail="y must be 1D and match rows of X.")
    if X.shape[1] != RAW_DIM:
        raise HTTPException(status_code=400, detail=f"Expected {RAW_DIM} raw features, got {X.shape[1]}.")

    # RAW -> standardized -> design
    X_std = (X - mu.reshape(1, -1)) / np.clip(sd.reshape(1, -1), 1e-12, None)
    X_design = np.concatenate([np.ones((X_std.shape[0], 1)), X_std], axis=1)

    rng = np.random.default_rng(2026)
    draws = int(req.n_draws)
    df_pred = 2.0 * an

    locs = (X_design @ mn).reshape(-1)

    scales = []
    for i in range(X_design.shape[0]):
        xi = X_design[i, :].reshape(-1, 1)
        scale2 = (bn / an) * (1.0 + float((xi.T @ Vn @ xi)[0, 0]))
        scales.append(np.sqrt(max(scale2, 1e-12)))
    scales = np.asarray(scales, dtype=float)

    y_rep = student_t.rvs(df=df_pred, loc=locs, scale=scales, size=(draws, X_design.shape[0]), random_state=rng)

    obs_mean = float(np.mean(y))
    rep_means = np.mean(y_rep, axis=1)
    p_mean = float(np.mean(rep_means >= obs_mean))

    obs_var = float(np.var(y, ddof=1)) if X_design.shape[0] > 1 else 0.0
    rep_vars = np.var(y_rep, axis=1, ddof=1) if X_design.shape[0] > 1 else np.zeros(draws)
    p_var = float(np.mean(rep_vars >= obs_var))

    resid = y - locs
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))

    return PPCResponse(
        summary={"n": int(X_design.shape[0]), "draws": draws, "df": float(df_pred)},
        bayes_p_values={"mean": p_mean, "variance": p_var},
        residuals={"mae": mae, "rmse": rmse},
        model_version=MODEL_VERSION,
    )


@app.get("/posterior/coefficients", response_model=CoefPosteriorResponse)
def posterior_coefficients(level: float = 0.95):
    if level < 0.5 or level > 0.99:
        raise HTTPException(status_code=400, detail="level must be in [0.5, 0.99].")

    coef_names = ["intercept"] + list(RAW_FEATURE_NAMES)
    df_post, rows = coef_intervals(mn, Vn, an, bn, level=level)
    scale_beta = beta_posterior_scale(an, bn)

    coefs: List[CoefSummary] = []
    for name, (m, sd_j, lo, hi) in zip(coef_names, rows):
        coefs.append(
            CoefSummary(
                name=name,
                mean=float(m),
                sd=float(sd_j),
                ci={"level": float(level), "lower": float(lo), "upper": float(hi)},
            )
        )

    return CoefPosteriorResponse(
        df=float(df_post),
        scale_beta=float(scale_beta),
        level=float(level),
        coefficients=coefs,
        model_version=MODEL_VERSION,
    )


@app.post("/posterior/simulate_coefficients", response_model=CoefSimResponse)
def posterior_simulate_coefficients(req: CoefSimRequest):
    coef_names = ["intercept"] + list(RAW_FEATURE_NAMES)
    rng = np.random.default_rng(int(req.seed))
    draws, _df_post = sample_beta(mn, Vn, an, bn, n_draws=int(req.n_draws), rng=rng)
    return CoefSimResponse(
        coefficient_names=coef_names,
        draws=draws.tolist(),
        model_version=MODEL_VERSION,
    )


@app.post("/posterior/predictive_curve", response_model=PredictiveCurveResponse)
def posterior_predictive_curve(req: PredictiveCurveRequest):
    if req.feature not in RAW_FEATURE_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown feature '{req.feature}'. Use GET /features for valid names.",
        )
    if req.x_max <= req.x_min:
        raise HTTPException(status_code=400, detail="x_max must be > x_min.")

    # Baseline RAW vector
    if req.baseline is None:
        x0 = mu.copy()  # training means
    else:
        x0 = np.asarray(req.baseline, dtype=float).reshape(-1)
        if x0.shape[0] != RAW_DIM:
            raise HTTPException(status_code=400, detail=f"baseline must have length {RAW_DIM}.")

    j = RAW_FEATURE_NAMES.index(req.feature)
    grid = np.linspace(float(req.x_min), float(req.x_max), int(req.n_points))

    means: List[float] = []
    lowers: List[float] = []
    uppers: List[float] = []
    all_samples: Optional[List[List[float]]] = [] if (req.n_samples_per_point and req.n_samples_per_point > 0) else None

    rng = np.random.default_rng(int(req.seed))

    last_df = None
    for v in grid:
        x_raw = x0.copy()
        x_raw[j] = float(v)

        x_design = to_design_vector(x_raw)
        df, loc, scale = posterior_predictive_params(x_design, mn, Vn, an, bn)
        lo, hi = predictive_interval(df, loc, scale, req.level)

        means.append(float(loc))
        lowers.append(float(lo))
        uppers.append(float(hi))
        last_df = df

        if all_samples is not None:
            s = predictive_samples(df, loc, scale, int(req.n_samples_per_point), rng)
            all_samples.append(s)

    return PredictiveCurveResponse(
        feature=req.feature,
        grid=grid.astype(float).tolist(),
        mean=means,
        lower=lowers,
        upper=uppers,
        level=float(req.level),
        df=float(last_df if last_df is not None else (2.0 * an)),
        model_version=MODEL_VERSION,
        samples=all_samples,
    )
