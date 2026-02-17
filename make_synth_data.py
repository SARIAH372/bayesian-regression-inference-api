import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main(n=2500, seed=2026, out_csv="data.csv"):
    rng = np.random.default_rng(seed)

    study_hours = np.clip(rng.normal(3.5, 1.6, n), 0, 10)
    sleep_hours = np.clip(rng.normal(7.0, 1.2, n), 3.5, 10)
    screen_time = np.clip(rng.normal(4.0, 1.8, n), 0, 12)
    exercise_mins = np.clip(rng.gamma(2.0, 20.0, n), 0, 180)
    caffeine_mg = np.clip(rng.lognormal(mean=4.8, sigma=0.45, size=n), 0, 800)
    attendance_rate = np.clip(rng.beta(8, 2, n), 0, 1)
    prior_gpa = np.clip(rng.normal(3.2, 0.35, n), 1.8, 4.0)
    ses_index = np.clip(rng.normal(0.0, 1.0, n), -2.5, 2.5)
    commute_mins = np.clip(rng.gamma(2.0, 12.0, n), 0, 120)
    social_support = np.clip(rng.normal(0.0, 1.0, n), -2.5, 2.5)
    stress = np.clip(rng.normal(0.0, 1.0, n), -2.5, 2.5)

    cog_load = (
        0.55 * stress
        + 0.25 * (screen_time - 4.0) / 2.0
        + 0.20 * (commute_mins - 25.0) / 20.0
        - 0.25 * (sleep_hours - 7.0) / 1.2
        - 0.15 * social_support
        + rng.normal(0, 0.35, n)
    )

    base = 55.0
    study_effect = 14.0 * np.tanh((study_hours - 2.0) / 2.0)
    sleep_effect = 10.0 * np.exp(-((sleep_hours - 7.5) ** 2) / (2 * 1.2**2))
    screen_effect = -6.0 * np.log1p(screen_time)
    stress_screen_interaction = -3.5 * sigmoid(stress) * np.log1p(screen_time)
    exercise_effect = 6.5 * (1.0 - np.exp(-exercise_mins / 45.0))
    caffeine_effect = 3.0 * np.tanh((250.0 - caffeine_mg) / 250.0)
    attendance_effect = 18.0 * (attendance_rate - 0.75)
    gpa_effect = 12.0 * (prior_gpa - 3.0)
    ses_effect = 2.5 * ses_index
    support_effect = 2.0 * social_support
    commute_effect = -2.5 * np.tanh(commute_mins / 45.0)
    load_effect = -9.0 * np.tanh(cog_load)

    y_signal = (
        base
        + study_effect
        + sleep_effect
        + screen_effect
        + stress_screen_interaction
        + exercise_effect
        + caffeine_effect
        + attendance_effect
        + gpa_effect
        + ses_effect
        + support_effect
        + commute_effect
        + load_effect
    )

    noise_scale = 3.0 + 3.5 * sigmoid(stress + 0.8 * cog_load) + 1.5 * sigmoid(6.0 - sleep_hours)
    y = y_signal + rng.normal(0, noise_scale, n)

    performance_index = np.clip(y, 0, 100)

    df = pd.DataFrame({
        "study_hours": study_hours,
        "sleep_hours": sleep_hours,
        "screen_time": screen_time,
        "exercise_mins": exercise_mins,
        "caffeine_mg": caffeine_mg,
        "attendance_rate": attendance_rate,
        "prior_gpa": prior_gpa,
        "ses_index": ses_index,
        "commute_mins": commute_mins,
        "social_support": social_support,
        "stress": stress,
        "cognitive_load": cog_load,
        "performance_index": performance_index
    })

    df.to_csv(out_csv, index=False)
    print(f"✅ Wrote {out_csv} with shape {df.shape}")
    print("Target column: performance_index")

if __name__ == "__main__":
    main()
