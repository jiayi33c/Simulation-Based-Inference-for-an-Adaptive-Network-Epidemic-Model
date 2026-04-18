from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import math
import numpy as np
import pandas as pd

from simulator import simulate

PROJECT_ROOT = Path.cwd()
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PRIOR_BOUNDS = {
    "beta": (0.05, 0.50),
    "gamma": (0.02, 0.20),
    "rho": (0.0, 0.8),
}

N_SAMPLES = 100_000
N_REPS = 40
ACCEPT_FRAC = 0.01
SEED = 2026
N_WORKERS = 28
BATCH_SIZE = 5_000

TRUE_THETA = {
    "beta": 0.168664,
    "gamma": 0.089009,
    "rho": 0.317590,
}

SELECTED_COLUMNS = [
    "infected_peak",
    "infected_time_of_peak",
    "infected_auc",
    "infected_extinction_time",
    "rewiring_peak",
    "rewiring_time_of_peak",
    "rewiring_auc",
    "rewiring_zero_fraction",
]


def simulate_one(beta: float, gamma: float, rho: float, seed: int):
    infected, rewiring, degrees = simulate(
        beta=beta,
        gamma=gamma,
        rho=rho,
        rng=np.random.default_rng(seed),
    )
    return {
        "infected": infected,
        "rewiring": rewiring,
        "degrees": degrees,
    }


def simulate_replicates(beta: float, gamma: float, rho: float, n_reps: int = 40, seed: int = 42):
    local_rng = np.random.default_rng(seed)
    return [simulate_one(beta, gamma, rho, int(local_rng.integers(0, 2**31 - 1))) for _ in range(n_reps)]


def summarize_time_series(x):
    x = np.asarray(x, dtype=float)
    t_peak = int(np.argmax(x))
    positive_idx = np.flatnonzero(x > 0)
    extinction_time = float(positive_idx[-1]) if len(positive_idx) else 0.0
    return {
        "peak": float(np.max(x)),
        "time_of_peak": float(t_peak),
        "auc": float(np.trapezoid(x)),
        "extinction_time": extinction_time,
        "zero_fraction": float(np.mean(x == 0)),
    }


def summarize_degree_histogram(degree_hist):
    counts = np.asarray(degree_hist, dtype=float)
    degree_values = np.arange(len(counts), dtype=float)
    total_nodes = counts.sum()
    mean_degree = float(np.sum(degree_values * counts) / total_nodes)
    var_degree = float(np.sum(((degree_values - mean_degree) ** 2) * counts) / total_nodes)
    frac_deg_le_5 = float(np.sum(counts[:6]) / total_nodes)
    frac_deg_ge_15 = float(np.sum(counts[15:]) / total_nodes)
    return {
        "mean_degree": mean_degree,
        "var_degree": var_degree,
        "frac_deg_le_5": frac_deg_le_5,
        "frac_deg_ge_15": frac_deg_ge_15,
    }


def summarize_replicate(rep):
    infected_stats = {f"infected_{k}": v for k, v in summarize_time_series(rep["infected"]).items()}
    rewiring_stats = {f"rewiring_{k}": v for k, v in summarize_time_series(rep["rewiring"]).items()}
    degree_stats = summarize_degree_histogram(rep["degrees"])
    return infected_stats | rewiring_stats | degree_stats


def summarize_replicates(replicates):
    per_rep = pd.DataFrame([summarize_replicate(rep) for rep in replicates])
    return per_rep.mean(axis=0)


def sample_prior(n_samples: int, rng: np.random.Generator):
    return pd.DataFrame({
        name: rng.uniform(low, high, size=n_samples)
        for name, (low, high) in PRIOR_BOUNDS.items()
    })


def fit_summary_scale(summary_df: pd.DataFrame):
    return summary_df.std(axis=0).replace(0.0, 1.0)


def weighted_distance(sim_summary, obs_summary, scale, selected_columns):
    diff = (sim_summary[selected_columns] - obs_summary[selected_columns]) / scale[selected_columns]
    return float(np.sqrt(np.sum(np.square(diff))))


def run_pilot(n_pilot: int = 20, n_reps: int = 40, seed: int = 123):
    pilot_rng = np.random.default_rng(seed)
    theta = sample_prior(n_pilot, pilot_rng)
    summaries = []

    for row in theta.itertuples(index=False):
        reps = simulate_replicates(
            row.beta,
            row.gamma,
            row.rho,
            n_reps=n_reps,
            seed=int(pilot_rng.integers(0, 2**31 - 1)),
        )
        summaries.append(summarize_replicates(reps))

    summary_df = pd.DataFrame(summaries)
    return fit_summary_scale(summary_df)


def evaluate_one(args):
    beta, gamma, rho, seed, observed_summary, summary_scale, selected_columns, n_reps = args
    reps = simulate_replicates(beta, gamma, rho, n_reps=n_reps, seed=seed)
    sim_summary = summarize_replicates(reps)
    dist = weighted_distance(sim_summary, observed_summary, summary_scale, selected_columns)

    row = {
        "beta": beta,
        "gamma": gamma,
        "rho": rho,
        "distance": dist,
    }

    for col in selected_columns:
        row[col] = sim_summary[col]

    return row


def run_batch(batch_idx, batch_size, observed_summary, summary_scale, selected_columns):
    batch_seed = SEED + batch_idx
    rng = np.random.default_rng(batch_seed)
    theta = sample_prior(batch_size, rng)

    jobs = [
        (
            row.beta,
            row.gamma,
            row.rho,
            int(rng.integers(0, 2**31 - 1)),
            observed_summary,
            summary_scale,
            selected_columns,
            N_REPS,
        )
        for row in theta.itertuples(index=False)
    ]

    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for result in ex.map(evaluate_one, jobs, chunksize=50):
            rows.append(result)

    batch_df = pd.DataFrame(rows)
    batch_path = RESULTS_DIR / f"synthetic_recovery_batch_{batch_idx:04d}.csv"
    batch_df.to_csv(batch_path, index=False)
    print(f"Finished batch {batch_idx} with {len(rows)} samples", flush=True)


def load_all_batches():
    batch_files = sorted(RESULTS_DIR.glob("synthetic_recovery_batch_*.csv"))
    if not batch_files:
        raise RuntimeError("No synthetic recovery batch files found.")
    frames = [pd.read_csv(path) for path in batch_files]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    synthetic_replicates = simulate_replicates(
        TRUE_THETA["beta"],
        TRUE_THETA["gamma"],
        TRUE_THETA["rho"],
        n_reps=N_REPS,
        seed=SEED,
    )
    observed_summary = summarize_replicates(synthetic_replicates)
    summary_scale = run_pilot(n_pilot=20)

    n_batches = math.ceil(N_SAMPLES / BATCH_SIZE)

    for batch_idx in range(n_batches):
        batch_path = RESULTS_DIR / f"synthetic_recovery_batch_{batch_idx:04d}.csv"
        if batch_path.exists():
            print(f"Skipping existing batch {batch_idx}", flush=True)
            continue

        current_batch_size = min(BATCH_SIZE, N_SAMPLES - batch_idx * BATCH_SIZE)
        print(f"Starting batch {batch_idx} ({current_batch_size} samples)", flush=True)
        run_batch(
            batch_idx=batch_idx,
            batch_size=current_batch_size,
            observed_summary=observed_summary,
            summary_scale=summary_scale,
            selected_columns=SELECTED_COLUMNS,
        )

    all_results = load_all_batches().sort_values("distance", ascending=True).reset_index(drop=True)
    n_accept = max(1, int(math.ceil(ACCEPT_FRAC * len(all_results))))
    accepted = all_results.head(n_accept).copy()

    all_results.to_csv(RESULTS_DIR / "synthetic_recovery_results.csv", index=False)
    accepted.to_csv(RESULTS_DIR / "synthetic_recovery_accepted.csv", index=False)
    pd.DataFrame([TRUE_THETA]).to_csv(RESULTS_DIR / "synthetic_recovery_truth.csv", index=False)

    print("Synthetic recovery complete.", flush=True)
    print("True parameters:", TRUE_THETA, flush=True)
    print(f"Accepted samples: {len(accepted)}", flush=True)
    print(accepted[["beta", "gamma", "rho"]].describe(), flush=True)
