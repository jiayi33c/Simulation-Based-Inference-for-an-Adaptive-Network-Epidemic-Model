from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import math
import os
import numpy as np
import pandas as pd

from simulator import simulate


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PRIOR_BOUNDS = {
    "beta": (0.05, 0.50),
    "gamma": (0.02, 0.20),
    "rho": (0.0, 0.8),
}

PARAM_NAMES = ["beta", "gamma", "rho"]
SUMMARY_COLUMNS = [
    "infected_peak",
    "infected_time_of_peak",
    "infected_auc",
    "infected_extinction_time",
    "rewiring_peak",
    "rewiring_time_of_peak",
    "rewiring_auc",
    "rewiring_zero_fraction",
]

N_PARTICLES = 1000
N_REPS = 40
SEED = 2024
N_WORKERS = max(1, int(os.environ.get("SMC_ABC_WORKERS", (os.cpu_count() or 2) - 1)))
TARGET_QUANTILES = (0.75, 0.50, 0.25, 0.10)
MIN_SCALE = np.array([0.002, 0.002, 0.005], dtype=float)
MAX_ATTEMPTS_PER_PARTICLE = 500


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


def simulate_replicates(beta: float, gamma: float, rho: float, n_reps: int = N_REPS, seed: int = 42):
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


def build_observed_replicates(infected_df, rewiring_df, degree_df):
    replicates = []
    replicate_ids = sorted(set(infected_df["replicate_id"]))
    for rep_id in replicate_ids:
        infected = infected_df.loc[infected_df["replicate_id"] == rep_id].sort_values("time")["infected_fraction"].to_numpy()
        rewiring = rewiring_df.loc[rewiring_df["replicate_id"] == rep_id].sort_values("time")["rewire_count"].to_numpy()
        degrees = degree_df.loc[degree_df["replicate_id"] == rep_id].sort_values("degree")["count"].to_numpy()
        replicates.append({
            "infected": infected,
            "rewiring": rewiring,
            "degrees": degrees,
        })
    return replicates


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


def run_pilot(n_pilot: int = 20, n_reps: int = N_REPS, seed: int = 123):
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


def in_prior_bounds(theta):
    return all(
        PRIOR_BOUNDS[name][0] <= value <= PRIOR_BOUNDS[name][1]
        for name, value in zip(PARAM_NAMES, theta)
    )


def evaluate_particle(args):
    theta, observed_summary, summary_scale, selected_columns, n_reps, seed = args
    beta, gamma, rho = theta
    reps = simulate_replicates(beta, gamma, rho, n_reps=n_reps, seed=seed)
    sim_summary = summarize_replicates(reps)
    distance = weighted_distance(sim_summary, observed_summary, summary_scale, selected_columns)

    row = {
        "beta": beta,
        "gamma": gamma,
        "rho": rho,
        "distance": distance,
    }
    for col in selected_columns:
        row[col] = sim_summary[col]
    return row


def evaluate_thetas(thetas, observed_summary, summary_scale, selected_columns, rng, n_reps=N_REPS):
    jobs = [
        (
            np.asarray(theta, dtype=float),
            observed_summary,
            summary_scale,
            selected_columns,
            n_reps,
            int(rng.integers(0, 2**31 - 1)),
        )
        for theta in thetas
    ]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        rows = list(ex.map(evaluate_particle, jobs, chunksize=25))
    return pd.DataFrame(rows)


def weighted_covariance(particles, weights):
    arr = particles[PARAM_NAMES].to_numpy()
    mean = np.average(arr, axis=0, weights=weights)
    centered = arr - mean
    cov = (centered * weights[:, None]).T @ centered
    cov /= weights.sum()
    cov += np.diag(MIN_SCALE ** 2)
    return cov


def mvn_density(x, mean, cov):
    dim = len(x)
    diff = x - mean
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance matrix must be positive definite.")
    inv_cov = np.linalg.inv(cov)
    log_norm = -0.5 * (dim * math.log(2 * math.pi) + logdet)
    log_kernel = log_norm - 0.5 * (diff @ inv_cov @ diff)
    return math.exp(log_kernel)


def normalized_weights(raw):
    raw = np.asarray(raw, dtype=float)
    total = raw.sum()
    if total <= 0.0:
        raise ValueError("Weight vector has non-positive total.")
    return raw / total


def resample_indices(weights, n_particles, rng):
    return rng.choice(len(weights), size=n_particles, replace=True, p=weights)


def propose_population(prev_particles, prev_weights, epsilon, kernel_cov, observed_summary, summary_scale, selected_columns, rng):
    kernel_scale = 2.0 * kernel_cov
    prev_thetas = prev_particles[PARAM_NAMES].to_numpy(dtype=float)
    jobs = [
        (
            prev_thetas,
            np.asarray(prev_weights, dtype=float),
            epsilon,
            kernel_scale,
            observed_summary,
            summary_scale,
            selected_columns,
            N_REPS,
            int(rng.integers(0, 2**31 - 1)),
        )
        for _ in range(len(prev_particles))
    ]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        proposed = list(ex.map(propose_one_particle, jobs, chunksize=10))

    accepted_rows = [row for row, _attempts in proposed]
    attempts_used = [attempts for _row, attempts in proposed]

    particles = pd.DataFrame(accepted_rows).reset_index(drop=True)
    raw_weights = []
    for theta in particles[PARAM_NAMES].to_numpy(dtype=float):
        denom = 0.0
        for w_prev, theta_prev in zip(prev_weights, prev_thetas):
            denom += w_prev * mvn_density(theta, theta_prev, kernel_scale)
        raw_weights.append(1.0 / denom)

    weights = normalized_weights(raw_weights)
    acceptance_rate = len(particles) / sum(attempts_used)
    return particles, weights, acceptance_rate, np.mean(attempts_used)


def propose_one_particle(args):
    (
        prev_thetas,
        prev_weights,
        epsilon,
        kernel_scale,
        observed_summary,
        summary_scale,
        selected_columns,
        n_reps,
        seed,
    ) = args

    rng = np.random.default_rng(seed)
    for attempt in range(1, MAX_ATTEMPTS_PER_PARTICLE + 1):
        # Resample the ancestor on each attempt so one bad parent does not
        # cause the whole worker to fail under a strict epsilon.
        base_idx = rng.choice(len(prev_thetas), p=prev_weights)
        base_theta = prev_thetas[base_idx]
        proposal = rng.multivariate_normal(base_theta, kernel_scale)
        if not in_prior_bounds(proposal):
            continue

        row = evaluate_particle(
            (
                proposal,
                observed_summary,
                summary_scale,
                selected_columns,
                n_reps,
                int(rng.integers(0, 2**31 - 1)),
            )
        )
        if row["distance"] <= epsilon:
            return row, attempt

    raise RuntimeError(
        f"Failed to accept a particle within {MAX_ATTEMPTS_PER_PARTICLE} proposals. "
        "Tolerance may be too strict."
    )


def run_smc_abc(observed_summary, summary_scale, selected_columns, n_particles=N_PARTICLES, quantiles=TARGET_QUANTILES, seed=SEED):
    rng = np.random.default_rng(seed)
    history_rows = []

    initial_theta = sample_prior(n_particles, rng)[PARAM_NAMES].to_numpy()
    particles = evaluate_thetas(
        initial_theta,
        observed_summary=observed_summary,
        summary_scale=summary_scale,
        selected_columns=selected_columns,
        rng=rng,
    )
    weights = np.full(n_particles, 1.0 / n_particles)

    for stage_idx, q in enumerate(quantiles, start=1):
        epsilon = float(np.quantile(particles["distance"], q))
        kernel_cov = weighted_covariance(particles, weights)

        if stage_idx == 1:
            eligible = particles[particles["distance"] <= epsilon].reset_index(drop=True)
            if eligible.empty:
                raise RuntimeError("No particles survived the first SMC threshold.")
            chosen = rng.choice(len(eligible), size=n_particles, replace=True)
            particles = eligible.iloc[chosen].reset_index(drop=True)
            weights = np.full(n_particles, 1.0 / n_particles)
            acceptance_rate = len(eligible) / len(initial_theta)
            mean_attempts = 1.0
        else:
            particles, weights, acceptance_rate, mean_attempts = propose_population(
                prev_particles=particles,
                prev_weights=weights,
                epsilon=epsilon,
                kernel_cov=kernel_cov,
                observed_summary=observed_summary,
                summary_scale=summary_scale,
                selected_columns=selected_columns,
                rng=rng,
            )

        stage_particles = particles.copy()
        stage_particles["weight"] = weights
        stage_particles["stage"] = stage_idx
        stage_particles["epsilon"] = epsilon
        stage_path = RESULTS_DIR / f"smc_abc_stage_{stage_idx:02d}.csv"
        stage_particles.to_csv(stage_path, index=False)

        history_rows.append({
            "stage": stage_idx,
            "epsilon_quantile": q,
            "epsilon": epsilon,
            "acceptance_rate": acceptance_rate,
            "mean_attempts": mean_attempts,
            "beta_mean": np.average(particles["beta"], weights=weights),
            "gamma_mean": np.average(particles["gamma"], weights=weights),
            "rho_mean": np.average(particles["rho"], weights=weights),
            "beta_std": math.sqrt(np.average((particles["beta"] - np.average(particles["beta"], weights=weights)) ** 2, weights=weights)),
            "gamma_std": math.sqrt(np.average((particles["gamma"] - np.average(particles["gamma"], weights=weights)) ** 2, weights=weights)),
            "rho_std": math.sqrt(np.average((particles["rho"] - np.average(particles["rho"], weights=weights)) ** 2, weights=weights)),
        })
        print(
            f"Stage {stage_idx}: epsilon={epsilon:.4f}, "
            f"acceptance_rate={acceptance_rate:.3f}, "
            f"beta_mean={history_rows[-1]['beta_mean']:.4f}, "
            f"gamma_mean={history_rows[-1]['gamma_mean']:.4f}, "
            f"rho_mean={history_rows[-1]['rho_mean']:.4f}",
            flush=True,
        )

    history = pd.DataFrame(history_rows)
    history.to_csv(RESULTS_DIR / "smc_abc_tuning_summary.csv", index=False)

    final_particles = particles.copy()
    final_particles["weight"] = weights
    final_particles.to_csv(RESULTS_DIR / "smc_abc_posterior_infected_plus_rewiring.csv", index=False)
    return history, final_particles


if __name__ == "__main__":
    infected_obs = pd.read_csv(DATA_DIR / "infected_timeseries.csv")
    rewiring_obs = pd.read_csv(DATA_DIR / "rewiring_timeseries.csv")
    degree_obs = pd.read_csv(DATA_DIR / "final_degree_histograms.csv")

    observed_replicates = build_observed_replicates(infected_obs, rewiring_obs, degree_obs)
    observed_summary = summarize_replicates(observed_replicates)
    summary_scale = run_pilot()

    history, posterior = run_smc_abc(
        observed_summary=observed_summary,
        summary_scale=summary_scale,
        selected_columns=SUMMARY_COLUMNS,
    )

    print("SMC-ABC summary:", flush=True)
    print(history, flush=True)
    print("Final weighted means:", flush=True)
    means = {
        name: np.average(posterior[name], weights=posterior["weight"])
        for name in PARAM_NAMES
    }
    print(means, flush=True)
