from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import math
import numpy as np
import pandas as pd

from simulator import simulate


PROJECT_ROOT = Path.cwd()
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

N_CHAINS = 4
N_STEPS = 4000
BURN_IN = 1000
THIN = 5
SL_REPS = 80
SEED = 2026
N_WORKERS = 4
PROPOSAL_SCALE_FACTOR = 0.6
COV_REG = 1e-4


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


def simulate_replicates(beta: float, gamma: float, rho: float, n_reps: int, seed: int):
    rng = np.random.default_rng(seed)
    return [
        simulate_one(beta, gamma, rho, int(rng.integers(0, 2**31 - 1)))
        for _ in range(n_reps)
    ]


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


def replicate_summary_matrix(replicates, selected_columns):
    per_rep = pd.DataFrame([summarize_replicate(rep) for rep in replicates])
    return per_rep[selected_columns].to_numpy(dtype=float)


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


def in_prior_bounds(theta):
    return all(
        PRIOR_BOUNDS[name][0] <= value <= PRIOR_BOUNDS[name][1]
        for name, value in zip(PARAM_NAMES, theta)
    )


def log_prior(theta):
    if in_prior_bounds(theta):
        return 0.0
    return -np.inf


def mvn_logpdf(x, mean, cov):
    diff = x - mean
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return -np.inf
    inv_cov = np.linalg.inv(cov)
    dim = len(x)
    return -0.5 * (
        dim * math.log(2.0 * math.pi)
        + logdet
        + diff @ inv_cov @ diff
    )


def synthetic_loglik(theta, observed_summary, selected_columns, n_reps, seed):
    beta, gamma, rho = theta
    reps = simulate_replicates(beta, gamma, rho, n_reps=n_reps, seed=seed)
    sim_matrix = replicate_summary_matrix(reps, selected_columns)

    mean_vec = sim_matrix.mean(axis=0)
    cov = np.cov(sim_matrix, rowvar=False)
    cov = np.atleast_2d(cov)
    cov = cov + np.eye(cov.shape[0]) * COV_REG

    s_obs = observed_summary[selected_columns].to_numpy(dtype=float)
    return mvn_logpdf(s_obs, mean_vec, cov)


def run_chain(args):
    chain_id, observed_summary, selected_columns, start_theta, proposal_scale, n_steps, burn_in, thin, seed = args
    rng = np.random.default_rng(seed)

    current = np.asarray(start_theta, dtype=float).copy()
    current_lp = log_prior(current)
    if not np.isfinite(current_lp):
        raise ValueError("Starting point is outside the prior bounds.")
    current_ll = synthetic_loglik(
        current,
        observed_summary=observed_summary,
        selected_columns=selected_columns,
        n_reps=SL_REPS,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    current_logpost = current_lp + current_ll

    chain_rows = []
    accepted_moves = 0

    for step in range(1, n_steps + 1):
        proposal = current + rng.normal(loc=0.0, scale=proposal_scale, size=len(PARAM_NAMES))
        proposal_lp = log_prior(proposal)

        accepted = 0
        proposal_ll = np.nan

        if np.isfinite(proposal_lp):
            proposal_ll = synthetic_loglik(
                proposal,
                observed_summary=observed_summary,
                selected_columns=selected_columns,
                n_reps=SL_REPS,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            proposal_logpost = proposal_lp + proposal_ll
            log_alpha = proposal_logpost - current_logpost

            if math.log(rng.uniform()) < log_alpha:
                current = proposal
                current_ll = proposal_ll
                current_logpost = proposal_logpost
                accepted = 1
                accepted_moves += 1

        row = {
            "step": step,
            "beta": current[0],
            "gamma": current[1],
            "rho": current[2],
            "log_synthetic_likelihood": current_ll,
            "accepted_move": accepted,
            "chain_id": chain_id,
        }
        chain_rows.append(row)

        if step % 500 == 0:
            print(
                f"Completed {step} / {n_steps} SL-MCMC steps for chain {chain_id}",
                flush=True,
            )

    chain = pd.DataFrame(chain_rows)
    posterior = chain.loc[chain["step"] > burn_in].iloc[::thin].reset_index(drop=True)
    acceptance_rate = accepted_moves / n_steps

    metrics = {
        "chain_id": chain_id,
        "acceptance_rate": acceptance_rate,
        "retained_samples": len(posterior),
        "beta_mean": posterior["beta"].mean(),
        "gamma_mean": posterior["gamma"].mean(),
        "rho_mean": posterior["rho"].mean(),
        "beta_std": posterior["beta"].std(),
        "gamma_std": posterior["gamma"].std(),
        "rho_std": posterior["rho"].std(),
    }
    return chain, posterior, metrics


if __name__ == "__main__":
    infected_obs = pd.read_csv(DATA_DIR / "infected_timeseries.csv")
    rewiring_obs = pd.read_csv(DATA_DIR / "rewiring_timeseries.csv")
    degree_obs = pd.read_csv(DATA_DIR / "final_degree_histograms.csv")

    observed_replicates = build_observed_replicates(infected_obs, rewiring_obs, degree_obs)
    observed_summary = summarize_replicates(observed_replicates)

    accepted_path = RESULTS_DIR / "final_accepted_infected_plus_rewiring.csv"
    accepted = pd.read_csv(accepted_path)

    best_row = accepted.sort_values("distance").iloc[0]
    start_theta = best_row[PARAM_NAMES].to_numpy(dtype=float)
    proposal_scale = (
        PROPOSAL_SCALE_FACTOR * accepted[PARAM_NAMES].std().clip(lower=pd.Series([0.002, 0.002, 0.005], index=PARAM_NAMES))
    ).to_numpy(dtype=float)

    jobs = [
        (
            chain_id,
            observed_summary,
            SUMMARY_COLUMNS,
            start_theta,
            proposal_scale,
            N_STEPS,
            BURN_IN,
            THIN,
            SEED + chain_id,
        )
        for chain_id in range(N_CHAINS)
    ]

    with ProcessPoolExecutor(max_workers=min(N_WORKERS, N_CHAINS)) as ex:
        results = list(ex.map(run_chain, jobs))

    chains = []
    posteriors = []
    metrics = []
    for chain, posterior, metric in results:
        chains.append(chain)
        posteriors.append(posterior)
        metrics.append(metric)
        print(
            f"Finished chain {metric['chain_id']}, acceptance_rate={metric['acceptance_rate']:.3f}",
            flush=True,
        )

    full_chain = pd.concat(chains, ignore_index=True)
    posterior_df = pd.concat(posteriors, ignore_index=True)
    metrics_df = pd.DataFrame(metrics)

    full_chain.to_csv(RESULTS_DIR / "synthetic_likelihood_chain.csv", index=False)
    posterior_df.to_csv(RESULTS_DIR / "synthetic_likelihood_posterior.csv", index=False)
    metrics_df.to_csv(RESULTS_DIR / "synthetic_likelihood_metrics.csv", index=False)

    print("Synthetic likelihood MCMC summary:", flush=True)
    print(metrics_df.to_string(index=False), flush=True)
