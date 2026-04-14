from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

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

N_REPS = 40
PILOT_SAMPLES = 20
N_STEPS = 5000
BURN_IN = 1000
THIN = 5
SEED = 2025
PROPOSAL_FACTORS = (0.8,)
EPSILON_QUANTILES = (0.9, 0.75)
N_CHAINS = 4


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
        "final": float(x[-1]),
        "auc": float(np.trapezoid(x)),
        "mean": float(np.mean(x)),
        "var": float(np.var(x)),
        "zero_fraction": float(np.mean(x == 0)),
        "extinction_time": extinction_time,
    }


def summarize_degree_histogram(degree_hist):
    counts = np.asarray(degree_hist, dtype=float)
    degree_values = np.arange(len(counts), dtype=float)
    total_nodes = counts.sum()

    if total_nodes == 0:
        raise ValueError("Degree histogram is empty.")

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
        infected = (
            infected_df.loc[infected_df["replicate_id"] == rep_id]
            .sort_values("time")["infected_fraction"]
            .to_numpy()
        )
        rewiring = (
            rewiring_df.loc[rewiring_df["replicate_id"] == rep_id]
            .sort_values("time")["rewire_count"]
            .to_numpy()
        )
        degrees = (
            degree_df.loc[degree_df["replicate_id"] == rep_id]
            .sort_values("degree")["count"]
            .to_numpy()
        )

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


def weighted_distance(sim_summary, obs_summary, scale, selected_columns=None):
    if selected_columns is None:
        selected_columns = obs_summary.index
    diff = (sim_summary[selected_columns] - obs_summary[selected_columns]) / scale[selected_columns]
    return float(np.sqrt(np.sum(np.square(diff))))


def run_pilot(n_pilot: int = PILOT_SAMPLES, n_reps: int = N_REPS, seed: int = 123):
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


def simulate_summary_distance(beta, gamma, rho, observed_summary, summary_scale, selected_columns, n_reps=N_REPS, seed=2024):
    reps = simulate_replicates(beta, gamma, rho, n_reps=n_reps, seed=seed)
    sim_summary = summarize_replicates(reps)
    dist = weighted_distance(sim_summary, observed_summary, summary_scale, selected_columns)
    return sim_summary, dist


def in_prior_bounds(theta):
    return all(
        PRIOR_BOUNDS[name][0] <= value <= PRIOR_BOUNDS[name][1]
        for name, value in zip(["beta", "gamma", "rho"], theta)
    )


def abc_mcmc(
    observed_summary,
    summary_scale,
    selected_columns,
    start_theta,
    epsilon,
    proposal_scale,
    n_steps=N_STEPS,
    n_reps=N_REPS,
    burn_in=BURN_IN,
    thin=THIN,
    seed=SEED,
):
    rng = np.random.default_rng(seed)
    current = np.asarray(start_theta, dtype=float).copy()
    _, current_dist = simulate_summary_distance(
        current[0], current[1], current[2],
        observed_summary=observed_summary,
        summary_scale=summary_scale,
        selected_columns=selected_columns,
        n_reps=n_reps,
        seed=int(rng.integers(0, 2**31 - 1)),
    )

    rows = []
    accepted_moves = 0

    for step in range(n_steps):
        proposal = current + rng.normal(scale=proposal_scale, size=3)
        accepted = False

        if in_prior_bounds(proposal):
            _, proposal_dist = simulate_summary_distance(
                proposal[0], proposal[1], proposal[2],
                observed_summary=observed_summary,
                summary_scale=summary_scale,
                selected_columns=selected_columns,
                n_reps=n_reps,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            if proposal_dist <= epsilon:
                current = proposal
                current_dist = proposal_dist
                accepted = True
                accepted_moves += 1

        rows.append({
            "step": step,
            "beta": current[0],
            "gamma": current[1],
            "rho": current[2],
            "distance": current_dist,
            "accepted_move": int(accepted),
        })

        if (step + 1) % 500 == 0:
            print(f"Completed {step + 1} / {n_steps} MCMC steps", flush=True)

    chain = pd.DataFrame(rows)
    posterior = chain.iloc[burn_in::thin].reset_index(drop=True)
    acceptance_rate = chain["accepted_move"].mean()
    return chain, posterior, acceptance_rate


def run_chain(args):
    (
        epsilon_quantile,
        proposal_factor,
        chain_id,
        observed_summary,
        summary_scale,
        start_theta,
        epsilon,
        posterior_std,
    ) = args

    proposal_scale = (proposal_factor * posterior_std).clip(min=0.002)
    chain, posterior, acceptance_rate = abc_mcmc(
        observed_summary=observed_summary,
        summary_scale=summary_scale,
        selected_columns=SUMMARY_COLUMNS,
        start_theta=start_theta,
        epsilon=epsilon,
        proposal_scale=proposal_scale,
        seed=SEED + 1000 * chain_id + int(proposal_factor * 100),
    )

    chain["chain_id"] = chain_id
    chain["epsilon_quantile"] = epsilon_quantile
    chain["proposal_factor"] = proposal_factor
    posterior["chain_id"] = chain_id
    posterior["epsilon_quantile"] = epsilon_quantile
    posterior["proposal_factor"] = proposal_factor

    metrics = {
        "epsilon_quantile": epsilon_quantile,
        "proposal_factor": proposal_factor,
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


def main():
    infected_obs = pd.read_csv(DATA_DIR / "infected_timeseries.csv")
    rewiring_obs = pd.read_csv(DATA_DIR / "rewiring_timeseries.csv")
    degree_obs = pd.read_csv(DATA_DIR / "final_degree_histograms.csv")

    observed_replicates = build_observed_replicates(infected_obs, rewiring_obs, degree_obs)
    observed_summary = summarize_replicates(observed_replicates)
    summary_scale = run_pilot()

    accepted_path = RESULTS_DIR / "final_accepted_infected_plus_rewiring.csv"
    final_accepted = pd.read_csv(accepted_path)

    start_theta = final_accepted.sort_values("distance").iloc[0][["beta", "gamma", "rho"]].to_numpy()
    posterior_std = final_accepted[["beta", "gamma", "rho"]].std().to_numpy()

    jobs = [
        (
            epsilon_quantile,
            proposal_factor,
            chain_id,
            observed_summary,
            summary_scale,
            start_theta,
            final_accepted["distance"].quantile(epsilon_quantile),
            posterior_std,
        )
        for epsilon_quantile in EPSILON_QUANTILES
        for proposal_factor in PROPOSAL_FACTORS
        for chain_id in range(N_CHAINS)
    ]

    all_chains = []
    all_posteriors = []
    metrics_rows = []

    with ProcessPoolExecutor(max_workers=min(N_CHAINS, len(jobs))) as ex:
        for chain, posterior, metrics in ex.map(run_chain, jobs):
            all_chains.append(chain)
            all_posteriors.append(posterior)
            metrics_rows.append(metrics)
            print(
                f"Finished epsilon_quantile={metrics['epsilon_quantile']}, "
                f"proposal_factor={metrics['proposal_factor']}, "
                f"chain_id={metrics['chain_id']}, "
                f"acceptance_rate={metrics['acceptance_rate']:.3f}",
                flush=True,
            )

    all_chains_df = pd.concat(all_chains, ignore_index=True)
    all_posteriors_df = pd.concat(all_posteriors, ignore_index=True)
    metrics_df = (
        pd.DataFrame(metrics_rows)
        .sort_values(["epsilon_quantile", "proposal_factor", "chain_id"])
        .reset_index(drop=True)
    )
    summary_df = (
        metrics_df.groupby(["epsilon_quantile", "proposal_factor"], as_index=False)
        .agg({
            "acceptance_rate": "mean",
            "retained_samples": "sum",
            "beta_mean": "mean",
            "gamma_mean": "mean",
            "rho_mean": "mean",
            "beta_std": "mean",
            "gamma_std": "mean",
            "rho_std": "mean",
        })
    )

    all_chains_df.to_csv(RESULTS_DIR / "abc_mcmc_chain_infected_plus_rewiring.csv", index=False)
    all_posteriors_df.to_csv(RESULTS_DIR / "abc_mcmc_posterior_infected_plus_rewiring.csv", index=False)
    metrics_df.to_csv(RESULTS_DIR / "abc_mcmc_tuning_metrics.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "abc_mcmc_tuning_summary.csv", index=False)

    print("Tuning summary:", flush=True)
    print(summary_df, flush=True)


if __name__ == "__main__":
    main()
