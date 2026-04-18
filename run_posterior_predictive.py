from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from simulator import simulate

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

POSTERIOR_PATH = RESULTS_DIR / "abc_mcmc_posterior_infected_plus_rewiring.csv"

N_POSTERIOR_DRAWS = 80
N_REPS = 40
SEED = 2027
N_WORKERS = 28


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


def evaluate_draw(args):
    draw_id, beta, gamma, rho, seed = args
    reps = simulate_replicates(beta, gamma, rho, n_reps=N_REPS, seed=seed)

    infected_mat = np.vstack([rep["infected"] for rep in reps])
    rewiring_mat = np.vstack([rep["rewiring"] for rep in reps])
    degree_mat = np.vstack([rep["degrees"] for rep in reps])

    return {
        "draw_id": draw_id,
        "beta": beta,
        "gamma": gamma,
        "rho": rho,
        "infected_mean": infected_mat.mean(axis=0),
        "rewiring_mean": rewiring_mat.mean(axis=0),
        "degree_mean": degree_mat.mean(axis=0),
    }


def expand_series_rows(records, key, value_name, index_name):
    rows = []
    for record in records:
        values = record[key]
        for idx, value in enumerate(values):
            rows.append({
                "draw_id": record["draw_id"],
                "beta": record["beta"],
                "gamma": record["gamma"],
                "rho": record["rho"],
                index_name: idx,
                value_name: float(value),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    infected_obs = pd.read_csv(DATA_DIR / "infected_timeseries.csv")
    rewiring_obs = pd.read_csv(DATA_DIR / "rewiring_timeseries.csv")
    degree_obs = pd.read_csv(DATA_DIR / "final_degree_histograms.csv")

    posterior_df = pd.read_csv(POSTERIOR_PATH)
    posterior_draws = posterior_df.sample(n=N_POSTERIOR_DRAWS, random_state=SEED).reset_index(drop=True)

    obs_replicates = build_observed_replicates(infected_obs, rewiring_obs, degree_obs)
    obs_infected = np.vstack([rep["infected"] for rep in obs_replicates]).mean(axis=0)
    obs_rewiring = np.vstack([rep["rewiring"] for rep in obs_replicates]).mean(axis=0)
    obs_degree = np.vstack([rep["degrees"] for rep in obs_replicates]).mean(axis=0)

    rng = np.random.default_rng(SEED)
    jobs = [
        (idx, row.beta, row.gamma, row.rho, int(rng.integers(0, 2**31 - 1)))
        for idx, row in enumerate(posterior_draws.itertuples(index=False))
    ]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        records = list(ex.map(evaluate_draw, jobs, chunksize=5))

    infected_pp = expand_series_rows(records, "infected_mean", "infected_mean", "time")
    rewiring_pp = expand_series_rows(records, "rewiring_mean", "rewiring_mean", "time")
    degree_pp = expand_series_rows(records, "degree_mean", "degree_mean", "degree")

    pd.DataFrame({
        "time": np.arange(len(obs_infected)),
        "observed_infected_mean": obs_infected,
    }).to_csv(RESULTS_DIR / "pp_observed_infected_mean.csv", index=False)

    pd.DataFrame({
        "time": np.arange(len(obs_rewiring)),
        "observed_rewiring_mean": obs_rewiring,
    }).to_csv(RESULTS_DIR / "pp_observed_rewiring_mean.csv", index=False)

    pd.DataFrame({
        "degree": np.arange(len(obs_degree)),
        "observed_degree_mean": obs_degree,
    }).to_csv(RESULTS_DIR / "pp_observed_degree_mean.csv", index=False)

    infected_pp.to_csv(RESULTS_DIR / "pp_infected_draw_means.csv", index=False)
    rewiring_pp.to_csv(RESULTS_DIR / "pp_rewiring_draw_means.csv", index=False)
    degree_pp.to_csv(RESULTS_DIR / "pp_degree_draw_means.csv", index=False)
    posterior_draws.to_csv(RESULTS_DIR / "pp_sampled_posterior_draws.csv", index=False)

    print(f"Posterior predictive complete using {len(posterior_draws)} posterior draws.", flush=True)
    print("Saved observed means and posterior predictive draw means to results/.", flush=True)
