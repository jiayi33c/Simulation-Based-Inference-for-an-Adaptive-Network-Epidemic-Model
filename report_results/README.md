# Report Results

This folder contains curated CSV outputs used for the report figures, tables, and robustness checks.

Large raw simulation files are intentionally excluded. In particular, the full `1,000,000`-proposal rejection ABC result and raw batch files are not stored here because they are large and can be regenerated from the scripts.

## Summary Files

- `method_comparison_summary.csv`: Posterior means, standard deviations, and acceptance or retention rates for the main methods.
- `rejection_abc/final_accepted_infected_plus_rewiring.csv`: Final retained rejection ABC posterior samples from the `1,000,000`-proposal run.
- `abc_mcmc/abc_mcmc_posterior_infected_plus_rewiring.csv`: Final retained ABC-MCMC posterior samples.
- `abc_mcmc/abc_mcmc_tuning_summary.csv`: ABC-MCMC tuning summary over tolerance and proposal scale choices.
- `abc_mcmc/abc_mcmc_tuning_metrics.csv`: Per-chain ABC-MCMC tuning diagnostics.
- `smc_abc/smc_abc_stage_*.csv`: SMC-ABC particle populations for each tolerance stage.
- `smc_abc/smc_abc_tuning_summary.csv`: SMC-ABC tolerance schedule, acceptance rates, and weighted posterior summaries.
- `regression_adjustment/regression_adjusted_posterior.csv`: Local weighted regression-adjusted ABC posterior.
- `regression_adjustment/regression_adjustment_bandwidth_sensitivity.csv`: Sensitivity of the regression-adjusted posterior to the local bandwidth.
- `synthetic_likelihood/synthetic_likelihood_posterior.csv`: Final synthetic-likelihood MCMC posterior samples.
- `synthetic_likelihood/synthetic_likelihood_metrics.csv`: Per-chain synthetic-likelihood diagnostics.
- `synthetic_likelihood/synthetic_likelihood_chain.csv`: Full synthetic-likelihood chain output.
- `synthetic_recovery/synthetic_recovery_accepted.csv`: Accepted posterior samples from the synthetic-truth recovery experiment.
- `synthetic_recovery/synthetic_recovery_truth.csv`: Parameter values used to generate the synthetic-truth dataset.
- `synthetic_recovery/synthetic_recovery_summary.csv`: Synthetic-truth recovery truth values and recovered posterior summaries.
- `posterior_predictive/*.csv`: Observed mean curves/histogram and posterior predictive draw means used for posterior predictive checks.

## Primary Estimate

The report treats ABC-MCMC as the primary final posterior estimate because it improves on independent rejection sampling by using a Markov chain proposal while remaining close to the stable `1,000,000`-proposal rejection ABC baseline.

SMC-ABC, local regression adjustment, and synthetic likelihood are included as robustness or exploratory checks.
