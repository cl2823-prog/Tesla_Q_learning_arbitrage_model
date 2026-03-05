# BESS Q-Learning Arbitrage (ERCOT Hub North)

This repository trains a tabular Q-learning policy for battery arbitrage using 15-minute ERCOT Hub North prices.

## What this project does
- Trains on `2016-2024` historical prices.
- Tests out-of-sample on `2025`.
- Decision every 15 minutes: `charge`, `idle`, or `discharge`.
- Models battery degradation with full equivalent cycles (FEC) and SOH updates.
- Applies reward penalties for degradation, cycle pacing, and terminal SOC mismatch.

## Repository layout
- `battery_q_learning.jl`: main training/evaluation script
- `battery_q_learning.ipynb`: notebook wrapper
- `data/hub_north_2021_2026.xlsx`: input dataset
- `results/`: generated outputs

## Requirements
- Julia `1.12+`
- Julia packages: `CSV`, `DataFrames`, `XLSX`, `Statistics`, `Dates`, `Random`

Install once in Julia:

```julia
using Pkg
Pkg.add(["CSV", "DataFrames", "XLSX"])
```

## Run
From repository root:

```bash
julia battery_q_learning.jl
```

Default behavior:
- train years: `2016:2024`
- test year: `2025`
- episodes: `50`
- reads data from `data/hub_north_2021_2026.xlsx`
- writes outputs to `results/`

## Outputs
After running, these files are generated in `results/`:
- `q_learning_2025_timeline.csv`
- `training_episode_log.csv` (or fallback `training_episode_log_new.csv` if file is locked)

Existing analysis artifacts:
- `results/bin_search_2024_validation.csv`
- `results/training_results_50_episodes.xlsx`

## Reproducibility note
Paths are relative (`data/...`, `results/...`) to avoid machine-specific path issues.

## Data note
If your organization does not allow publishing raw data, remove files under `data/` and provide internal instructions for where to place the dataset.
