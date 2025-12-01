# 2D Ising MCMC

2D Ising Model simulation and analysis using Monte Carlo methods.

## Project Structure

```
2d-ising-mcmc/
├── src/
│   └── ising_mcmc/
│       ├── __init__.py       # Package initialization and exports
│       ├── __main__.py       # Entry point for running tasks
│       ├── ising_model.py    # IsingModel class implementation
│       ├── clusters.py       # Cluster finding algorithm
│       └── tasks.py          # Task functions for analysis
├── pyproject.toml            # Project configuration
└── README.md
```

## Installation

This project uses `uv` for package management. To install:

```bash
uv sync
```

## Usage

Run all tasks:

```bash
uv run python -m ising_mcmc
```

Or use the console script:

```bash
uv run 2d-ising-mcmc
```

Use as a library:

```python
from ising_mcmc import IsingModel, find_clusters

model = IsingModel(N=50, T=2.27)
model.run_mcmc(steps=10000)
clusters = find_clusters(model.config)
```

## Tasks

- **Task 2**: Exploring MCMC parameters (acceptance rates, equilibration)
- **Task 3**: Statistical properties and critical temperature estimation
- **Task 4**: Cluster size analysis at different temperatures
