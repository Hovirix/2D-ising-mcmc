"""2D Ising MCMC: 2D Ising Model simulation using Monte Carlo methods."""

from .clusters import find_clusters
from .ising_model import IsingModel

__version__ = "0.1.0"

__all__ = ["IsingModel", "find_clusters"]
