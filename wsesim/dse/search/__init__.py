"""Search strategy registry."""

from wsesim.dse.search.base import SearchStrategy
from wsesim.dse.search.bayesian import BayesianSearch
from wsesim.dse.search.genetic import GeneticSearch
from wsesim.dse.search.grid import GridSearch
from wsesim.dse.search.random import RandomSearch

__all__ = [
    "SearchStrategy",
    "GridSearch",
    "RandomSearch",
    "BayesianSearch",
    "GeneticSearch",
]
