# Implementation of SearchSpace class

from collections import Counter
from itertools import product
import random
from toolz import pipe
from functools import partial
from typing import Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd

import ai_research_experiments.abm.abm_utils as abm_ut


class SearchSpace:
    """Class defining a space of rewards, discoveries and search"""

    def __init__(self, size: int, iterations: int, **kwargs):
        """
        Args:
            size: dimensions of the square grid
            seed_n: number of rewards seed
            iterations: number of iterations to generate a reward space

        """
        self.size: int = size
        self.seed_n: int = kwargs["seed_n"]

        self.iterations: int = iterations
        self.rewards: Dict[Tuple, int] = {
            tuple(loc): 0 for loc in product(range(size), range(size))
        }
        self.discoveries: Dict[Tuple, int] = {
            tuple(loc): 0 for loc in product(range(size), range(size))
        }
        self.search_params: Dict[str, float] = kwargs
        self.metrics: List = []

    def set_search_params(self, **kwargs):
        """Set search params"""

        self.__dict__.update(kwargs)

        # Debugging: what is the maximum probability of a cell being a reward?
        self.max_prob_rew = self.surprise_rew + 6 * self.scale_rew
        self.max_prob_bench = (
            self.surprise_bench + 6 * self.scale_bench + 6 * self.scale_bench_rew
        )

    def seed_rewards(self):
        """Seed rewards in the space"""

        self.seeds = [
            (x, y)
            for x, y in zip(
                random.sample(range(self.size), self.seed_n),
                random.sample(range(self.size), self.seed_n),
            )
        ]

        self.rewards = {
            loc: 2 if loc in self.seeds else 0 for loc in self.rewards.keys()
        }

        return self

    def propagate_rewards(self):
        """Propagate rewards through the space"""

        it = 0
        while it < self.iterations:
            it += 1
            for loc in self.rewards.keys():
                if self.rewards[loc] != 0:
                    pass
                else:
                    # A cell has a base probability of acquiring a reward + a prob which is function of
                    # whether its neighbours are rewards

                    p = self.search_params["surprise_rew"] + self.search_params[
                        "scale_rew"
                    ] * sum(
                        [
                            self.rewards[neigh] == 2
                            for neigh in abm_ut.find_neighbours(loc, span=self.size)
                        ]
                    )

                    if random.random() < p:
                        self.rewards[loc] = 2

                    # If a cell isn't a reward, it might still be a benchmark

                    else:
                        p_2 = self.search_params["surprise_bench"] + (
                            self.search_params["scale_bench"]
                            * sum(
                                [
                                    self.rewards[neigh] == 1
                                    for neigh in abm_ut.find_neighbours(
                                        loc, span=self.size
                                    )
                                ]
                            )
                            + self.search_params["scale_bench_rew"]
                            * sum(
                                [
                                    self.rewards[neigh] == 2
                                    for neigh in abm_ut.find_neighbours(
                                        loc, span=self.size
                                    )
                                ]
                            )
                        )

                        # The benchmark gets second dibs. Is this
                        # a problem?

                        if random.random() < p_2:
                            self.rewards[loc] = 1

        # Also store the reward distribution
        self.reward_distr = dict(Counter(self.rewards.values()))
        self.reward_clusters_n = abm_ut.find_reward_clusters(self.rewards, [2])[1]
        self.all_clusters_n = abm_ut.find_reward_clusters(self.rewards, [1, 2])[1]
        self.rew_bench_mean_dist = abm_ut.calculate_rew_disc_dist(self.rewards)

        return self

    def set_rewards(self, rewards_dict: Dict):
        """We can also choose to set rewards e.g. for reproducibility and comparisons"""

        self.rewards = rewards_dict
        self.reward_distr = dict(Counter(self.rewards.values()))
        self.reward_clusters_n = abm_ut.find_reward_clusters(self.rewards, 2)[1]

    def make_agents(self, agent_dict: Dict):
        """Create agents

        Args:
            agent_characteristics: a dict where every element is a group with the
            characteristics of its agents.
        """

        self.agents = {
            i: {
                **{"group": k},
                **{k_2: v_2 for k_2, v_2 in v.items() if k_2 != "size"},
                **{"discoveries": []},
            }
            for k, v in agent_dict.items()
            for i in range(v["size"])
        }

    def search(self, iterations: int = 10):
        """Agents search for rewards

        Args:
            iterations: number of iterations to search

        """

        it = 0
        while it < iterations:
            it += 1
            for i, v in self.agents.items():

                select = pipe(
                    abm_ut.expected_rewards(
                        self.discoveries,
                        self.size,
                        v["reward_weight"],
                        **self.search_params
                    ),
                    partial(abm_ut.select_search_location, top_n=v["search_span"]),
                )[0]

                if self.rewards[select] == 0:
                    outcome = -1
                elif self.rewards[select] == 1:
                    outcome = 1
                elif self.rewards[select] == 2:
                    outcome = 2

                self.agents[i]["discoveries"].append([select, outcome])
                self.discoveries[select] = outcome

    def plot_rewards(self) -> alt.Chart:
        """Plot rewards"""

        return abm_ut.plot_status_space(
            abm_ut.make_df(self.rewards).assign(
                status=lambda df: df["status"].replace(
                    {0: "empty", 1: "benchmark", 2: "discovery"}
                )
            ),
            domain=["discovery", "benchmark", "empty"],
            range=["red", "green", "beige"],
        )

    def plot_discoveries(self) -> alt.Chart:
        """Plot discoveries"""
        return abm_ut.plot_status_space(
            abm_ut.make_df(self.discoveries).assign(
                status=lambda df: df["status"].replace(
                    {-1: "useless", 0: "unexplored", 1: "benchmark", 2: "discovery"}
                )
            ),
            domain=["discovery", "benchmark", "useless", "unexplored"],
            range=["red", "green", "blue", "beige"],
        )

    def plot_reward_clusters(self, category: int) -> alt.Chart:
        """Plots the location of reward clusters

        Args:
            category: the category of reward clusters to plot
        """

        return abm_ut.plot_reward_clusters(
            abm_ut.find_reward_clusters(self.rewards, category)[0]
        )

    def make_metrics(self, categories: list = [2, 1]) -> pd.DataFrame:
        """Make discovery and diversity metrics

        Args:
            categories: list of categories to include in the metrics
            (1 is benchmarks, 2 is discoveries)

        """

        return pd.concat(
            [
                abm_ut.discovery_metrics(self.discoveries, self.rewards, category=c)
                for c in categories
            ]
        )

    def reset_discoveries(self):
        """Reset discoveries"""
        self.discoveries = {
            tuple(loc): 0 for loc in product(range(self.size), range(self.size))
        }
