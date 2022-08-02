# Utils to implement the ABM

import logging
import random
from functools import partial
from typing import Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
from scipy.ndimage import label
from toolz import pipe

from sklearn.metrics import pairwise_distances


def find_neighbours(loc: Tuple, span: int) -> List:
    """
    Finds a cell's neighbours

    Args:
        loc: coordinates of the cell whose neighbours we want to find
        span: the span of the grid. We ignore neighbours outside the span

    Returns:
        Coordinates for neighbours
    """

    west = (loc[0] - 1, loc[1])
    northwest = (loc[0] - 1, loc[1] - 1)
    southwest = (loc[0] - 1, loc[1] + 1)
    east = (loc[0] + 1, loc[1])
    northeast = (loc[0] + 1, loc[1] + 1)
    southeast = (loc[0] + 1, loc[1] + 1)
    north = (loc[0], loc[1] - 1)
    south = (loc[0], loc[1] + 1)

    return [
        k
        for k in [west, northwest, southwest, east, northeast, southeast, north, south]
        if all(x in range(span) for x in k)
    ]


def make_df(tuple_dict: Dict) -> pd.DataFrame:
    """
    Creates a dataframe from a tuple with cells and their status

    Args:
        tuple_dict: dict with cells locations and their status
    """
    return pd.concat(
        [
            pd.DataFrame(tuple_dict.keys(), columns=["x", "y"]),
            pd.DataFrame(tuple_dict.values(), columns=["status"]),
        ],
        axis=1,
    )


def plot_status_space(df: pd.DataFrame, domain: List, range: List) -> alt.Chart:
    """Plots the status space"""
    return (
        alt.Chart(df)
        .mark_rect(stroke="lightgrey", strokeWidth=0.2)
        .encode(
            x=alt.X("x:N", axis=alt.Axis(labels=False, ticks=False), title=None),
            y=alt.Y("y:N", axis=alt.Axis(labels=False, ticks=False), title=None),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(domain=domain, range=range)
                # legend=None
            ),
        )
        .properties(width=200, height=200)
    )


def expected_rewards(
    discoveries: Dict, size: int, reward_weight: float, **rews
) -> list:
    """Estimates expected rewards for all cells in a space

    Args:
        discoveries: dict of discoveries so far
        size: size of the space (to ignore out of space neighbours)
        reward_weight: weight towards looking for rewards vs. looking for benchmarks
        rews: keywords describing the reward generation process

    Returns:
        A list of expected rewards for all cells
        that remain unexplored
    """

    vector_probs = []

    for loc in discoveries.keys():
        if discoveries[loc] != 0:
            pass  # We don't want to estimate rewards for already discovered cells
        else:
            # We add this control flow to avoid searching if an agent is not interested in a type of reward
            if reward_weight != 0:
                prob_rew = rews["surprise_rew"] + rews["scale_rew"] * sum(
                    [
                        discoveries[neigh] == 2
                        for neigh in find_neighbours(loc, span=size)
                    ]
                )
            else:
                prob_rew = 0

            if reward_weight != 1:
                prob_bench = rews["surprise_bench"] + (
                    rews["scale_bench"]
                    * sum(
                        [
                            discoveries[neigh] == 1
                            for neigh in find_neighbours(loc, span=size)
                        ]
                    )
                    + rews["scale_bench_rew"]
                    * sum(
                        [
                            discoveries[neigh] == 2
                            for neigh in find_neighbours(loc, span=size)
                        ]
                    )
                )
            else:
                prob_bench = 0

            prob_weighted = reward_weight * prob_rew + (1 - reward_weight) * prob_bench

            vector_probs.append([loc, prob_weighted])

    return vector_probs


def select_search_location(expected_rewards: List, top_n: int = 10) -> Tuple:
    """Selects the search location

    Args:
        expected_rewards: vector of expected rewards
        top_n: top rewards among which one is chosen randomly

    Argds:
        Search location
    """

    sorted_rews = sorted(expected_rewards, key=lambda x: x[1], reverse=True)

    # This samples a cell from the top_n also taking into account any cells
    # with the same expected rewards as the lowest in the top_n

    return random.sample([x for x in sorted_rews if x[1] >= sorted_rews[top_n][1]], 1)[
        0
    ]


def find_reward_clusters(rewards: Dict, category: int) -> np.array:
    """Use image segmentation to detect the number of clusters in a reward space

    Args:
        rewards: dict with cells and their reward status
    """

    rew_grid = pd.DataFrame([k + (v,) for k, v in rewards.items()]).pivot_table(
        index=0, columns=1, values=2
    )

    return label(
        rew_grid.applymap(lambda x: x == category),
        structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    )


def plot_reward_clusters(cluster_grid: np.array) -> alt.Chart:
    """Plots the location of reward clusters

    Args:
        cluster_grid: an array assigning each cell to a cluster
    """

    return (
        alt.Chart(
            pd.DataFrame(cluster_grid)
            .stack()
            .reset_index(name="cluster")
            .rename(columns={"level_0": "x", "level_1": "y"})
        )
        .mark_rect()
        .encode(
            x=alt.X("x:N", axis=alt.Axis(labels=False, ticks=False), title=None),
            y=alt.Y("y:N", axis=alt.Axis(labels=False, ticks=False), title=None),
            color=alt.Color("cluster:N", legend=None),
        )
        .properties(width=200, height=200)
    )


def count_clusters_found(
    reward_clusters: np.array, discoveries: Dict, category: int
) -> int:
    """Counts the number of clusters discovered by a solution

    Args:
        reward_cluster: array of cells-clusters
        discoveries: dict of cells and their status

    Returns number of clusters found
    """

    clust_locs = {
        (x[0], x[1]): x[2]
        for x in pd.DataFrame(reward_clusters).stack().reset_index().to_numpy()
    }

    return pipe(
        [clust_locs[k] for k, v in discoveries.items() if v == category], set, len
    )


def discovery_metrics(discoveries: Dict, rewards: Dict, category: int) -> pd.DataFrame:
    """Metrics of discovery efficiency and diversity

    Args:
        discoveries: dict of cells and their status
        rewards: dict of cells and their reward status
        category: category of reward we are interested in

    Returns:
        A dataframe with results

    """

    # % of all categories discovered
    discov_share = sum([x == category for x in discoveries.values()]) / sum(
        [r == category for r in rewards.values()]
    )

    # % of all searches attempted which are successful
    efficiency = sum([x == category for x in discoveries.values()]) / (
        sum([x != 0 for x in discoveries.values()])
    )

    # Num clusters found
    clust_grid, num_clust = find_reward_clusters(rewards, category=2)

    share_clusters_found = (
        count_clusters_found(clust_grid, discoveries, category=category) / num_clust
    )

    # Mean distance between discoveries
    # We need at least one 2 discoveries to calculate pairwise discovery metrics
    if len(make_df(discoveries).query(f"status=={category}")) < 2:
        logging.info("Not enough discoveries to calculate mean discovery distance")
        mean_discov_dist = np.nan
    else:
        mean_discov_dist = pipe(
            make_df(discoveries).query(f"status=={category}")[["x", "y"]].to_numpy(),
            partial(pairwise_distances, metric="cosine"),
            np.mean,
        )

    return pd.DataFrame(
        [
            [category, "discovery_share", discov_share],
            [category, "efficiency", efficiency],
            [category, "share_cluster_found", share_clusters_found],
            [category, "mean_pw_discov_dist", mean_discov_dist],
        ],
        columns=["category", "variable", "value"],
    )
