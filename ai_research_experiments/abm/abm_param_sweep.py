import logging
import random
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from toolz import pipe
from functools import partial
from typing import Dict
import uuid
import pickle

import ai_research_experiments.abm.abm_utils as abm_ut
from ai_research_experiments.abm.abm import SearchSpace
from ai_research_experiments import PROJECT_DIR
from numpy import VisibleDeprecationWarning

import warnings

warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)


if __name__ == "__main__":

    reward_sample = pipe(
        {
            "seed_n": range(1, 10, 1),
            "surprise_rew": np.arange(0, 0.0011, 0.0001),
            "scale_rew": np.arange(0.01, 0.11, 0.01),
            "surprise_bench": np.arange(0, 0.0011, 0.0001),
            "scale_bench": np.arange(0.01, 0.11, 0.01),
            "scale_bench_rew": np.arange(0.01, 0.11, 0.01),
        },
        abm_ut.generate_alternative_rewards,
        partial(random.sample, k=40),
    )

    space_iterations = 3
    search_iterations = 15

    agent_sample = pipe(
        {
            "size": 10,
            "search_span": range(1, 101, 10),
            "reward_weight": np.arange(0, 1.1, 0.1),
        },
        abm_ut.generate_alternative_agents,
        partial(random.sample, k=40),
    )
    logging.info(len(reward_sample) * len(agent_sample))

    logging.info("Starting parameter sweep")

    rewards_df = []
    rewards_discoveries_map: Dict = {}

    space_iterations = 3
    search_iterations = 15

    for reward in reward_sample:
        logging.info(reward)

        for _ in range(space_iterations):

            space = SearchSpace(size=50, iterations=search_iterations, **reward)
            space.seed_rewards()
            space.propagate_rewards()

            space_id = str(uuid.uuid1())

            rewards_discoveries_map[space_id] = {}
            rewards_discoveries_map[space_id]["rewards"] = space.plot_rewards()
            rewards_discoveries_map[space_id]["discoveries"] = {}

            for agent in agent_sample:

                for n, a_version in enumerate(agent):
                    parallel_outputs = Parallel(n_jobs=6)(
                        delayed(abm_ut.execute_strategy)(
                            space, a_version, iterations=search_iterations
                        )
                        for _ in range(3)
                    )

                    # FIX this ugliness
                    label_dict = {
                        **reward,
                        **a_version[f"group_{str(n+1)}"],
                        **space.reward_distr,
                        **{
                            "reward_cluster_n": space.reward_clusters_n,
                            "all_cluster_n": space.all_clusters_n,
                            "reward_bench_dist": space.rew_bench_mean_dist,
                        },
                    }

                    for results in parallel_outputs:
                        rewards_df.append(
                            pipe(
                                results[0],
                                partial(abm_ut.label_df, label_dict=label_dict),
                            ).assign(space_id=space_id)
                        )
                        rewards_discoveries_map[space_id]["discoveries"].update(
                            **results[1]
                        )

    scores_df = (
        pd.concat(rewards_df)
        .reset_index(drop=True)
        .rename(columns={0: "empty", 1: "benchmarks", 2: "applications"})
    )

    scores_df.to_csv(f"{PROJECT_DIR}/data/abm_scores_test.csv", index=False)

    with open(f"{PROJECT_DIR}/data/abm_discoveries_test.p", "wb") as f:
        pickle.dump(rewards_discoveries_map, f)
