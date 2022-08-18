from typing import Dict, List

import altair as alt
import numpy as np
import pandas as pd
from itertools import chain, product, combinations
from collections import Counter
from toolz import pipe

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline

from scipy.spatial.distance import cosine


def get_concept(concept_dict: Dict, name: str):
    """Returns information for a concept dictionary"""

    return concept_dict[name]


def get_related_concepts(concept_dict: Dict, name: str):
    """Get concepts related to a concept"""

    return [
        c["display_name"] for c in get_concept(concept_dict, name)["related_concepts"]
    ]


def get_single_descendent(concept: Dict, ancestor_set: set):
    """Returns descendents with a single parent & parent in ancestor set"""

    if len(concept["ancestors"]) == 1:
        if any(c["display_name"] in ancestor_set for c in concept["ancestors"]):
            return concept["display_name"]


def get_all_single_descendents(concept_dict: Dict, ancestor_set: set):
    """Returns all descendents with a single ascentor connected to items
    in the ancestor set
    """

    ancestor_set_ = ancestor_set.copy()

    for c in concept_dict.values():

        ancestor_set_.add(get_single_descendent(c, ancestor_set_))

    return ancestor_set_ - {None}


def get_discipline_descendent(
    concept: Dict, ancestor_set: set, discipline_set: set, thres: float = 0.25
):
    """Returns descendent with ancestors in ancestor set & lots of ancestors in disciplinary set"""

    if any(c["display_name"] in ancestor_set for c in concept["ancestors"]):
        if (
            np.mean([k["display_name"] in discipline_set for k in concept["ancestors"]])
            > thres
        ):
            return concept["display_name"]


def get_all_discipline_descendents(
    concept_dict: Dict, ancestor_set: set, discipline_set: set, thres: float = 0.25
):
    """Returns all descendents from the ancestor set with dominant other ancestors
    in a discipline_set
    """

    ancestor_set_ = ancestor_set.copy()

    for c in concept_dict.values():

        ancestor_set_.add(
            get_discipline_descendent(c, ancestor_set_, discipline_set, thres)
        )

    return ancestor_set_ - {None}


def expand_concept_set(concept_dict: Dict, initial_set: set):
    """Gets all highly related terms to
    an initial concept set that were not in the concept set.
    """

    related_terms = [
        c
        for c in chain(*[get_related_concepts(concept_dict, c) for c in initial_set])
        if c not in initial_set
    ]

    return Counter(related_terms).most_common()


def tuple_list_to_df(tuple_list: List, population: str):
    """Turn a tuple counter into a df.
    Population is a label for the frequency column"""

    return pd.DataFrame(
        {
            "concept_name": [t[0] for t in tuple_list],
            f"frequency_{population}": [t[1] for t in tuple_list],
        }
    )


def segment_concepts(concept_dict: Dict, concept_set: set):
    """Segments concepts by level"""

    return {
        level: set(
            concept_dict[c]["display_name"]
            for c in concept_set
            if concept_dict[c]["level"] == level
        )
        for level in range(5)
    }


def make_coocc_from_tuple(tuple_list) -> pd.DataFrame:
    """Creates a co-occurrence df from concept co-occurrences"""

    return pd.DataFrame(
        {
            "ai_technique": [t[0][0] for t in tuple_list],
            "application": [t[0][1] for t in tuple_list],
            "frequency": [t[1] for t in tuple_list],
        }
    )


def make_lq(table: pd.DataFrame) -> pd.DataFrame:
    """Calculate a table's Location quotients"""

    table_shares = table.apply(lambda x: x / x.sum())

    total_shares = table.sum(axis=1) / table.sum().sum()

    return table_shares.apply(lambda x: x / total_shares)


def get_concept_clusters(concepts_table, pca_n=100, k_n=10):
    """Extracts clusters from the pipeline"""

    concepts_wide = concepts_table.pivot_table(
        index="doc_id", columns="display_name", aggfunc="size", fill_value=0
    )

    # Create an int to concept lookup
    concept_name_lookup = {n: concept for n, concept in enumerate(concepts_wide)}

    # Cluster
    cluster_pipe = make_pipeline(PCA(n_components=pca_n), TSNE())

    km_fit = pipe(
        cluster_pipe.fit_transform(concepts_wide.T.to_numpy()),
        lambda proj: KMeans(n_clusters=k_n).fit(proj),
    )

    return concept_name_lookup, km_fit


def make_concept_coocurrences(concept_df, ai_vocabulary, applications):
    """Calculates concept cooccurrences between ai_techniques and applications"""

    co_occ = concept_df.groupby("doc_id")["display_name"].apply(
        lambda x: product(
            [c for c in set(x) if c in ai_vocabulary],
            [c for c in set(x) if c in applications],
        )
    )

    co_occ_list = list(chain(*[list(co) for co in co_occ]))

    tup_df = make_coocc_from_tuple(Counter(co_occ_list).most_common())

    tup_lq = (
        pipe(
            tup_df.pivot_table(
                index="application", columns="ai_technique", values="frequency"
            ).fillna(0),
            make_lq,
        )
        .stack()
        .reset_index(name="lq")
    ).replace(0, np.nan)

    return tup_df.merge(tup_lq, on=["application", "ai_technique"]).assign(
        freq_share=lambda df: df["frequency"] / df["frequency"].sum()
    )


def plot_coocc(
    freq_table, technique_cluster_list, application_cluster_list, title="All"
):

    return (
        alt.Chart(freq_table)
        .mark_point(filled=True, shape="square", stroke="grey", strokeWidth=0.5)
        .encode(
            x=alt.X(
                "ai_technique",
                sort=[el[0] for el in technique_cluster_list][::-1],
                axis=alt.Axis(labels=False, ticks=False),
            ),
            y=alt.X(
                "application",
                sort=[el[0] for el in application_cluster_list][::-1],
                axis=alt.Axis(labels=False, ticks=False),
            ),
            tooltip=["ai_technique", "application", "lq"],
            color=alt.Color(
                "lq",
                scale=alt.Scale(type="quantile", scheme="viridis"),
                sort="descending",
                legend=None,
            ),
            size=alt.Size("freq_share", scale=alt.Scale(type="log")),
        )
        .properties(width=400, height=400, title=title)
    )


def order_clusters(cluster_coords):
    """Sorts clusters based on their similarity
    The logic is to merge the closest clusters, then the
        cluster closest to the merged cluster, and so on

    """

    coords = {n: c for n, c in enumerate(cluster_coords)}
    removed = []
    order = []

    pairs = [tuple(p) for p in combinations(coords.keys(), 2)]

    it = 0

    while len(order) < len(cluster_coords) - 1:

        similarities = []

        for p in pairs:

            similarities.append([p, 1 - cosine(coords[p[0]], coords[p[1]])])

        merged = sorted(similarities, key=lambda x: x[1], reverse=True)[0]

        coords[merged[0]] = np.mean(
            [coords[merged[0][0]], coords[merged[0][1]]], axis=0
        )

        order.append(merged[0])

        if it == 0:
            removed.append(merged[0][0])
            removed.append(merged[0][1])
        else:
            removed.append(merged[0][1])

        pairs = product(
            [merged[0]], [n for n in range(len(cluster_coords)) if n not in removed]
        )

        it += 1

    return removed
