from itertools import chain
from collections import Counter
from toolz import pipe
from functools import partial

import altair as alt
import ai_research_experiments.getters.openalex as oalex
import ai_research_experiments.empirical.openalex_utils as oal_ut

from ai_research_experiments.utils import altair_save_utils as alt_save

webdriver = alt_save.google_chrome_driver_setup()


alt.data_transformers.disable_max_rows()


if __name__ == "__main__":

    print("Start concept search")

    concept_dict = pipe(
        oalex.get_concept_taxonomy(),
        lambda _list: {c["display_name"]: c for c in _list},
    )

    ai = oal_ut.get_concept(concept_dict, "Artificial intelligence")

    cs_seed = {"Computer science"}

    cs_concepts = oal_ut.get_all_single_descendents(concept_dict, cs_seed)

    ai_seed = {
        "Artificial intelligence",
        "Machine learning",
        "Natural language processing",
    }

    ai_concepts = oal_ut.get_all_discipline_descendents(
        concept_dict, ai_seed, cs_concepts, 0.9
    )

    print("Expanding AI concepts")
    expanded_terms = pipe(
        oal_ut.expand_concept_set(concept_dict, ai_concepts),
        partial(oal_ut.tuple_list_to_df, population="ai"),
    )

    # This generates a concept relatedness frequency for all
    # concepts
    expanded_terms_all = pipe(
        Counter(
            chain(
                *[
                    oal_ut.get_related_concepts(concept_dict, c)
                    for c in concept_dict.keys()
                ]
            )
        ).most_common(),
        partial(oal_ut.tuple_list_to_df, population="all"),
    )

    # We compare the term frequency for AI-related and all concepts.
    # We look for terms that are overrepresented for AI

    term_comparison = (
        expanded_terms.merge(expanded_terms_all, on=["concept_name"])
        .fillna(0)
        .assign(freq_comp=lambda df: df["frequency_ai"] / df["frequency_all"])
        .sort_values("freq_comp", ascending=False)
    ).reset_index(drop=True)

    print(term_comparison.query("freq_comp>0.2").tail(n=20))

    ai_vocabulary_expanded = ai_concepts.union(
        set(term_comparison.query("freq_comp>0.1")["concept_name"])
    )

    # This generates a list of terms at each level of the Oalex taxonomy
    ai_vocabulary_expanded_level = oal_ut.segment_concepts(
        concept_dict, ai_vocabulary_expanded
    )

    # Some terms to remove. This should probably be in a config file
    remove_terms = ["graphics", "virtual", "reality", "rendering", "3d"]

    ai_vocabulary_technique = {
        t
        for t in ai_vocabulary_expanded_level[2]
        if not any(bad in t.lower() for bad in remove_terms)
    }

    # Data analysis
    print("cluster techniques and applications")
    concepts_works = oalex.get_concepts()

    # Look at concept co-occurrence in AO
    print("get concept co-occurrence")
    # AI concepts
    concepts_ai_works = concepts_works.loc[
        concepts_works["display_name"].isin(ai_vocabulary_technique)
    ].reset_index(drop=True)

    # Extract clusters
    concept_ai_name_lookup, km_fit = oal_ut.get_concept_clusters(concepts_ai_works)

    # Cluster interpretations
    cluster_ai_interp = sorted(
        [(concept_ai_name_lookup[n], cl) for n, cl in enumerate(km_fit.labels_)],
        key=lambda x: x[1],
        reverse=False,
    )

    # Analysis for other disciplines

    # Extract all level 1 not in AI and cluster them

    all_concepts_segmented = oal_ut.segment_concepts(concept_dict, concept_dict.keys())

    segmented_level_2 = all_concepts_segmented[1] - ai_vocabulary_expanded_level[1]

    concepts_non_ai = concepts_works.loc[
        concepts_works["display_name"].isin(segmented_level_2)
    ].reset_index(drop=True)

    concept_no_ai_name_lookup, km_no_ai_fit = oal_ut.get_concept_clusters(
        concepts_non_ai
    )

    cluster_no_ai_interp = sorted(
        [
            (concept_no_ai_name_lookup[n], cl)
            for n, cl in enumerate(km_no_ai_fit.labels_)
        ],
        key=lambda x: x[1],
        reverse=False,
    )

    # Map clusters
    concept_coocc_all = oal_ut.make_concept_coocurrences(
        concept_df=concepts_works,
        ai_vocabulary=ai_vocabulary_technique,
        applications=segmented_level_2,
    )

    ordered_ai_clusters = oal_ut.order_clusters(km_fit.cluster_centers_)
    ordered_no_ai_clusters = oal_ut.order_clusters(km_no_ai_fit.cluster_centers_)

    # Sort clusters
    concepts_ai_sorted, concepts_no_ai_sorted = [
        list(
            chain(
                *[
                    [el for el in cl_unordered if el[1] == c]
                    for c in oal_ut.order_clusters(k.cluster_centers_)
                ]
            )
        )
        for k, cl_unordered in zip(
            [km_fit, km_no_ai_fit], [cluster_ai_interp, cluster_no_ai_interp]
        )
    ]

    ai_application_map = oal_ut.plot_coocc(
        concept_coocc_all,
        technique_cluster_list=concepts_ai_sorted,
        application_cluster_list=concepts_no_ai_sorted,
        title="All papers",
    )

    alt_save.save_altair(ai_application_map, "ai_application_map", driver=webdriver)

    print("Compare education and companies")
    # And now focusing on companies vs. educational institutions
    auth_works = oalex.get_authors().assign(
        inst_type=lambda df: df["inst_id"].map(
            oalex.get_institute_metadata().set_index("id")["type"].to_dict()
        )
    )

    edu_paper, comp_paper = [
        pipe(auth_works.query(f"inst_type=='{inst}'")["id"], set)
        for inst in ["education", "company"]
    ]

    edu, comp = [
        pipe(
            oal_ut.make_concept_coocurrences(
                concepts_works.loc[concepts_works["doc_id"].isin(corpus)],
                ai_vocabulary_technique,
                segmented_level_2,
            ),
            partial(
                oal_ut.plot_coocc,
                technique_cluster_list=cluster_ai_interp,
                application_cluster_list=cluster_no_ai_interp,
                title=title,
            ),
        )
        for corpus, title in zip(
            [edu_paper, comp_paper], ["Research in academia", "Research in industry"]
        )
    ]

    alt_save.save_altair(
        alt.hconcat(edu, comp).resolve_scale(size="independent", color="independent"),
        "ai_application_map_public_private",
        driver=webdriver,
    )


# # %%
# # Patent comparison

# def get_ros_dataset():
#     return pd.read_csv(
#         f"{PROJECT_DIR}/inputs/data/reliance_on_science/_pcs_mag_doi_pmid.tsv",
#         delimiter="\t",
#     )

# def get_ai_meta():
#     return pd.read_csv(f"{OA_AI_PATH}/ai_openalex_works.csv")

# # %%
# ros = get_ros_dataset()

# # %%
# ai_works = get_ai_meta()

# # %%
# ros_dois = pipe(
#     ros.query("confscore>5")["doi"].dropna().values,
#     lambda _list: ["https://doi.org/" + el for el in _list],
#     set,
# )

# # %%
# ai_works_patent_cited = ai_works.loc[ai_works["doi"].isin(ros_dois)]

# ai_works_patent_cited["publication_year"].value_counts()

# # %%
# ai_works_patent_cited_id = set(ai_works_patent_cited["work_id"])

# # %%
# patent_map = make_concept_coocurrences(
#     conc.loc[conc["doc_id"].isin(ai_works_patent_cited_id)],
#     ai_vocabulary_technique,
#     segmented_2,
# )
# plot_coocc(patent_map, title="Cited by patents")

# # %%
# patents_papers_merged = (
#     pd.merge(
#         pipe(
#             conc.query("year<2018"),
#             partial(
#                 make_concept_coocurrences,
#                 ai_vocabulary=ai_vocabulary_technique,
#                 applications=segmented_2,
#             ),
#         ).rename(columns={"freq_share": "freq_all"}),
#         patent_map.rename(columns={"freq_share": "freq_cited"}),
#         how="left",
#         on=["application", "ai_technique"],
#     )
#     .fillna(0)
#     .assign(lq_dif=lambda df: df["lq_y"] / df["lq_x"])
# )

# # %%
# patents_papers_merged[["freq_all", "freq_cited"]].corr()

# # %%
# patents_papers_merged[["lq_x", "lq_y"]].corr()

# # %%
# patents_papers_merged.head()

# # %%
# cit_patent_comp = (
#     alt.Chart(patents_papers_merged)
#     .mark_point(filled=True, stroke="black", strokeWidth=1)
#     .encode(
#         x=alt.X("freq_cited"),
#         y=alt.Y("freq_all", scale=alt.Scale(type="log")),
#         size="frequency_x",
#         color=alt.Color(
#             "lq_dif", scale=alt.Scale(type="symlog", scheme="redblue", domainMid=1)
#         ),
#         tooltip=["ai_technique", "application"],
#     )
# )

# # %%
# cit_patent_comp

# # %%
