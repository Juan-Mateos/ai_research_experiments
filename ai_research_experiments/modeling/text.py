import re
from functools import reduce
from typing import List, Optional, Dict, Any

import re

import altair as alt
import toolz.curried as t
from gensim.models import Phrases
from gensim.models.phrases import FrozenPhrases
from pandas import DataFrame
import tomotopy as tp


from string import punctuation

PUNCT = "|\\".join([x for x in punctuation])


def remove_symbols(doc: str):
    """Remove symbols from a document"""

    return re.sub("\n", " ", re.sub(PUNCT, "", doc.lower()))


def remove_stop_punct(doc):
    """Remove stop words and punctuation"""

    return [d.lower_ for d in doc if (d.is_punct is False) & (d.is_stop is False)]


def build_ngrams(
    documents: List[List[str]], n: int = 2, phrase_kws: Optional[Dict[str, Any]] = None
) -> List[List[str]]:
    """Create ngrams using Gensim's phrases.
    Args:
        documents: List of tokenised documents.
        n: The `n` in n-gram.
        phrase_kws: Passed to `gensim.models.Phrases`.
    Returns:
        List of n-grammed documents.
    """
    if n < 2:
        return documents

    def_phrase_kws = {
        "scoring": "npmi",
        "threshold": 0.25,
        "min_count": 2,
        "delimiter": "_",
    }
    phrase_kws = t.merge(def_phrase_kws, phrase_kws or {})

    def step(documents, n):
        print(f"N-gram: {n}")
        bigram = FrozenPhrases(Phrases(documents, **phrase_kws))
        return bigram[documents]

    return reduce(step, range(2, n + 1), documents)


def train_lda(docs: List[str], k: int = 50, top_remove: int = 500):
    """Train an LDA model on a list of tokenised documents"""
    mdl = tp.LDAModel(tw=tp.TermWeight.ONE, min_cf=3, rm_top=top_remove, k=k)
    for doc in docs:
        mdl.add_doc(doc)
    mdl.burn_in = 100
    mdl.train(0)
    print(
        "Num docs:",
        len(mdl.docs),
        ", Vocab size:",
        len(mdl.used_vocabs),
        ", Num words:",
        mdl.num_words,
    )
    print("Removed top words:", mdl.removed_top_words)
    print("Training...", flush=True)
    for i in range(0, 1000, 10):
        mdl.train(10)
        # print("Iteration: {}\tLog-likelihood: {}".format(i, mdl.ll_per_word))

    mdl.summary()

    for k in range(mdl.k):
        print("Topic #{}".format(k))
        for word, prob in mdl.get_topic_words(k):
            print("\t", word, prob, sep="\t")

    return mdl


def create_topic_names(mdl: tp.LDAModel, k: int, n_words=10) -> list:
    """Create a list of topic names"""

    return [
        "_".join([el[0] for n, el in enumerate(mdl.get_topic_words(n)) if n < n_words])
        for n in range(k)
    ]


def create_doc_topics(
    mdl: tp.LDAModel, n_docs: int, topic_names: list, doc_ids: list
) -> DataFrame:
    """Make a list of topic probabilities for each document"""

    return DataFrame(
        [mdl.docs[n].get_topic_dist() for n in range(n_docs)],
        columns=topic_names,
        index=doc_ids,
    )


def create_topics_years(
    doc_topics: DataFrame,
    topics_to_drop: List,
    papers_years: dict,
    weight_thres: float = 0.4,
):
    """Creates table to study topic evolution"""

    return (
        doc_topics.drop(axis=1, labels=topics_to_drop)
        .applymap(lambda weight: weight > weight_thres)
        .stack()
        .reset_index(name="has_topic")
        .assign(year=lambda df: df["paper_url"].map(papers_years))
        .rename(columns={"level_1": "topic"})
    )


def plot_topic_evolution(doc_topics: DataFrame, focus_topics: list):
    """
    Plot topic evolution for a list of topics in a dataframe of topic weights for each document.
    Args:
        doc_topics: Dataframe of topic weights for each document.
        focus_topics: List of topics to focus on.
    """

    doc_topic_year = (
        doc_topics.groupby(["year", "topic"])["has_topic"]
        .mean()
        .reset_index(drop=False)
    )

    print(doc_topic_year.head())

    return (
        alt.Chart(doc_topic_year.loc[doc_topic_year["topic"].isin(focus_topics)])
        .mark_line(point=True)
        .encode(
            x="year:O",
            y="has_topic",
            color=alt.Color("topic"),
            tooltip=["topic"],
        )
        .properties(
            title="Topic evolution",
        )
    )
