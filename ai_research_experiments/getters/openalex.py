import json
from typing import Dict, List

import pandas as pd

from ai_research_experiments import PROJECT_DIR

OALEX_DIR = f"{PROJECT_DIR}/data/openalex"


def get_works(test: bool = False) -> pd.DataFrame:
    """Get table with AI work (paper) metadata

    Args:
        test: whether to get the top 10 rows only
    """

    return pd.read_csv(f"{OALEX_DIR}/ai_openalex_works.csv", nrows=10 if test else None)


def get_authors(test: bool = False) -> pd.DataFrame:
    """Get table with AI works - authors

    Args:
        test: whether to get the top 10 rows only
    """

    return pd.read_csv(f"{OALEX_DIR}/ai_authors.csv", nrows=10 if test else None)


def get_concepts(test: bool = False) -> pd.DataFrame:
    """Get table with AI works - concepts

    Args:
        test: whether to get the top 10 rows only
    """

    return pd.read_csv(f"{OALEX_DIR}/ai_concepts.csv", nrows=10 if test else None)


def get_mesh(test: bool = False) -> pd.DataFrame:
    """Get table with AI works - MeSH terms

    Args:
        test: whether to get the top 10 rows only
    """

    return pd.read_csv(f"{OALEX_DIR}/ai_mesh.csv", nrows=10 if test else None)


def get_institute_metadata(test: bool = False) -> pd.DataFrame:
    """Get table with institute metadata

    Args:
        test: whether to get the top 10 rows only
    """

    return pd.read_csv(
        f"{OALEX_DIR}/ai_institute_metadata.csv", nrows=10 if test else None
    )


def get_concept_taxonomy() -> List[Dict]:
    """Get Openalex concept taxonomy"""

    with open(f"{OALEX_DIR}/concepts.json", "r") as infile:
        return json.load(infile)


def get_abstracts() -> Dict[str, str]:
    """Get dict with abstracts"""

    with open(f"{OALEX_DIR}/ai_abstracts.json", "r") as infile:
        return json.load(infile)


def get_citations() -> Dict[str, str]:
    """Get dict with citations"""

    with open(f"{OALEX_DIR}/ai_citations.json", "r") as infile:
        return json.load(infile)


if __name__ == "__main__":

    print("checking that this works")

    for f in [get_works, get_authors, get_concepts, get_mesh, get_institute_metadata]:

        print(f)
        print(f(test=True).head())
        print("\n")

    print("concept taxonomy")
    print(get_concept_taxonomy()[0])

    print("\n")

    print("abstracts")
    print(list(get_abstracts().values())[0])

    print("\n")

    print("citations")
    print(list(get_citations().values())[0])
