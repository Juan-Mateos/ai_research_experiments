import json
import gzip
import requests
import os

from ai_research_experiments import PROJECT_DIR

URL = "https://production-media.paperswithcode.com/about/papers-with-abstracts.json.gz"
PATH = f"{PROJECT_DIR}/data/papers_with_code"

os.makedirs(PATH, exist_ok=True)


def fetch_file(url, path):
    """Fetch and save a URL"""

    name = url.split("/")[-1]
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(f"{path}/{name}", "wb") as f:
            f.write(response.raw.read())
    else:
        logging.info(f"Failed {url} download")


if __name__ == "__main__":
    fetch_file(URL, PATH)
