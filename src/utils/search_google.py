import os
from typing import List

import requests
from pydantic import BaseModel


class SearchResult(BaseModel):
    title: str
    url: str
    description: str


def search_google(query: str, num_results: int) -> List[SearchResult]:
    API_KEY = os.getenv("API_KEY")
    SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

    if not API_KEY or not SEARCH_ENGINE_ID:
        raise ValueError(f"API KEY: {API_KEY} or SEARCH_ENGINE_ID: {SEARCH_ENGINE_ID} is not set.")

    # Construct the API URL
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": API_KEY, "cx": os.getenv("SEARCH_ENGINE_ID"), "q": query, "num": num_results}

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    results_list = []
    for item in data.get("items", []):
        results_list.append(
            SearchResult(title=item.get("title", ""), url=item.get("link", ""), description=item.get("snippet", ""))
        )

    return results_list
