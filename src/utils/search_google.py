import json
import os
from typing import List

import requests
from pydantic import BaseModel


class SearchResult(BaseModel):
    title: str
    url: str
    description: str

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "description": self.description}


def search_google(config, query: str, num_results: int, save_search_results: bool) -> List[SearchResult]:
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
    if save_search_results:
        with open(config.output_dir + "search_results.jsonl", "a") as jsonl_file:
            for item in data.get("items", []):
                search_result = SearchResult(
                    title=item.get("title", ""), url=item.get("link", ""), description=item.get("snippet", "")
                )
                # Convert the SearchResult object to a dictionary and then to a JSON string
                json_line = json.dumps(search_result.to_dict())
                # Write the JSON string to the file, followed by a newline
                jsonl_file.write(json_line + "\n")
                results_list.append(search_result)
    else:
        for item in data.get("items", []):
            results_list.append(
                SearchResult(title=item.get("title", ""), url=item.get("link", ""), description=item.get("snippet", ""))
            )
    return results_list
