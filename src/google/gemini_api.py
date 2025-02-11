import os
from typing import List

from pydantic import BaseModel

from google import genai

from ..utils.config import GeminiConfig


class Response(BaseModel):
    movie_title: str
    explanation: str


def invoke_gemini(requests: List[str], config=GeminiConfig) -> List[dict]:
    """
    Invokes the Gemini model for each request and returns a list of Response objects.

    Args:
        requests: A list of strings, where each string is a movie review.

    Returns:
        A list of dictionaries, one for each request.  Each Response object contains
        the predicted movie title and the explanation behind the movie title prediction.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    client = genai.Client(api_key=api_key)
    list_of_responses = []
    for item in requests:
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"Predict the most likely movie title given the following review of the movie: {item}",
                config={"response_mime_type": "application/json", "response_schema": list[Response]},
            )
        except Exception as e:
            print(f"Error during Gemini API call: {e}, for review {item}")
        list_of_responses.append(
            {
                "review": item,
                "movie_title": response.parsed[0].movie_title,
                "explanation": response.parsed[0].explanation,
            }
        )
    return list_of_responses
