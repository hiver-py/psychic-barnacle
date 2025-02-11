# psychic-barnacle

This project investigates the agreement between Google Search, proprietary LLMs, and local LLMs in identifying the movie corresponding to a given review. It uses the unsupervised split of the Hugging Face IMDb dataset. 

If theoretically possible, this repo will one day get a more suitable name.

## Set up 


```
pip install uv
uv venv
# mac os 
source .venv/bin/activate
# Windows
.venv\Scripts\activate
uv pip install -r pyproject.toml
```

To run a google search create a .env in this directory and fill out the following environment variables. Remember there is a cost associated with their custom API 
once you exceed a certain amount of requests. See [ Custom Search JSON API ](https://developers.google.com/custom-search/v1/overview) for more information.

```
API_KEY = 
SEARCH_ENGINE_ID = 
GEMINI_API_KEY = 
```

## Usage

Activate the uv environment and run:

```
python run_pipeline.py
```

## Resources

Keep up to date with the supported [Gemini models](https://ai.google.dev/gemini-api/docs/models/) and their supported functionality or run:

```python
 for m in genai.list_models():
     if 'generateContent' in m.supported_generation_methods:
         print(m)
 ```

A valid model name shoulder be specified in the config. 

Overview of [Google Geminis SDK](https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview), and the official [repository](https://github.com/googleapis/python-genai).

## Running pre-commits

How to run the pre-commits using uv:

```
uv run pre-commit install
uv run pre-commit run -a
```

## To Dos:
- Reformat saving of google search jsonl to take place after processing of search results.
- Add processing of search results.
- Add processing of Gemini model predictions
- Add support to unpack GeminiConfig together with the structured output config.
- Add support to limit costs for Gemini usage? 
- Add local LLM capabilities.