# psychic-barnacle

This project investigates the agreement between Google Search, proprietary LLMs, and local LLMs in identifying the movie corresponding to a given review. It uses the unsupervised split of the Hugging Face IMDb dataset.

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

## Usage

Activate the uv environment and run:

```
python run_pipeline.py
```

## Running pre-commits

How to run the pre-commits using uv:

```
uv run pre-commit install
uv run pre-commit run -a
```