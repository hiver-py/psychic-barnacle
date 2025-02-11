from dotenv import load_dotenv

from src.acquisition.acquire_data import load_and_split_data
from src.google.gemini_api import invoke_gemini
from src.google.google_search import search_google
from src.preprocessing.text_cleaning import clean_text
from src.utils.utils import load_config_from_json

load_dotenv()


def main():
    print("Loading config.")
    config = load_config_from_json()
    print("Loading dataset.")
    data = load_and_split_data(config)
    # While developing the code a subsample is sufficient.
    data = data["unsupervised"].shuffle(seed=config.seed).select(range(4, 7))
    print("Cleans text.")
    cleaned_dataset = [clean_text(review) for review in data["text"]]
    search_results = search_google(config, cleaned_dataset[0], num_results=3, save_search_results=True)
    for result in search_results:
        print(result)
    print("Calling Gemini API")
    gemini_results = invoke_gemini(cleaned_dataset)
    for result in gemini_results:
        print(result)


if __name__ == "__main__":
    main()
