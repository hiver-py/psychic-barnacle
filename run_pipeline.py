from src.acquisition.acquire_data import load_and_split_data
from src.utils.utils import load_config_from_json
from src.preprocessing.text_cleaning import clean_text

def main():
    print("Loading config.")
    config = load_config_from_json()
    print("Loading dataset.")
    data = load_and_split_data(config)
    data = data["unsupervised"].shuffle(seed=config.seed).select(range(100))
    cleaned_dataset = [clean_text(review) for review in data["text"]]
    print(cleaned_dataset)

if __name__ == "__main__":
    main()
