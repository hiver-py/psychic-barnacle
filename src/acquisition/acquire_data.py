import logging
from src.utils.config import Config
from datasets import load_dataset, DatasetDict, Dataset

logger = logging.getLogger(__name__)


def load_data(config: Config) -> DatasetDict | Dataset:
    try:
        dataset = load_dataset(config.dataset)
        if config.target in dataset["train"].features:
            config.num_labels = len(set(dataset["train"][config.target]))
        else:
            logger.warning("'label' column not found in dataset. Unable to set num_labels.")
        logger.info(f"Dataset loaded from HuggingFace Hub. Number of labels: {config.num_labels}")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load from HuggingFace Hub: {e}")
        raise RuntimeError(f"Error loading dataset '{config.dataset}': {str(e)}") from e


def split_data(config, data):
    """Splits huggingface datasets that only has a train key."""
    if len(data.keys()) == 1 and "train" in data.keys():
        train_test = data["train"].train_test_split(
            test_size=0.2, seed=config.seed, stratify=data["train"][config.target]
        )
        train_valid = train_test["train"].train_test_split(
            test_size=0.125, seed=config.seed, stratify=train_test["train"][config.target]
        )
        split_dataset = DatasetDict(
            {"train": train_valid["train"], "validation": train_valid["test"], "test": train_test["test"]}
        )
        return split_dataset
    else:
        return data


def load_and_split_data(config: Config):
    dataset = load_data(config)
    if len(dataset.keys()) == 1:
        dataset = split_data(dataset)
    return dataset
