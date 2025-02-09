import logging

from datasets import Dataset, DatasetDict, load_dataset

from src.utils.config import Config

logger = logging.getLogger(__name__)


def load_data(config: Config) -> DatasetDict | Dataset:
    """
    Load a dataset from the HuggingFace Hub.

    This function attempts to load a dataset specified in the config object. If successful,
    it also tries to set the number of labels based on the target column in the dataset.

    Args:
        config (Config): A configuration object containing dataset information.

    Returns:
        DatasetDict | Dataset: The loaded dataset.

    Raises:
        RuntimeError: If there's an error loading the dataset.
    """
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
    """
    Split a HuggingFace dataset that only has a train key into train, validation, and test sets.

    If the input dataset has only a 'train' split, this function creates a stratified split
    to produce train, validation, and test sets. If the dataset already has multiple splits,
    it returns the original dataset unchanged.

    Args:
        config: A configuration object containing split parameters.
        data (DatasetDict | Dataset): The input dataset to be split.

    Returns:
        DatasetDict: A dataset dictionary with 'train', 'validation', and 'test' splits.
    """
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
    """
    Load a dataset and split it if necessary.

    This function first loads the dataset using the load_data function, then checks if
    the loaded dataset needs to be split (i.e., if it only has a 'train' split). If splitting
    is needed, it calls the split_data function to create train, validation, and test splits.

    Args:
        config (Config): A configuration object containing dataset and split information.

    Returns:
        DatasetDict: A dataset dictionary with appropriate splits (train, validation, test).
    """
    dataset = load_data(config)
    if len(dataset.keys()) == 1:
        dataset = split_data(dataset)
    return dataset
