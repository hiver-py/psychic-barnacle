import json
import logging

import transformers

from .config import Config

logger = logging.getLogger(__name__)


def load_config_from_json(file_path: str = "config.json") -> Config:
    """Loads and returns the config. If config.json exists, this is used to override the base config
        otherwise just uses the default settings specified in the config.py config.
    Args
        file_path (str): A filepath to a custom config.

    Returns
        config (Config): The loaded config.
    """
    try:
        with open(file_path, "r") as f:
            config_kwargs = json.load(f)
        return Config(**config_kwargs)
    except Exception as e:
        logger.error(f"No config provided: {e}. Falling back to base config.")
        return Config()


def load_model(config: Config) -> transformers.AutoModel:
    """
    Loads a transformer model based on the provided configuration.

    If a pretrained model is specified in the configuration, this function loads the
    pretrained model for sequence classification. Otherwise, it initializes a new
    model with the specified classifier and number of labels. If the model's
    `pad_token_id` is not set, it will be assigned the value of `eos_token_id`
    from the model's configuration.

    Args:
        config (Config): A configuration object containing the following attributes:
        pretrained_model (bool): Whether to load a pretrained model or not.
        classifier (str): The name or path of the model architecture or
              pretrained model to use.
        num_labels (int): The number of labels for sequence classification.

    Returns:
        transformers.AutoModel: The loaded or initialized transformer model.
    """
    if config.pretrained_model:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            config.classifier, num_labels=config.num_labels
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id
        return model
    else:
        model = transformers.AutoModel(config.classifier, num_labels=config.num_labels)
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id
        return model
