import json
import transformers
from .config import Config
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging


logger = logging.getLogger(__name__)

def compute_metrics(eval_pred: transformers.EvalPrediction, config: Config):
    logits, ground_truth = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(ground_truth, predictions),
        "f1-score": f1_score(ground_truth, predictions),
        "precision": precision_score(ground_truth, predictions),
        "recall": recall_score(ground_truth, predictions),
    }


def load_config_from_json(file_path: str = "config.json") -> Config:
    try:
        with open(file_path, "r") as f:
            config_kwargs = json.load(f)
        return Config(**config_kwargs)
    except Exception as e:
        logger.error(f"No config provided: {e}. Falling back to base config.")
        return Config()


def load_model(config: Config) -> transformers.AutoModel:
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