import pydantic
import torch


class Config(pydantic.BaseModel):
    """Config to train a text classifier"""

    classifier: str | None = pydantic.Field(
        "SmolLM2-135M", description="Name of model used for training and inference."
    )
    tokenizer: str | None = pydantic.Field(None, description="Name of model used to tokenize data.")
    dataset: str | None = pydantic.Field("stanfordnlp/imdb", description="Name of dataset.")
    target: str | int | None = pydantic.Field(1, description="Name of target variable.")
    num_labels: int | None = pydantic.Field(
        None, description="Number of unique labels, if 'None' will be inferred from train set."
    )
    device: str = pydantic.Field(
        "cuda" if torch.cuda.is_available() else "cpu", description="Whether to use GPU or CPU."
    )
    seed: int = pydantic.Field(10, description="Seed to use for reproducibility.")
    output_dir: str = pydantic.Field("src/artifacts/", description="Directory to save the trained model and artifacts")
