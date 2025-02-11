import pydantic
import torch


class Config(pydantic.BaseModel):
    """Config to train a text classifier"""

    classifier: str | None = pydantic.Field(
        "SmolLM2-135M", description="Name of model used for training and inference."
    )
    gemini_model: str | None = pydantic.Field("gemini-2.0-flash", description="Name of Gemini model to use.")
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


class GeminiConfig(pydantic.BaseModel):
    """Config to train a text classifier"""

    gemini_model: str | None = pydantic.Field("gemini-2.0-flash", description="Name of Gemini model to use.")
    temperature: float | None = pydantic.Field(
        0, description="Controls randomness; lower values makes output more deterministic."
    )
    top_p: float = pydantic.Field(
        0.95, description="Nucleus sampling threshold; considers tokens with cumulative probability <= top_p."
    )
    top_k: int = pydantic.Field(20, description="Limits sampling to the top_k most probable tokens.")
    candidate_count: int = pydantic.Field(0, description="Number of candidate outputs to generate.")
    seed: int = pydantic.Field(5, description="Random seed for reproducibility.")
    max_output_tokens: int = pydantic.Field(100, description="Maximum number of tokens to generate in the output.")
    stop_sequences: list[str] = pydantic.Field(
        ["STOP!"], description="Sequences that will stop generation when encountered."
    )
    presence_penalty: float = pydantic.Field(
        0.0, description="Penalty for introducing new tokens; discourages repetition."
    )
    frequency_penalty: float = pydantic.Field(
        0.0, description="Penalty for repeating tokens based on frequency in the output."
    )
