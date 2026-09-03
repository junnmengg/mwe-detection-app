"""Central configuration for the MWE detection application.

Everything that describes *what* the app offers - the label scheme, the
available models, their published test metrics and where their weights live -
is declared here so that the UI (`app.py`) and the inference layer
(`inference.py`) never hard-code model details.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

# --------------------------------------------------------------------------- #
# Tagging scheme
# --------------------------------------------------------------------------- #

#: BIO tagging scheme used during fine-tuning.
#: ``B-MWE`` begins a multi-word expression, ``I-MWE`` continues one and
#: ``O`` marks a token outside any expression.
MWE_LABEL_TO_ID: Final[Mapping[str, int]] = MappingProxyType({"B-MWE": 0, "I-MWE": 1, "O": 2})

#: Reverse lookup used to turn model output indices back into tags.
ID_TO_MWE_LABEL: Final[Mapping[int, str]] = MappingProxyType(
    {index: label for label, index in MWE_LABEL_TO_ID.items()}
)

NUM_LABELS: Final[int] = len(MWE_LABEL_TO_ID)

#: Tokens emitted by the tokenizers that must never appear in the output table.
SPECIAL_TOKENS: Final[frozenset[str]] = frozenset(
    {"<s>", "</s>", "<pad>", "<unk>", "[CLS]", "[SEP]", "[PAD]", "[UNK]"}
)

#: Fixed sequence length the LSTM-CRF model was trained with.
MAX_SEQUENCE_LENGTH: Final[int] = 128

#: Minimum number of tokens a predicted span must have to count as a
#: multi-word expression. Single-token spans are model noise by definition.
MIN_MWE_TOKENS: Final[int] = 2

#: Column the batch-prediction page expects in an uploaded workbook.
INPUT_COLUMN: Final[str] = "Sentence"


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TestMetrics:
    """Held-out test-set metrics for a single model.

    Precision, recall and F1 are reported for the positive (MWE) class only;
    token accuracy is dominated by the ``O`` tag and is included for
    completeness rather than as a headline figure.
    """

    # Tells pytest not to try collecting this as a test class: the name
    # matches its default `Test*` pattern, but it is a domain type.
    __test__ = False

    accuracy: float
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Everything needed to load and describe one model variant."""

    key: str
    """Stable identifier used in the UI and in the secrets file."""

    backbone: str
    """Hugging Face checkpoint the encoder is initialised from, if any."""

    weights_filename: str
    """Name of the ``.pth`` state dict stored in the Hugging Face repo."""

    secret_key: str
    """Key in ``.streamlit/secrets.toml`` holding the Hugging Face repo id."""

    metrics: TestMetrics
    description: str
    uses_crf: bool = True
    vocab_filename: str | None = None
    """Only set for the LSTM-CRF model, which ships its own vocabulary."""


MODEL_REGISTRY: Final[Mapping[str, ModelSpec]] = MappingProxyType(
    {
        "BERT": ModelSpec(
            key="BERT",
            backbone="bert-base-uncased",
            weights_filename="bert_model_weights.pth",
            secret_key="BERT_REPO_ID",
            uses_crf=False,
            description=(
                "BERT base encoder with a linear token-classification head. "
                "Tags are predicted independently per token."
            ),
            metrics=TestMetrics(
                accuracy=0.8610, precision=0.51, recall=0.58, f1=0.54, support=1415
            ),
        ),
        "BERT-CRF": ModelSpec(
            key="BERT-CRF",
            backbone="bert-base-uncased",
            weights_filename="bert_crf_model_weights.pth",
            secret_key="BERT_CRF_REPO_ID",
            description=(
                "BERT base encoder with a Conditional Random Field decoder, "
                "which enforces valid BIO transitions across the sequence."
            ),
            metrics=TestMetrics(
                accuracy=0.8499, precision=0.49, recall=0.57, f1=0.53, support=1443
            ),
        ),
        "RoBERTa-CRF": ModelSpec(
            key="RoBERTa-CRF",
            backbone="roberta-base",
            weights_filename="roberta_crf_model_weights.pth",
            secret_key="ROBERTA_CRF_REPO_ID",
            description=(
                "RoBERTa base encoder with a CRF decoder. The strongest of the "
                "four variants on the held-out test set."
            ),
            metrics=TestMetrics(
                accuracy=0.8871, precision=0.62, recall=0.70, f1=0.66, support=1396
            ),
        ),
        "LSTM-CRF": ModelSpec(
            key="LSTM-CRF",
            backbone="",
            weights_filename="lstm_crf_model_weights.pth",
            secret_key="LSTM_CRF_REPO_ID",
            vocab_filename="lstm_crf_vocab.json",
            description=(
                "Bidirectional LSTM over randomly initialised embeddings with a "
                "CRF decoder. Included as a non-transformer baseline."
            ),
            metrics=TestMetrics(
                accuracy=0.8577, precision=0.49, recall=0.48, f1=0.49, support=1137
            ),
        ),
    }
)

#: Order in which models are offered in the UI.
MODEL_CHOICES: Final[tuple[str, ...]] = tuple(MODEL_REGISTRY)

#: Secret keys used before v1.1. Kept so that existing deployments keep working
#: after the rename; remove once every deployment has migrated.
LEGACY_SECRET_KEYS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "BERT_REPO_ID": "USERNAME1",
        "BERT_CRF_REPO_ID": "USERNAME2",
        "ROBERTA_CRF_REPO_ID": "USERNAME3",
        "LSTM_CRF_REPO_ID": "USERNAME4",
    }
)
