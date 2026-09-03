"""Model loading and inference for multi-word expression detection.

Weights are not stored in this repository. They are pulled on demand from the
Hugging Face Hub and cached by Streamlit for the lifetime of the process, so
the first prediction for a given model is slow and every later one is not.
"""

from __future__ import annotations

import json
from typing import Any

import nltk
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from config import MODEL_REGISTRY, NUM_LABELS, ModelSpec
from models import (
    BertCRFTagger,
    BertTokenClassifier,
    LSTMCRFTagger,
    RobertaCRFTagger,
)
from text_processing import (
    SubwordStyle,
    TokenPrediction,
    decode_predictions,
    encode_with_vocab,
    extract_mwes,
)

__all__ = ["LoadedModel", "ModelLoadError", "load_model", "predict"]

_LSTM_EMBEDDING_DIM = 100
_LSTM_HIDDEN_DIM = 256
_CRF_TRANSITION_KEYS = (
    "crf.start_transitions",
    "crf.end_transitions",
    "crf.transitions",
)

_SUBWORD_STYLES: dict[str, SubwordStyle] = {
    "BERT": SubwordStyle.WORDPIECE,
    "BERT-CRF": SubwordStyle.WORDPIECE,
    "RoBERTa-CRF": SubwordStyle.BPE,
    "LSTM-CRF": SubwordStyle.NONE,
}


class ModelLoadError(RuntimeError):
    """Raised when weights or vocabulary cannot be fetched or loaded."""


class LoadedModel:
    """A ready-to-use model together with everything inference needs.

    Attributes:
        spec: Registry entry the model was built from.
        module: The evaluated :class:`torch.nn.Module`.
        tokenizer: Hugging Face tokenizer, or ``None`` for the LSTM baseline.
        vocab: Word-level vocabulary, only set for the LSTM baseline.
        pad_idx: Padding index matching ``vocab``.
    """

    __slots__ = ("module", "pad_idx", "spec", "tokenizer", "vocab")

    def __init__(
        self,
        spec: ModelSpec,
        module: torch.nn.Module,
        tokenizer: Any | None = None,
        vocab: dict[str, int] | None = None,
        pad_idx: int = 0,
    ) -> None:
        self.spec = spec
        self.module = module
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.pad_idx = pad_idx

    @property
    def subword_style(self) -> SubwordStyle:
        """How this model's tokenizer marks word boundaries."""
        return _SUBWORD_STYLES[self.spec.key]


def _ensure_nltk_punkt() -> None:
    """Download the NLTK tokenizer tables if they are missing."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def _build_module(spec: ModelSpec, vocab_size: int, pad_idx: int) -> torch.nn.Module:
    """Instantiate the architecture described by ``spec`` with random weights."""
    if spec.key == "BERT":
        return BertTokenClassifier(spec.backbone, num_labels_mwe=NUM_LABELS)
    if spec.key == "BERT-CRF":
        return BertCRFTagger(spec.backbone, num_labels_mwe=NUM_LABELS)
    if spec.key == "RoBERTa-CRF":
        return RobertaCRFTagger(spec.backbone, num_labels_mwe=NUM_LABELS)
    if spec.key == "LSTM-CRF":
        return LSTMCRFTagger(
            vocab_size=vocab_size,
            embedding_dim=_LSTM_EMBEDDING_DIM,
            hidden_dim=_LSTM_HIDDEN_DIM,
            num_labels=NUM_LABELS,
            pad_idx=pad_idx,
        )
    raise ModelLoadError(f"Unknown model key: {spec.key!r}")


def _fill_missing_crf_transitions(state_dict: dict[str, torch.Tensor], num_tags: int) -> None:
    """Add zeroed CRF transition tensors absent from an older checkpoint.

    Some early LSTM-CRF checkpoints were saved without the CRF transition
    matrices. Zero-filling them keeps loading strict elsewhere instead of
    silently accepting arbitrary missing keys.
    """
    for key in _CRF_TRANSITION_KEYS:
        if key in state_dict:
            continue
        is_pairwise = key.endswith(".transitions")
        state_dict[key] = torch.zeros(num_tags, num_tags) if is_pairwise else torch.zeros(num_tags)


@st.cache_resource(show_spinner=False)
def load_model(model_key: str, repo_id: str, token: str | None) -> LoadedModel:
    """Download and initialise one model variant.

    Args:
        model_key: Key into :data:`config.MODEL_REGISTRY`.
        repo_id: Hugging Face repository holding the weights.
        token: Hugging Face access token, or ``None`` for public repositories.

    Returns:
        A :class:`LoadedModel` in evaluation mode.

    Raises:
        ModelLoadError: If the weights or vocabulary cannot be retrieved.
    """
    try:
        spec = MODEL_REGISTRY[model_key]
    except KeyError as exc:  # pragma: no cover - guarded by the UI
        raise ModelLoadError(f"Unknown model key: {model_key!r}") from exc

    try:
        vocab: dict[str, int] | None = None
        pad_idx = 0
        if spec.vocab_filename:
            vocab_path = hf_hub_download(repo_id=repo_id, filename=spec.vocab_filename, token=token)
            with open(vocab_path, encoding="utf-8") as handle:
                vocab = json.load(handle)
            pad_idx = vocab.get("<PAD>", 0)

        module = _build_module(spec, vocab_size=len(vocab or {}), pad_idx=pad_idx)

        weights_path = hf_hub_download(repo_id=repo_id, filename=spec.weights_filename, token=token)
        state_dict = torch.load(weights_path, map_location="cpu")

        if spec.key == "LSTM-CRF":
            _fill_missing_crf_transitions(state_dict, num_tags=NUM_LABELS)
            module.load_state_dict(state_dict, strict=False)
        else:
            module.load_state_dict(state_dict)

        module.eval()
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(f"Could not load {model_key} from '{repo_id}': {exc}") from exc

    tokenizer = None
    if spec.backbone:
        tokenizer = AutoTokenizer.from_pretrained(repo_id, token=token)

    return LoadedModel(spec, module, tokenizer, vocab, pad_idx)


def _encode(
    loaded: LoadedModel, sentence: str
) -> tuple[torch.Tensor, torch.Tensor | None, list[str]]:
    """Turn ``sentence`` into model inputs and the token strings to display."""
    if loaded.spec.key == "LSTM-CRF":
        _ensure_nltk_punkt()
        from nltk.tokenize import word_tokenize

        tokens = word_tokenize(sentence.lower())
        assert loaded.vocab is not None
        ids = encode_with_vocab(tokens, loaded.vocab, loaded.pad_idx)
        return torch.tensor([ids]), None, tokens

    assert loaded.tokenizer is not None
    encoded = loaded.tokenizer(
        sentence.lower(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True,
    )
    tokens = loaded.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    return encoded["input_ids"], encoded["attention_mask"], tokens


@torch.inference_mode()
def predict(loaded: LoadedModel, sentence: str) -> tuple[list[TokenPrediction], list[str]]:
    """Tag ``sentence`` and extract the multi-word expressions it contains.

    Args:
        loaded: A model returned by :func:`load_model`.
        sentence: Raw input text.

    Returns:
        A ``(token_predictions, expressions)`` pair. Both are empty for blank
        input.
    """
    if not sentence or not sentence.strip():
        return [], []

    input_ids, attention_mask, tokens = _encode(loaded, sentence)

    if loaded.spec.uses_crf:
        tag_ids = loaded.module(input_ids=input_ids, attention_mask=attention_mask)[0]
    else:
        logits = loaded.module(input_ids=input_ids, attention_mask=attention_mask)
        tag_ids = torch.argmax(logits.squeeze(0), dim=-1).tolist()

    predictions = decode_predictions(tokens, tag_ids, loaded.subword_style)
    return predictions, extract_mwes(predictions)
