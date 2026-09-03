"""Pure text and tag post-processing helpers.

This module deliberately depends only on the standard library and
:mod:`config`. Keeping tag decoding free of PyTorch, Streamlit and Hugging Face
imports makes it fast to unit test and reusable outside the web application.

The main job here is turning a model's per-token output back into human
readable expressions, which means undoing subword tokenisation. BERT
(WordPiece) marks continuations with ``##``; RoBERTa (byte-level BPE) marks
*word starts* with ``U+0120``. Both are handled explicitly rather than guessed
at, because merging the wrong tokens silently corrupts the output.
"""

from __future__ import annotations

import string
from collections.abc import Iterable, Sequence
from enum import Enum

from config import ID_TO_MWE_LABEL, MAX_SEQUENCE_LENGTH, MIN_MWE_TOKENS, SPECIAL_TOKENS

__all__ = [
    "SubwordStyle",
    "TokenPrediction",
    "decode_predictions",
    "encode_with_vocab",
    "extract_mwes",
    "is_punctuation",
    "is_special_token",
    "strip_subword_marker",
]

#: One row of the token-level prediction table.
TokenPrediction = dict[str, str]

_PUNCTUATION = frozenset(string.punctuation)
_WORD_START_MARKER = "Ġ"  # RoBERTa byte-level BPE word boundary.
_SUBWORD_MARKER = "##"  # BERT WordPiece continuation.

_OUTSIDE_TAG = "O"
_BEGIN_TAG = "B-MWE"
_INSIDE_TAG = "I-MWE"


class SubwordStyle(str, Enum):
    """How a tokenizer signals that a token continues the previous word."""

    WORDPIECE = "wordpiece"
    """BERT-style: continuations carry a ``##`` prefix."""

    BPE = "bpe"
    """RoBERTa-style: word starts carry a ``U+0120`` prefix."""

    NONE = "none"
    """Already word-level, e.g. NLTK output for the LSTM baseline."""


def is_punctuation(token: str) -> bool:
    """Return ``True`` when every character in ``token`` is punctuation.

    Empty strings are not punctuation.

    >>> is_punctuation("...")
    True
    >>> is_punctuation("kick")
    False
    """
    return bool(token) and all(char in _PUNCTUATION for char in token)


def strip_subword_marker(token: str) -> str:
    """Remove tokenizer-specific subword markers from ``token``.

    >>> strip_subword_marker("Ġbucket")
    'bucket'
    >>> strip_subword_marker("##ing")
    'ing'
    """
    if token.startswith(_WORD_START_MARKER):
        return token[len(_WORD_START_MARKER) :]
    if token.startswith(_SUBWORD_MARKER):
        return token[len(_SUBWORD_MARKER) :]
    return token


def is_special_token(token: str) -> bool:
    """Return ``True`` for tokenizer control tokens such as ``[CLS]``."""
    return token in SPECIAL_TOKENS


def _continues_previous_word(token: str, style: SubwordStyle, is_first: bool) -> bool:
    """Return ``True`` when ``token`` is a continuation of the preceding word."""
    if is_first or style is SubwordStyle.NONE:
        return False
    if style is SubwordStyle.WORDPIECE:
        return token.startswith(_SUBWORD_MARKER)
    # Byte-level BPE: anything not marked as a word start continues the word.
    return not token.startswith(_WORD_START_MARKER)


def decode_predictions(
    tokens: Sequence[str],
    tag_ids: Iterable[int],
    style: SubwordStyle = SubwordStyle.NONE,
) -> list[TokenPrediction]:
    """Pair tokens with predicted BIO tags, reassembling whole words.

    Control tokens are dropped, subword pieces are merged back into the word
    they belong to (the first piece's tag wins, since that is the piece the
    BIO scheme anchors to), and punctuation is forced to ``O`` because the
    annotation guidelines never place punctuation inside an expression.

    Args:
        tokens: Tokens in the order the model saw them.
        tag_ids: Predicted label indices, aligned with ``tokens``.
        style: How ``tokens`` encode word boundaries.

    Returns:
        One ``{"Token": ..., "Prediction": ...}`` mapping per whole word.

    >>> rows = decode_predictions(
    ...     ["[CLS]", "kick", "the", "buck", "##et"], [2, 0, 1, 1, 1],
    ...     SubwordStyle.WORDPIECE,
    ... )
    >>> [(row["Token"], row["Prediction"]) for row in rows]
    [('kick', 'B-MWE'), ('the', 'I-MWE'), ('bucket', 'I-MWE')]
    """
    predictions: list[TokenPrediction] = []
    seen_first_word = False

    # strict=False: a model may emit padding tags beyond the visible tokens,
    # and truncated input may yield fewer. Both are handled by stopping early.
    for token, tag_id in zip(tokens, tag_ids, strict=False):
        if is_special_token(token):
            continue

        continuation = _continues_previous_word(token, style, not seen_first_word)
        piece = strip_subword_marker(token)
        if not piece:
            continue

        if continuation and predictions:
            predictions[-1]["Token"] += piece
            continue

        seen_first_word = True
        tag = ID_TO_MWE_LABEL.get(tag_id, _OUTSIDE_TAG)
        predictions.append({"Token": piece, "Prediction": tag})

    # Punctuation can only be judged once whole words are assembled.
    for prediction in predictions:
        if is_punctuation(prediction["Token"]):
            prediction["Prediction"] = _OUTSIDE_TAG

    return predictions


def extract_mwes(
    predictions: Sequence[TokenPrediction],
    min_tokens: int = MIN_MWE_TOKENS,
) -> list[str]:
    """Group consecutive ``B-MWE``/``I-MWE`` tokens into expression strings.

    A ``B-MWE`` always opens a new span, so two adjacent expressions stay
    separate instead of merging. Spans shorter than ``min_tokens`` are dropped:
    a multi-word expression is by definition more than one word.

    >>> extract_mwes([
    ...     {"Token": "kick", "Prediction": "B-MWE"},
    ...     {"Token": "the", "Prediction": "I-MWE"},
    ...     {"Token": "bucket", "Prediction": "I-MWE"},
    ...     {"Token": "today", "Prediction": "O"},
    ... ])
    ['kick the bucket']
    """
    expressions: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if len(current) >= min_tokens:
            expressions.append(" ".join(current))
        current.clear()

    for prediction in predictions:
        tag = prediction["Prediction"]
        if tag == _BEGIN_TAG:
            flush()
            current.append(prediction["Token"])
        elif tag == _INSIDE_TAG:
            current.append(prediction["Token"])
        else:
            flush()

    flush()
    return expressions


def encode_with_vocab(
    tokens: Sequence[str],
    vocab: dict[str, int],
    pad_idx: int,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> list[int]:
    """Map ``tokens`` to vocabulary ids, padded or truncated to ``max_length``.

    Unknown words fall back to the ``<UNK>`` entry, or to index 0 when the
    vocabulary does not define one.

    >>> encode_with_vocab(["a", "zzz"], {"a": 5, "<UNK>": 1}, pad_idx=0, max_length=4)
    [5, 1, 0, 0]
    """
    unknown_idx = vocab.get("<UNK>", 0)
    encoded = [vocab.get(token, unknown_idx) for token in tokens[:max_length]]
    encoded.extend([pad_idx] * (max_length - len(encoded)))
    return encoded
