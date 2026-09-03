"""Unit tests for the pure tag post-processing layer.

These tests import no PyTorch, Streamlit or Hugging Face code, so the whole
suite runs in well under a second and needs no model weights.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from text_processing import (
    SubwordStyle,
    decode_predictions,
    encode_with_vocab,
    extract_mwes,
    is_punctuation,
    strip_subword_marker,
)

B, I, O = 0, 1, 2  # noqa: E741 - mirrors the BIO tag ids in config.MWE_LABEL_TO_ID


def tagged(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    """Build a prediction table from ``(token, tag)`` pairs."""
    return [{"Token": token, "Prediction": tag} for token, tag in pairs]


class TestIsPunctuation:
    @pytest.mark.parametrize("token", [".", "...", "!?", "--"])
    def test_punctuation_only_tokens(self, token: str) -> None:
        assert is_punctuation(token)

    @pytest.mark.parametrize("token", ["kick", "don't", "3.14", ""])
    def test_non_punctuation_tokens(self, token: str) -> None:
        assert not is_punctuation(token)


class TestStripSubwordMarker:
    def test_strips_bpe_word_start(self) -> None:
        assert strip_subword_marker("Ġbucket") == "bucket"

    def test_strips_wordpiece_continuation(self) -> None:
        assert strip_subword_marker("##ing") == "ing"

    def test_leaves_plain_tokens_untouched(self) -> None:
        assert strip_subword_marker("bucket") == "bucket"


class TestDecodePredictions:
    def test_drops_control_tokens(self) -> None:
        result = decode_predictions(["[CLS]", "rain", "[SEP]"], [O, B, O])
        assert [row["Token"] for row in result] == ["rain"]

    def test_merges_wordpiece_subwords_into_one_word(self) -> None:
        result = decode_predictions(["by", "##pro", "##duct"], [B, I, I], SubwordStyle.WORDPIECE)
        assert result == tagged(("byproduct", "B-MWE"))

    def test_merges_bpe_subwords_into_one_word(self) -> None:
        result = decode_predictions(["Ġby", "pro", "duct"], [B, I, I], SubwordStyle.BPE)
        assert result == tagged(("byproduct", "B-MWE"))

    def test_word_level_tokens_are_never_merged(self) -> None:
        result = decode_predictions(["kick", "the", "bucket"], [B, I, I])
        assert len(result) == 3

    def test_punctuation_is_forced_outside(self) -> None:
        result = decode_predictions(["hit", "!", "hard"], [B, I, I])
        assert result[1] == {"Token": "!", "Prediction": "O"}

    def test_extra_tags_are_ignored(self) -> None:
        """zip() stops at the shorter sequence rather than raising."""
        assert len(decode_predictions(["one"], [B, I, O])) == 1

    def test_unknown_tag_id_falls_back_to_outside(self) -> None:
        assert decode_predictions(["word"], [99])[0]["Prediction"] == "O"


class TestExtractMwes:
    def test_extracts_a_contiguous_span(self) -> None:
        predictions = tagged(("kick", "B-MWE"), ("the", "I-MWE"), ("bucket", "I-MWE"), ("now", "O"))
        assert extract_mwes(predictions) == ["kick the bucket"]

    def test_a_new_begin_tag_splits_adjacent_spans(self) -> None:
        predictions = tagged(
            ("hot", "B-MWE"), ("dog", "I-MWE"), ("ice", "B-MWE"), ("cream", "I-MWE")
        )
        assert extract_mwes(predictions) == ["hot dog", "ice cream"]

    def test_single_token_spans_are_discarded(self) -> None:
        assert extract_mwes(tagged(("run", "B-MWE"), ("fast", "O"))) == []

    def test_span_running_to_the_end_is_captured(self) -> None:
        assert extract_mwes(tagged(("give", "B-MWE"), ("up", "I-MWE"))) == ["give up"]

    def test_no_expressions(self) -> None:
        assert extract_mwes(tagged(("just", "O"), ("words", "O"))) == []

    def test_empty_input(self) -> None:
        assert extract_mwes([]) == []

    def test_min_tokens_is_configurable(self) -> None:
        predictions = tagged(("run", "B-MWE"), ("fast", "O"))
        assert extract_mwes(predictions, min_tokens=1) == ["run"]


class TestEncodeWithVocab:
    VOCAB: ClassVar[dict[str, int]] = {"<PAD>": 0, "<UNK>": 1, "kick": 2, "bucket": 3}

    def test_pads_short_sequences(self) -> None:
        assert encode_with_vocab(["kick"], self.VOCAB, pad_idx=0, max_length=3) == [
            2,
            0,
            0,
        ]

    def test_truncates_long_sequences(self) -> None:
        tokens = ["kick", "bucket", "kick"]
        assert encode_with_vocab(tokens, self.VOCAB, pad_idx=0, max_length=2) == [2, 3]

    def test_unknown_words_map_to_unk(self) -> None:
        assert encode_with_vocab(["zzz"], self.VOCAB, pad_idx=0, max_length=1) == [1]

    def test_missing_unk_entry_falls_back_to_zero(self) -> None:
        assert encode_with_vocab(["zzz"], {"a": 5}, pad_idx=9, max_length=1) == [0]
