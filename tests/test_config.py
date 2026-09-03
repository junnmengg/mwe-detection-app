"""Tests guarding the model registry against accidental corruption."""

from __future__ import annotations

import pytest

from config import (
    ID_TO_MWE_LABEL,
    LEGACY_SECRET_KEYS,
    MODEL_CHOICES,
    MODEL_REGISTRY,
    MWE_LABEL_TO_ID,
    NUM_LABELS,
)


def test_label_maps_are_inverses() -> None:
    assert {v: k for k, v in MWE_LABEL_TO_ID.items()} == dict(ID_TO_MWE_LABEL)


def test_label_ids_are_contiguous_from_zero() -> None:
    assert sorted(MWE_LABEL_TO_ID.values()) == list(range(NUM_LABELS))


def test_every_choice_resolves_to_a_spec() -> None:
    assert set(MODEL_CHOICES) == set(MODEL_REGISTRY)


@pytest.mark.parametrize("key", list(MODEL_REGISTRY))
def test_spec_key_matches_registry_key(key: str) -> None:
    assert MODEL_REGISTRY[key].key == key


@pytest.mark.parametrize("key", list(MODEL_REGISTRY))
def test_metrics_are_in_range(key: str) -> None:
    metrics = MODEL_REGISTRY[key].metrics
    for name in ("accuracy", "precision", "recall", "f1"):
        value = getattr(metrics, name)
        assert 0.0 <= value <= 1.0, f"{key}.{name} out of range: {value}"
    assert metrics.support > 0


def test_weights_filenames_are_unique() -> None:
    filenames = [spec.weights_filename for spec in MODEL_REGISTRY.values()]
    assert len(filenames) == len(set(filenames))


def test_legacy_secret_keys_map_to_current_keys() -> None:
    current = {spec.secret_key for spec in MODEL_REGISTRY.values()}
    assert set(LEGACY_SECRET_KEYS) <= current


def test_only_the_lstm_baseline_ships_a_vocabulary() -> None:
    with_vocab = {k for k, s in MODEL_REGISTRY.items() if s.vocab_filename}
    assert with_vocab == {"LSTM-CRF"}
