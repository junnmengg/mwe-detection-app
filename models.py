"""PyTorch model definitions for multi-word expression tagging.

Four architectures are provided, all producing one BIO tag per token:

* :class:`BertTokenClassifier` - BERT encoder + linear head.
* :class:`BertCRFTagger` - BERT encoder + CRF decoder.
* :class:`RobertaCRFTagger` - RoBERTa encoder + CRF decoder.
* :class:`LSTMCRFTagger` - BiLSTM encoder + CRF decoder (non-transformer baseline).

Each ``forward`` returns the training loss when gold labels are supplied and
predictions otherwise, so the same module serves both training and inference.

.. note::
   Submodule attribute names (``bert``, ``roberta``, ``crf``, ...) form the
   prefixes of the keys in every published checkpoint. Renaming one silently
   breaks ``load_state_dict``, so treat them as part of the public API.
"""

from __future__ import annotations

from torch import Tensor, nn
from torchcrf import CRF
from transformers import AutoModel

# Names ending in "Model" are backwards-compatible aliases for the
# pre-refactor class names; the "Tagger"/"Classifier" names are preferred.
__all__ = [
    "BertCRFModel",
    "BertCRFTagger",
    "BertModel",
    "BertTokenClassifier",
    "LSTMCRFModel",
    "LSTMCRFTagger",
    "RoBertaCRFModel",
    "RobertaCRFTagger",
]

_IGNORE_INDEX = -100
_DROPOUT = 0.1


class BertTokenClassifier(nn.Module):
    """BERT encoder with an independent softmax classifier per token.

    Args:
        model_name: Hugging Face checkpoint used to initialise the encoder.
        num_labels_mwe: Size of the tag set.
    """

    def __init__(self, model_name: str, num_labels_mwe: int) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(_DROPOUT)
        self.classifier_mwe = nn.Linear(self.bert.config.hidden_size, num_labels_mwe)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels_mwe: Tensor | None = None,
    ) -> Tensor:
        """Return cross-entropy loss when ``labels_mwe`` is given, else logits."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits_mwe = self.classifier_mwe(sequence_output)

        if labels_mwe is None:
            return logits_mwe

        loss_fn = nn.CrossEntropyLoss(ignore_index=_IGNORE_INDEX)
        return loss_fn(logits_mwe.view(-1, logits_mwe.shape[-1]), labels_mwe.view(-1))


def _crf_forward(
    crf: CRF,
    logits: Tensor,
    attention_mask: Tensor,
    labels_mwe: Tensor | None,
) -> Tensor | list[list[int]]:
    """Decode with the CRF, or return its negative log-likelihood when training.

    The CRF models transitions between adjacent tags, which rules out
    structurally invalid sequences such as an ``I-MWE`` that does not follow a
    ``B-MWE``. Positions labelled with the ignore index are remapped to tag 0
    because the CRF has no notion of an ignored position; the mask keeps them
    out of the likelihood.
    """
    mask = attention_mask.bool()
    if labels_mwe is None:
        return crf.decode(logits, mask=mask)

    labels_mwe = labels_mwe.clone()
    labels_mwe[labels_mwe == _IGNORE_INDEX] = 0
    return -crf(logits, labels_mwe, mask=mask)


class BertCRFTagger(nn.Module):
    """BERT encoder with a Conditional Random Field decoder.

    Args:
        model_name: Hugging Face checkpoint used to initialise the encoder.
        num_labels_mwe: Size of the tag set.
    """

    def __init__(self, model_name: str, num_labels_mwe: int) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(_DROPOUT)
        self.classifier_mwe = nn.Linear(self.bert.config.hidden_size, num_labels_mwe)
        self.crf = CRF(num_labels_mwe, batch_first=True)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels_mwe: Tensor | None = None,
    ) -> Tensor | list[list[int]]:
        """Return negative log-likelihood when training, else decoded tag paths."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits_mwe = self.classifier_mwe(self.dropout(outputs.last_hidden_state))
        return _crf_forward(self.crf, logits_mwe, attention_mask, labels_mwe)


class RobertaCRFTagger(nn.Module):
    """RoBERTa encoder with a Conditional Random Field decoder.

    Args:
        model_name: Hugging Face checkpoint used to initialise the encoder.
        num_labels_mwe: Size of the tag set.
    """

    def __init__(self, model_name: str, num_labels_mwe: int) -> None:
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(_DROPOUT)
        self.classifier_mwe = nn.Linear(self.roberta.config.hidden_size, num_labels_mwe)
        self.crf = CRF(num_labels_mwe, batch_first=True)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels_mwe: Tensor | None = None,
    ) -> Tensor | list[list[int]]:
        """Return negative log-likelihood when training, else decoded tag paths."""
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        logits_mwe = self.classifier_mwe(self.dropout(outputs.last_hidden_state))
        return _crf_forward(self.crf, logits_mwe, attention_mask, labels_mwe)


class LSTMCRFTagger(nn.Module):
    """Bidirectional LSTM tagger with a CRF decoder.

    Args:
        vocab_size: Number of entries in the word-level vocabulary.
        embedding_dim: Dimensionality of the randomly initialised embeddings.
        hidden_dim: Hidden size of each LSTM direction.
        num_labels: Size of the tag set.
        pad_idx: Vocabulary index used for padding, masked out of the loss.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_labels: int,
        pad_idx: int,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels_mwe: Tensor | None = None,
    ) -> Tensor | list[list[int]]:
        """Return negative log-likelihood when training, else decoded tag paths.

        ``attention_mask`` is accepted for interface parity with the
        transformer taggers; padding is derived from ``pad_idx`` instead.
        """
        embeddings = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeddings)
        logits = self.fc(lstm_out)

        if labels_mwe is None:
            return self.crf.decode(logits)

        # Previously ``pad_idx`` was read from module scope here, which raised
        # NameError on the training path.
        mask = input_ids != self.pad_idx
        return -self.crf(logits, labels_mwe, mask=mask)


BertModel = BertTokenClassifier
BertCRFModel = BertCRFTagger
RoBertaCRFModel = RobertaCRFTagger
LSTMCRFModel = LSTMCRFTagger
