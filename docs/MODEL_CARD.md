# Model card: MWE sequence taggers

Four models are published for this project. They share a task, a tag set and an
evaluation protocol, and differ only in the encoder and decoder used.

> **Status.** These are final-year-project research artefacts, not production
> models. The training code and annotated corpus are not part of this
> repository, so the numbers below cannot currently be reproduced from source.
> Treat them as reported results, not verified ones.

## Intended use

**In scope.** Exploring how architecture choice affects multi-word expression
tagging; a demonstration of BIO sequence labelling; a teaching example.

**Out of scope.** Any setting where a missed or invented expression carries
cost — content moderation, translation pipelines, clinical or legal text
processing. Peak F1 is 0.66, so roughly a third of the model's predictions are
wrong. Do not deploy these weights against text they were not evaluated on.

## Task and tag set

Each token receives one of three tags:

| Tag | Meaning |
| --- | --- |
| `B-MWE` | First token of a multi-word expression |
| `I-MWE` | Continuation of the current expression |
| `O` | Outside any expression |

A *multi-word expression* is a sequence of two or more words whose combined
meaning is not fully predictable from the individual words — idioms
(*kick the bucket*), light-verb constructions (*take a decision*), verb-particle
constructions (*give up*) and fixed compounds.

## Architectures

| Model | Encoder | Decoder | Parameters (approx.) |
| --- | --- | --- | --- |
| BERT | `bert-base-uncased` | Linear + softmax, per token | 110M |
| BERT-CRF | `bert-base-uncased` | Linear + CRF | 110M |
| RoBERTa-CRF | `roberta-base` | Linear + CRF | 125M |
| LSTM-CRF | BiLSTM, 100-d embeddings, 256-d hidden | Linear + CRF | ~5M |

The CRF decoder scores whole tag sequences rather than tokens in isolation,
which makes structurally invalid output such as an `I-MWE` with no preceding
`B-MWE` impossible.

The LSTM baseline uses randomly initialised embeddings over an NLTK word-level
vocabulary, with sequences padded or truncated to 128 tokens.

## Results

Held-out test set. Precision, recall and F1 are for the MWE class only.

| Model | Token accuracy | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: | ---: |
| BERT | 86.10% | 0.51 | 0.58 | 0.54 | 1,415 |
| BERT-CRF | 84.99% | 0.49 | 0.57 | 0.53 | 1,443 |
| **RoBERTa-CRF** | **88.71%** | **0.62** | **0.70** | **0.66** | 1,396 |
| LSTM-CRF | 85.77% | 0.49 | 0.48 | 0.49 | 1,137 |

### How to read these numbers

**Token accuracy is not the headline.** `O` dominates the label distribution,
so a model that predicted `O` everywhere would still score in the low eighties.
F1 on the MWE class is the number that matters, and the spread there — 0.49 to
0.66 — is much wider than the accuracy column suggests.

**RoBERTa-CRF wins on every metric.** The gap over BERT-CRF (+0.13 F1) is far
larger than the gap between BERT and BERT-CRF (−0.01 F1), which points at the
pretraining corpus and the byte-level BPE tokenizer as the deciding factor
rather than the decoder.

**The CRF did not help the transformers.** BERT-CRF scores marginally *below*
plain BERT. With only three tags and mostly short spans there is little
sequential structure for the CRF to exploit, and it adds parameters that the
available training data may not support. The CRF matters much more for the
LSTM, which has no other mechanism for modelling tag dependencies.

**Support differs between rows.** The counts in the last column are not
identical across models, so the four numbers were not produced by a single
evaluation pass over one fixed test set. Until that is reconciled, treat
cross-model differences smaller than a few points as noise.

## Limitations

* **English only**, and only the register of the training corpus.
* **Single sentences.** No cross-sentence context is used; expressions split
  across a sentence boundary are missed.
* **128-token ceiling** on the LSTM baseline; longer input is truncated.
* **Input is lowercased** before tagging, discarding a capitalisation signal
  that is useful for proper-noun compounds.
* **Punctuation is forced to `O`** in post-processing, so hyphenated or
  apostrophised expressions may be split.
* **No calibration.** The app reports tags, not confidence scores, so a
  borderline prediction looks identical to a confident one.

## Ethical considerations

The models inherit whatever social biases are present in the BERT and RoBERTa
pretraining corpora and in the annotation of the training data. No bias
evaluation was carried out. Because the annotated corpus is not published, the
demographic and topical coverage of the training data cannot be independently
audited.

## Security

Weights are loaded with `torch.load`, which unpickles arbitrary Python objects.
Only load weights from repositories you trust. See [SECURITY.md](../SECURITY.md).

## Licence

Code is MIT-licensed. The published weights inherit the licences of their base
checkpoints: `bert-base-uncased` and `roberta-base` are both Apache 2.0.
