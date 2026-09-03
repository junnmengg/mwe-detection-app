<div align="center">

# MWEs Detection

**Find the multi-word expressions in English text — four sequence-labelling models, one interface.**

[![Live demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fyp-mwedetection.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://docs.astral.sh/ruff/)

[**Try it live**](https://fyp-mwedetection.streamlit.app/) · [Model card](docs/MODEL_CARD.md) · [Architecture](docs/ARCHITECTURE.md) · [Contributing](CONTRIBUTING.md)

</div>

<!--
  TODO: add a demo recording here once captured. See docs/images/README.md.

  <p align="center">
    <img src="docs/images/demo.gif" alt="Selecting a model, entering a sentence, and seeing the detected expression highlighted" width="800">
  </p>
-->

---

## What this does

*Kick the bucket* has nothing to do with buckets. **Multi-word expressions** are
sequences of two or more words whose meaning is not the sum of their parts —
idioms, light-verb constructions like *take a decision*, verb-particle pairs
like *give up*. They are ordinary for a human reader and a persistent problem
for machine translation, parsing and information extraction, because the
correct reading only exists at the level of the whole phrase.

This project frames the problem as **BIO sequence labelling**: every token is
tagged `B-MWE` (begins an expression), `I-MWE` (continues one) or `O` (outside),
and adjacent `B`/`I` runs are stitched back into phrases.

```
He   decided  to   give   up    on    the   idea
O    O        O    B-MWE  I-MWE O     O     O
                   └─ give up ─┘
```

Four architectures are implemented so their behaviour can be compared directly
in the same interface, on the same input, side by side.

## Features

- **Four models, one picker** — BERT, BERT-CRF, RoBERTa-CRF and an LSTM-CRF
  baseline, each with its held-out test metrics shown next to it.
- **Interactive tagging** — type a sentence, see the per-token tags and the
  extracted expressions immediately.
- **Batch prediction** — upload an `.xlsx` file with a `Sentence` column and
  download it back with predicted labels and expressions attached.
- **Correct subword handling** — WordPiece and byte-level BPE pieces are
  reassembled into whole words before spans are extracted, so `byproduct` is
  reported as one word rather than `by ##product`.
- **No weights in the repository** — models are fetched from the Hugging Face
  Hub on first use and cached.

## Results

Held-out test set. Precision, recall and F1 are for the MWE class only.

| Model | Token accuracy | Precision | Recall | **F1 (MWE)** |
| :--- | ---: | ---: | ---: | ---: |
| BERT | 86.10% | 0.51 | 0.58 | 0.54 |
| BERT-CRF | 84.99% | 0.49 | 0.57 | 0.53 |
| **RoBERTa-CRF** | **88.71%** | **0.62** | **0.70** | **0.66** |
| LSTM-CRF | 85.77% | 0.49 | 0.48 | 0.49 |

Three things worth reading out of that table:

**Ignore the accuracy column.** `O` dominates the label distribution, so
tagging everything `O` would already score in the low eighties. F1 on the MWE
class is the real measure, and it ranges from 0.49 to 0.66 — a far wider spread
than accuracy implies.

**The CRF did not help the transformers.** BERT-CRF lands marginally *below*
plain BERT. With three tags and mostly short spans there is little sequential
structure left for a CRF to model once a transformer has attended over the
whole sentence. It earns its keep on the LSTM, which has no other way to
capture tag dependencies.

**RoBERTa's pretraining is what moves the needle.** Swapping BERT for RoBERTa
under an identical CRF decoder gains 0.13 F1 — an order of magnitude more than
adding the decoder did.

Full breakdown, limitations and caveats: [**docs/MODEL_CARD.md**](docs/MODEL_CARD.md).

> [!NOTE]
> These are research artefacts from a final-year project. Peak F1 is 0.66, so
> roughly a third of predictions are wrong. The training code and annotated
> corpus are not in this repository, so the figures above are reported rather
> than independently reproducible. Do not use these weights where a missed or
> invented expression carries real cost.

## Quick start

```bash
git clone https://github.com/junnmengg/mwe-detection-app.git
cd mwe-detection-app

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml with your Hugging Face repository ids

streamlit run app.py
```

The app opens at <http://localhost:8501>. The first prediction for each model
downloads its weights (roughly 400–500 MB for the transformers), which takes a
minute; subsequent predictions are instant.

### Configuration

All settings live in `.streamlit/secrets.toml`, which is git-ignored. Copy
`.streamlit/secrets.toml.example` and fill it in:

| Key | Required | Purpose |
| :--- | :--- | :--- |
| `BERT_REPO_ID` | yes | Hugging Face repo holding the BERT weights |
| `BERT_CRF_REPO_ID` | yes | …the BERT-CRF weights |
| `ROBERTA_CRF_REPO_ID` | yes | …the RoBERTa-CRF weights |
| `LSTM_CRF_REPO_ID` | yes | …the LSTM-CRF weights and vocabulary |
| `HF_TOKEN` | no | Read-only access token; only needed for private repos |

The pre-v1.1 names (`USERNAME1`–`USERNAME4` and `TOKEN`) are still read as a
fallback, so an existing deployment keeps working until you migrate it.

Deploying to Streamlit Community Cloud? Paste the same keys into
**App settings → Secrets** rather than committing a file.

> [!WARNING]
> Never hard-code a Hugging Face token in source or commit a real
> `secrets.toml`. If you already have, revoke the token before doing anything
> else — see [SECURITY.md](SECURITY.md).

### Batch prediction format

The uploaded workbook needs one column named exactly `Sentence`:

| Sentence |
| :--- |
| He decided to give up on the idea. |
| The committee will take a decision tomorrow. |

The download adds `Predicted Labels` (the per-token BIO sequence) and
`Detected MWEs` (the extracted phrases, comma-separated).

## How it works

```
app.py            Streamlit UI: pages, widgets, error surfaces
   │
inference.py      Weight download, caching, encode → model → decode
   │         ╲
models.py     text_processing.py     PyTorch modules │ pure tag decoding
   │         ╱
config.py         Label scheme, model registry, published metrics
```

`config.py` and `text_processing.py` import nothing beyond the standard
library. That is a deliberate constraint: it keeps the tag-decoding logic
unit-testable in milliseconds and lets CI run the full suite without
downloading PyTorch.

Adding a fifth model means adding one entry to `MODEL_REGISTRY` in `config.py`
and one branch in `inference.py` — the model picker, the metrics panel and the
loader all read from the registry. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt

ruff check .            # lint
ruff format .           # format
pytest                  # unit tests + doctests
```

CI runs all three on Python 3.10 and 3.12, plus a job that installs the full
runtime and imports every module. See [CONTRIBUTING.md](CONTRIBUTING.md).

The rendered interface is treated as frozen — refactors must not change what
appears on screen. [CONTRIBUTING.md](CONTRIBUTING.md#the-interface-is-frozen)
explains what that means in practice.

## Troubleshooting

<details>
<summary><code>ModuleNotFoundError: No module named 'torchvision'</code> in the console</summary>

Streamlit's hot-reload watcher walks every module in `sys.modules` and asks each
one for its `__path__`. Transformers v5 resolves submodules lazily, so that
question imports unrelated vision models — ZoeDepth and friends — which do need
torchvision. Streamlit catches the failure and logs it, so **the app works
normally**; the traceback is noise, not a crash.

`.streamlit/config.toml` disables the watcher, which removes the cause. If you
would rather keep hot reload, set `fileWatcherType = "auto"` there and install
torchvision into your development environment only:

```bash
pip install torchvision
```

Do not add it to `requirements.txt` — the application never imports it, and it
would be a pointless download on every deploy.

</details>

<details>
<summary>The first prediction takes a minute</summary>

Weights are downloaded from the Hugging Face Hub on first use (400–500 MB for
the transformer models) and then cached by `st.cache_resource` for the life of
the process. Later predictions with the same model are instant. Restarting the
app clears the cache.

</details>

<details>
<summary><code>Missing configuration: BERT_REPO_ID is not set</code></summary>

`.streamlit/secrets.toml` is missing or incomplete. Copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in your
Hugging Face repository ids. On Streamlit Community Cloud, put the same keys in
**App settings → Secrets** instead.

</details>

## Project structure

```text
├── app.py                      Streamlit UI (sentence + batch pages)
├── config.py                   Label scheme, model registry, metrics
├── models.py                   PyTorch architectures
├── inference.py                Weight loading, caching, prediction
├── text_processing.py          Tokenizer-agnostic tag decoding
├── utils.py                    Deprecated shim, kept for compatibility
├── tests/                      Unit tests for the pure layer
├── .streamlit/
│   ├── config.toml             Streamlit settings (committed, no secrets)
│   └── secrets.toml.example    Template for your own secrets.toml
├── docs/
│   ├── MODEL_CARD.md           Intended use, results, limitations
│   └── ARCHITECTURE.md         Design decisions and request flow
└── .github/workflows/
    ├── ci.yml                  Lint, format, test, import check
    └── keep-awake.yml          Keeps the hosted demo from sleeping
```

## Roadmap

- [ ] Publish the training scripts and evaluation notebook
- [ ] Release the annotated dataset, or document how to obtain it
- [ ] Reconcile the differing support counts across the four evaluations
- [ ] Report per-span F1 alongside per-token F1
- [ ] Add confidence scores to the UI so borderline predictions are visible
- [ ] Package a `predict()` entry point usable without Streamlit

Ideas and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

```bibtex
@software{wjm_mwe_detection,
  author = {Wong Jun Meng},
  title  = {MWE Detection: comparing transformer and LSTM sequence taggers
            for multi-word expression identification},
  year   = {2025},
  url    = {https://github.com/junnmengg/mwe-detection-app}
}
```

## Acknowledgements

Built on [Hugging Face Transformers](https://github.com/huggingface/transformers),
[pytorch-crf](https://github.com/kmkurn/pytorch-crf) and
[Streamlit](https://streamlit.io/). Originally developed as a final-year
project.

## Author

**Wong Jun Meng** — [GitHub](https://github.com/junnmengg) · [LinkedIn](https://linkedin.com/in/junnmengg)

## License

[MIT](LICENSE). Model weights inherit the licences of their base checkpoints
(`bert-base-uncased` and `roberta-base`, both Apache 2.0).
