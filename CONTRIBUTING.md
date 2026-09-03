# Contributing

Thanks for your interest in this project. Bug reports, model improvements and
documentation fixes are all welcome.

## Getting set up

```bash
git clone https://github.com/junnmengg/mwe-detection-app.git
cd mwe-detection-app

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt -r requirements-dev.txt
```

Then configure your model repositories:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml with your own Hugging Face repository ids
streamlit run app.py
```

`.streamlit/secrets.toml` is git-ignored. **Never commit real credentials** —
see [SECURITY.md](SECURITY.md) if you think you have.

## Before opening a pull request

```bash
ruff check .            # lint
ruff format .           # format
pytest                  # tests, including doctests in the pure modules
```

CI runs exactly these three commands on Python 3.10 and 3.12, plus a separate
job that installs the full runtime and imports every module.

## Project layout

| Module | Responsibility | Depends on |
| --- | --- | --- |
| `config.py` | Label scheme, model registry, published metrics | stdlib only |
| `text_processing.py` | Tokenizer-agnostic tag decoding and span extraction | stdlib + `config` |
| `models.py` | PyTorch architectures | `torch`, `transformers`, `torchcrf` |
| `inference.py` | Weight loading, caching, end-to-end prediction | all of the above |
| `app.py` | Streamlit UI | all of the above |

The layering is deliberate: `config.py` and `text_processing.py` import nothing
heavy, which is why the whole test suite runs in under a second and CI does not
need to download PyTorch.

## The interface is frozen

`app.py` renders the same interface as v1.0 — same navigation, headings,
tables, colours, spinner text and button labels. Refactors must not change what
appears on screen. If you have a genuine UI improvement, open an issue first so
it can be discussed as a deliberate change rather than a side effect.

Practical consequences:

* The sidebar stylesheet in `app.py` is kept verbatim; `E501` is disabled for
  that file rather than rewrapping the string.
* `_classification_frame` reproduces the v1.0 `DataFrame` construction,
  including its object dtype, so `st.table` renders `1415` and not `1415.0`.

## Conventions

* **Docstrings** — Google style, enforced by `ruff` (`pydocstyle`). Explain
  *why* a non-obvious decision was made, not what the line does.
* **Type hints** — required on every public function signature.
* **Tests** — new behaviour in `config.py` or `text_processing.py` needs a
  test. Behaviour that requires model weights is not unit tested; describe your
  manual verification in the pull request instead.
* **Commits** — [Conventional Commits](https://www.conventionalcommits.org/),
  e.g. `fix: merge WordPiece continuations before span extraction`.

## Adding a model

1. Publish the weights to a Hugging Face repository.
2. Add a `ModelSpec` to `MODEL_REGISTRY` in `config.py`, including its
   held-out test metrics.
3. Add a branch to `_build_module` in `inference.py` and an entry to
   `_SUBWORD_STYLES`.
4. Add the secret key to `.streamlit/secrets.toml.example`.
5. Add a row to the results table in `README.md` and a section in
   `docs/MODEL_CARD.md`.

No UI change is needed — the model picker is generated from the registry.

## Things that are explicitly out of scope

* Committing model weights or datasets to this repository.
* Anything that requires a GPU at inference time; the demo runs on free-tier
  CPU hosting.

## Code of Conduct

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
