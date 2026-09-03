# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0]

The rendered Streamlit interface is unchanged in this release. Navigation,
headings, tables, colours, spinner text and button labels are identical to
v1.0; only the code behind them was reorganised.

### Security

- Removed the last hard-coded Hugging Face token from the codebase and
  documented credential rotation in [SECURITY.md](SECURITY.md). **Any token
  that appeared in this repository's history must be treated as compromised
  and revoked.**
- `.gitignore` now excludes `.env` files, key material and model artefacts.

### Added

- `config.py`: a single model registry driving the UI, the loader and the
  published metrics.
- `text_processing.py`: dependency-free tag decoding and span extraction.
- A `pytest` suite covering the pure layer, plus doctests.
- CI (lint, format, tests on Python 3.10 and 3.12, and a full-runtime import
  check), Dependabot, issue and pull request templates.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, this changelog, a
  model card and an architecture note.
- `requirements-lock.txt` alongside a range-bounded `requirements.txt`.
- `.gitattributes`, so a Windows checkout no longer reports every file as
  modified when inspected on Linux.
- A progress bar on the batch page, and readable messages when a model fails
  to load instead of a raw traceback.
- `.streamlit/config.toml`, which turns off Streamlit's module watcher. The
  watcher asks every module in `sys.modules` for its `__path__`, which makes
  Transformers v5 lazily import torchvision-dependent vision models and log a
  `ModuleNotFoundError` traceback on every run. The app was unaffected, but the
  noise buried real errors.
- A troubleshooting section in the README covering that warning, first-run
  download times and missing secrets.

### Changed

- Split `utils.py` into `inference.py` and `text_processing.py`. `utils.py`
  remains as a deprecated shim.
- Secret keys renamed from `USERNAME1`–`USERNAME4` to descriptive names such as
  `ROBERTA_CRF_REPO_ID`; the old names are still read as a fallback.
- The Hugging Face token is now optional, so public model repositories need no
  credentials at all.
- The keep-awake workflow makes a plain HTTPS request instead of launching a
  headless browser, cutting each run from minutes to seconds.

### Fixed

- Subword pieces are now reassembled into whole words before spans are
  extracted, so BERT no longer reports `by ##product` and RoBERTa no longer
  splits words at byte-BPE boundaries.
- Single-token spans are no longer reported as multi-word expressions.
- A `B-MWE` immediately following another span now starts a new expression
  instead of being merged into it.
- `LSTMCRFTagger.forward` read `pad_idx` from module scope, raising
  `NameError` whenever labels were supplied.
- The duplicated model-loading block on the two pages has been removed.

## [1.0.0]

- Initial release: Streamlit app with BERT, BERT-CRF, RoBERTa-CRF and LSTM-CRF,
  single-sentence and Excel batch prediction.

[Unreleased]: https://github.com/junnmengg/mwe-detection-app/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/junnmengg/mwe-detection-app/releases/tag/v1.1.0
[1.0.0]: https://github.com/junnmengg/mwe-detection-app/releases/tag/v1.0.0
