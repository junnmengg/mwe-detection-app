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
- The keep-awake workflow now runs every four hours instead of six, visits the
  demo in a real browser so Community Cloud registers a genuine session, clicks
  the wake button when it finds the app already asleep, and fails the run when
  it cannot confirm the app rendered. Community Cloud's sleep threshold is 12
  hours, and GitHub's scheduled runs are best effort, so the previous six-hour
  plain HTTPS request left no margin and failed silently.
- A `keepalive` job keeps the schedule from being auto-disabled after 60 days
  of repository inactivity. It is a separate job so the job installing
  third-party npm packages never holds write access. It runs even when the
  visit fails, so a broken demo cannot also silence the schedule.
- The keep-awake script polls for readiness in short slices rather than issuing
  one long `waitForSelector`. Puppeteer caps every individual call at
  `protocolTimeout` (180s by default), so a longer wait died with an opaque
  `ProtocolError` instead of a usable timeout. It also waits for
  `domcontentloaded` rather than `networkidle2`, because Streamlit holds a
  WebSocket open and the network may never go idle.
- The script probes every frame rather than only the top-level document.
  Community Cloud serves the app in a nested browsing context, so the outer
  page has the right title but an empty body and none of Streamlit's elements.
- Readiness now falls back to a page-title match when no Streamlit element is
  visible. The request still reached a running container, which is what keeps
  the app awake, so failing the run in that case was crying wolf.
- On failure the script saves a screenshot and the page HTML, which the
  workflow uploads as an artifact. Polling output is throttled to one line
  every fifteen seconds instead of one every three.
- GitHub Actions bumped to `checkout@v7`, `setup-node@v6`, `setup-python@v6`
  and `upload-artifact@v7`, clearing the Node 20 deprecation warning.

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
