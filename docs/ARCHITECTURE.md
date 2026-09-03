# Architecture

## Layering

The codebase is arranged so that the cheap, testable logic has no heavy
dependencies. Each layer may import from the layers below it and never the
reverse.

```
app.py            Streamlit UI: pages, widgets, error surfaces
   |
inference.py      Weight download, caching, encode -> model -> decode
   |         \
models.py     text_processing.py    PyTorch modules | pure tag decoding
   |         /
config.py         Label scheme, model registry, published metrics
```

`config.py` and `text_processing.py` import nothing outside the standard
library. That is what lets the test suite run in under a second and lets CI
skip a multi-gigabyte PyTorch install for the lint-and-test job.

## Request flow

A single sentence prediction:

1. `app.py` resolves the selected model's Hugging Face repository id from
   Streamlit secrets, falling back to the pre-v1.1 key names.
2. `inference.load_model` downloads the weights (and, for the LSTM baseline, a
   vocabulary file), builds the architecture from `config.MODEL_REGISTRY`, and
   loads the state dict. The result is cached by `st.cache_resource`, so this
   happens once per model per process.
3. `inference._encode` turns the sentence into tensors — a Hugging Face
   tokenizer for the transformers, NLTK plus a vocabulary lookup for the LSTM.
4. The module returns either CRF-decoded tag paths or logits that are argmaxed.
5. `text_processing.decode_predictions` drops control tokens, reassembles
   subword pieces into whole words and forces punctuation to `O`.
6. `text_processing.extract_mwes` groups consecutive `B`/`I` tags into spans
   and discards single-token ones.

## Three design decisions worth knowing about

### The model registry

Everything that varies between the four models — backbone checkpoint, weights
filename, secret key, tokenizer style, published metrics — lives in one
`MODEL_REGISTRY` mapping. The UI's model picker, the loader's dispatch and the
metrics panel all read from it, so adding a fifth model touches one data
structure and two small functions rather than every page.

### Subword reassembly

BERT and RoBERTa split rare words into pieces, and a naive implementation
reports them as separate words: `byproduct` becomes `by ##product`. The two
tokenizers signal boundaries in opposite ways — WordPiece marks *continuations*
with `##`, byte-level BPE marks *word starts* with `U+0120` — so `SubwordStyle`
records which convention a model uses and `decode_predictions` merges
accordingly. The first piece's tag wins, since that is the piece the BIO scheme
anchors to.

### A frozen interface

`app.py` was rewritten on top of the new layers, but the rendered interface is
deliberately identical to v1.0. Two places pay a small cost for that:

* The sidebar stylesheet is embedded verbatim as a single long line, so `E501`
  is disabled for `app.py` rather than rewrapping a string that is part of the
  frozen output.
* `_classification_frame` rebuilds the performance table the same way v1.0 did
  — via a dict whose first row holds the column names — because that produces
  object-dtype cells, and `st.table` therefore renders `1415` rather than
  `1415.0`. A "cleaner" `DataFrame` would change what users see.

## Checkpoint compatibility

Submodule attribute names (`bert`, `roberta`, `crf`, `classifier_mwe`, …) form
the key prefixes in every published `state_dict`. Renaming one silently breaks
`load_state_dict`. They are effectively public API and are documented as such
in `models.py`.

## Deployment

The app is hosted on Streamlit Community Cloud, which suspends an app after
**12 hours** without traffic. `.github/workflows/keep-awake.yml` visits it
every four hours so a visitor never lands on a cold start.

Three details in that workflow are deliberate, because none of the moving
parts is guaranteed on its own:

* **A real browser, not `curl`.** Streamlit's docs say an app sleeps without
  "traffic" but never define the word. A plain HTTPS request fetches the page
  shell; it does not open the WebSocket that carries a real Streamlit session.
  Puppeteer removes the ambiguity at the cost of a couple of minutes per run,
  which is free on a public repository.
* **Four hours against a twelve-hour threshold.** GitHub's `schedule` trigger
  is best effort and is routinely delayed under load. Three consecutive runs
  must be lost before the app can sleep.
* **A keepalive job.** GitHub disables scheduled workflows after 60 days
  without repository activity, with no warning. An empty commit every 50 days
  resets that clock. It runs as a separate job so the job installing
  third-party npm packages never holds write access.

The script also clicks Community Cloud's "get this app back up" button when it
finds the app already asleep, and exits non-zero if it cannot confirm the app
rendered — turning a silent stop into a failed run and a notification.

Even so, this cannot promise zero cold starts: pushing a commit redeploys the
app, and Community Cloud recycles containers for its own reasons. For a demo
that must never be cold, add a second, independent monitor so the two failure
modes are not correlated.

Weights are never committed. They are fetched from the Hugging Face Hub at
first use and cached in the container's filesystem for the life of that
container.
