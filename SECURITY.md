# Security policy

## Supported versions

Only the `main` branch and the hosted demo are supported. Fixes are not
backported to earlier tags.

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Report privately through
[GitHub Security Advisories](https://github.com/junnmengg/mwe-detection-app/security/advisories/new).
Include the affected component, how to reproduce the issue, and what an
attacker could achieve. You can expect an acknowledgement within seven days.

## Handling credentials

This project reads all credentials from `.streamlit/secrets.toml`, which is
git-ignored. Never commit a real secrets file, and never hard-code a Hugging
Face token in source.

If a token is ever committed:

1. **Revoke it first** at <https://huggingface.co/settings/tokens>. Rewriting
   history does not help — the value is already public and copies may exist in
   forks, clones and caches.
2. Issue a replacement and store it in the deployment's secret store.
3. Only then consider rewriting history with
   [`git filter-repo`](https://github.com/newren/git-filter-repo).

## Model supply chain

Weights are loaded from the Hugging Face Hub with `torch.load`, which
deserialises a Python pickle and can execute arbitrary code. Only point this
application at model repositories you control or trust.

## Handling uploaded files

The batch prediction page parses user-supplied `.xlsx` files with pandas and
openpyxl. Formulas are not evaluated and no macros are executed, but uploaded
data is processed in memory on the server; do not upload confidential material
to the public demo.
