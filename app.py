"""Streamlit front end for multi-word expression detection.

Two pages are served: interactive single-sentence tagging, and batch tagging of
an uploaded Excel workbook. Both read from the model registry in :mod:`config`,
so adding a model requires no change to this file.

.. note::
   The rendered interface here is intentionally identical to v1.0 - same
   navigation, headings, tables, colours and button labels. Only the code
   behind it changed: model loading moved to :mod:`inference` and the
   per-model branching moved to :data:`config.MODEL_REGISTRY`.
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from config import (
    INPUT_COLUMN,
    LEGACY_SECRET_KEYS,
    MODEL_CHOICES,
    MODEL_REGISTRY,
    ModelSpec,
)
from inference import LoadedModel, ModelLoadError, load_model, predict

PAGE_SENTENCE = "Sentence Prediction"
PAGE_BATCH = "Excel File Prediction"
PREVIEW_ROWS = 10

#: Row labels of the performance table. The underlying evaluation reported one
#: positive class, so the averaged rows all repeat the MWE figures.
_REPORT_ROWS = ("MWE", "Micro Avg", "Macro Avg", "Weighted Avg")
_REPORT_COLUMNS = ("Precision", "Recall", "F1-Score", "Support")

st.set_page_config(page_title="MWEs Prediction App", page_icon="🚀")

_SIDEBAR_CSS = """
        <style>
        .sidebar-title { font-size: 20px; font-weight: bold; color: #1E90FF; margin-bottom: 20px; }
        .sidebar-button { display: block; width: 100%; padding: 10px; margin: 5px 0; text-align: center; font-size: 16px; font-weight: bold; color: white; background-color: #4CAF50; border: none; border-radius: 5px; cursor: pointer; text-decoration: none; }
        .sidebar-button:hover { background-color: #45a049; }
        </style>
        """


# --------------------------------------------------------------------------- #
# Configuration lookup
# --------------------------------------------------------------------------- #


def _resolve_repo_id(spec: ModelSpec) -> str:
    """Look up the Hugging Face repository id configured for ``spec``.

    Falls back to the pre-v1.1 secret name so existing deployments keep working
    after the rename.

    Raises:
        KeyError: If neither the current nor the legacy key is configured.
    """
    if spec.secret_key in st.secrets:
        return str(st.secrets[spec.secret_key])

    legacy_key = LEGACY_SECRET_KEYS.get(spec.secret_key)
    if legacy_key and legacy_key in st.secrets:
        return str(st.secrets[legacy_key])

    raise KeyError(spec.secret_key)


def _hf_token() -> str | None:
    """Return the configured Hugging Face token, if any."""
    for key in ("HF_TOKEN", "TOKEN"):
        if key in st.secrets:
            return str(st.secrets[key])
    return None


def _load_selected_model(model_key: str) -> LoadedModel | None:
    """Load ``model_key``, showing a message instead of a traceback on failure.

    The spinner text matches v1.0 exactly.
    """
    spec = MODEL_REGISTRY[model_key]
    try:
        repo_id = _resolve_repo_id(spec)
    except KeyError:
        st.error(
            f"Missing configuration: `{spec.secret_key}` is not set. "
            "See `.streamlit/secrets.toml.example` for the expected keys."
        )
        return None

    with st.spinner(f"Loading {model_key} model..."):
        try:
            return load_model(model_key, repo_id, _hf_token())
        except ModelLoadError as exc:
            st.error(str(exc))
            return None


# --------------------------------------------------------------------------- #
# Shared rendering
# --------------------------------------------------------------------------- #


def _classification_frame(model_key: str) -> pd.DataFrame:
    """Build the classification-report table shown under the model picker.

    Reproduces the v1.0 layout exactly: metric names as columns, one row per
    averaging scheme, and object-dtype cells so integers render as ``1415``
    rather than ``1415.0``.
    """
    metrics = MODEL_REGISTRY[model_key].metrics
    values = [metrics.precision, metrics.recall, metrics.f1, metrics.support]

    report: dict[str, list[object]] = {"Metric": list(_REPORT_COLUMNS)}
    report.update({row: list(values) for row in _REPORT_ROWS})

    frame = pd.DataFrame.from_dict(report, orient="index")
    frame.columns = frame.iloc[0]
    return frame[1:]


def _render_model_section(model_key: str) -> None:
    """Render the skyblue performance heading, accuracy line and table."""
    st.markdown(
        f'<h3 style="color:skyblue;">Model Performance on the Test Set ({model_key})</h3>',
        unsafe_allow_html=True,
    )
    st.write(f"**Accuracy**: {MODEL_REGISTRY[model_key].metrics.accuracy:.4f}")
    st.table(_classification_frame(model_key))


def _render_results(table_data: list[dict[str, str]], detected_mwes: list[str]) -> None:
    """Render the token table and the detected expressions, v1.0 styling."""
    st.markdown(
        '<h3 style="color:skyblue;">Token-Level Predictions</h3>',
        unsafe_allow_html=True,
    )
    st.table(table_data)

    st.markdown('<h3 style="color:skyblue;">Detected MWEs</h3>', unsafe_allow_html=True)
    if detected_mwes:
        for mwe in detected_mwes:
            st.markdown(
                f'<span style="color:green; font-weight:bold;">-[{mwe}]</span>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No MWEs detected in the sentence.")
    st.markdown("---")


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


def render_sentence_page() -> None:
    """Page 1: interactive single-sentence prediction."""
    st.title("✍️ MWEs Detection with Different Deep Learning Models")
    st.write(
        "This app detects **Multi-Word Expressions (MWEs)** in a sentence using "
        "different fine-tuned **Deep Learning Models**."
    )

    selected_model = st.selectbox("Choose the model:", MODEL_CHOICES)
    model = _load_selected_model(selected_model)

    _render_model_section(selected_model)

    user_sentence = st.text_input("Type your sentence here:")
    if not user_sentence or model is None:
        return

    table_data, detected_mwes = predict(model, user_sentence)
    _render_results(table_data, detected_mwes)


@st.cache_data(show_spinner=False)
def _to_excel_bytes(dataframe: pd.DataFrame) -> bytes:
    """Serialise ``dataframe`` to an in-memory ``.xlsx`` file."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Sheet1")
    return output.getvalue()


def render_batch_page() -> None:
    """Page 2: batch prediction over an uploaded Excel workbook."""
    st.title("📄 MWEs Detection on Excel File")
    st.write(
        "This app detects **Multi-Word Expressions (MWEs)** in an excel file "
        "using different fine-tuned **Deep Learning Models**."
    )

    selected_model = st.selectbox("Choose the model:", MODEL_CHOICES)
    model = _load_selected_model(selected_model)

    uploaded_file = st.file_uploader("Upload your excel file:", type=["xlsx"])
    if uploaded_file is None or model is None:
        return

    try:
        df = pd.read_excel(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the workbook: {exc}")
        return

    if INPUT_COLUMN not in df.columns:
        st.error(f"The uploaded file must contain a '{INPUT_COLUMN}' column.")
        return

    sentences = df[INPUT_COLUMN].fillna("").astype(str)
    predicted_labels: list[str] = []
    detected_mwes: list[str] = []

    progress = st.progress(0.0)
    total = max(len(sentences), 1)
    for position, sentence in enumerate(sentences, start=1):
        table_data, mwes = predict(model, sentence)
        predicted_labels.append(", ".join(row["Prediction"] for row in table_data))
        detected_mwes.append(", ".join(mwes))
        progress.progress(position / total)
    progress.empty()

    df = pd.DataFrame(
        {
            INPUT_COLUMN: sentences,
            "Predicted Labels": predicted_labels,
            "Detected MWEs": detected_mwes,
        }
    )

    st.markdown('<h3 style="color:skyblue;">Updated Excel File</h3>', unsafe_allow_html=True)
    st.dataframe(df.head(PREVIEW_ROWS))

    st.download_button(
        label="📥 Download Updated Excel File",
        data=_to_excel_bytes(df),
        file_name="Updated_Dataset.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    """Render the sidebar navigation and dispatch to the selected page."""
    if "page" not in st.session_state:
        st.session_state["page"] = PAGE_SENTENCE

    with st.sidebar:
        st.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">Navigation Bar</div>', unsafe_allow_html=True)

        if st.button("✍️ Sentence Prediction"):
            st.session_state["page"] = PAGE_SENTENCE
        if st.button("📄 Excel File Prediction"):
            st.session_state["page"] = PAGE_BATCH

    if st.session_state["page"] == PAGE_SENTENCE:
        render_sentence_page()
    else:
        render_batch_page()


if __name__ == "__main__":
    main()
