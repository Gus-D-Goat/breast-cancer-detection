import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from inference import (
    CLASS_NAMES,
    LOW_CONFIDENCE_THRESHOLD,
    load_inference_model,
    predict_array,
    top_prediction,
)

BASE_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = BASE_DIR / "Test_images"
IMAGE_DIR = BASE_DIR / "image"
EXAMPLES = sorted(EXAMPLE_DIR.glob("*.jpg"))

st.set_page_config(
    page_title="OncologyBreastScanner",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
        .hero {
            background: linear-gradient(135deg, #fff7ed 0%, #f8fafc 55%, #eef2ff 100%);
            border: 1px solid #e2e8f0;
            border-radius: 22px;
            padding: 2rem 2.2rem;
            margin-bottom: 1.5rem;
        }
        .hero h1 { margin: 0 0 .4rem 0; font-size: 2.35rem; line-height: 1.15; color: #0f172a; }
        .hero p { margin: 0; color: #475569; font-size: 1.03rem; max-width: 820px; }
        .pill {
            display: inline-block;
            margin-top: 1rem;
            padding: .35rem .75rem;
            border-radius: 999px;
            background: #ffedd5;
            color: #9a3412;
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .02em;
        }
        .section-title { font-size: 1.05rem; font-weight: 700; color: #0f172a; margin: .2rem 0 .75rem 0; }
        .note {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: .85rem 1rem;
            color: #475569;
            font-size: .9rem;
        }
        .footer { color: #64748b; font-size: .82rem; text-align: center; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _setting(name):
    value = os.environ.get(name)
    if value:
        return value
    if not st.secrets.load_if_toml_exists():
        return None
    return st.secrets.get(name)


@st.cache_resource(show_spinner="Loading the model...")
def get_model(weights_url):
    return load_inference_model(weights_url=weights_url)


def _clear_result():
    st.session_state.scores = None


def _set_image(image):
    st.session_state.image = image
    st.session_state.uploader_key += 1
    st.session_state.upload_token = None
    _clear_result()


def _select_example(path):
    _set_image(Image.open(path).convert("RGB"))


def _clear_all():
    _set_image(None)


for key, value in {
    "image": None,
    "scores": None,
    "uploader_key": 0,
    "upload_token": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.markdown(
    """
    <div class="hero">
        <h1>OncologyBreastScanner</h1>
        <p>
            Upload a mammogram to estimate breast tissue density and benign or malignant
            classification across eight categories. Results are intended for research and
            educational use only.
        </p>
        <span class="pill">DenseNet201 · 8-class classifier</span>
    </div>
    """,
    unsafe_allow_html=True,
)

weights_url = _setting("MODEL_WEIGHTS_URL")
try:
    model = get_model(weights_url)
except Exception as exc:
    st.error(
        "The model weights could not be loaded. Check the deployment logs, the Google "
        "Drive quota, or set MODEL_WEIGHTS_URL to a direct download URL."
    )
    with st.expander("Technical details"):
        st.exception(exc)
    st.stop()

tab_analyze, tab_how, tab_guide = st.tabs(
    ["Analyze mammogram", "How it works", "Visual guide"]
)

with tab_analyze:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(
            '<div class="section-title">1. Select a mammogram</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload a PNG, JPG, BMP, or TIFF image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            key=f"upload_{st.session_state.uploader_key}",
            label_visibility="collapsed",
        )

        if uploaded is not None:
            token = getattr(uploaded, "file_id", None) or (uploaded.name, uploaded.size)
            if st.session_state.upload_token != token:
                try:
                    st.session_state.image = Image.open(uploaded).convert("RGB")
                except Exception:
                    st.error("This image could not be read. Try a different file.")
                else:
                    st.session_state.upload_token = token
                    _clear_result()

        if st.session_state.image is not None:
            st.image(
                st.session_state.image,
                caption="Selected mammogram",
                use_container_width=True,
            )
        else:
            st.info("No image selected yet. Upload a file or choose an example below.")

        st.markdown(
            '<div class="section-title" style="margin-top:1.2rem;">2. Run the analysis</div>',
            unsafe_allow_html=True,
        )
        analyze_col, clear_col = st.columns([2, 1])
        analyze_clicked = analyze_col.button(
            "Analyze mammogram",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.image is None,
        )
        clear_col.button(
            "Clear",
            use_container_width=True,
            on_click=_clear_all,
            disabled=st.session_state.image is None and st.session_state.scores is None,
        )

        if analyze_clicked:
            with st.spinner("Applying preprocessing and running the model..."):
                scores = predict_array(model, np.array(st.session_state.image))
            if scores is None:
                st.error("The image could not be preprocessed. Try another mammogram.")
            else:
                st.session_state.scores = scores

        if EXAMPLES:
            st.markdown(
                '<div class="section-title" style="margin-top:1.4rem;">Example cases</div>',
                unsafe_allow_html=True,
            )
            example_cols = st.columns(len(EXAMPLES))
            for column, path in zip(example_cols, EXAMPLES):
                with column:
                    st.image(str(path), use_container_width=True)
                    st.button(
                        path.stem,
                        key=f"example_{path.stem}",
                        use_container_width=True,
                        on_click=_select_example,
                        args=(path,),
                    )

    with right:
        st.markdown(
            '<div class="section-title">Analysis result</div>',
            unsafe_allow_html=True,
        )
        scores = st.session_state.scores
        if scores is None:
            st.info("The prediction and class probabilities will appear here.")
        else:
            label, confidence, low_confidence = top_prediction(scores)
            if low_confidence:
                st.warning(
                    f"{label}. The highest class score is only {confidence:.1%}, so the "
                    "model is not confident enough to report a reliable category."
                )
            else:
                st.metric(
                    label="Most likely classification",
                    value=label,
                    delta=f"{confidence:.1%} confidence",
                    delta_color="off",
                )

            st.markdown(
                '<div class="section-title" style="margin-top:1rem;">Class probabilities</div>',
                unsafe_allow_html=True,
            )
            for name, probability in sorted(
                scores.items(), key=lambda item: item[1], reverse=True
            ):
                value = min(max(probability, 0.0), 1.0)
                st.progress(value, text=f"{name} — {probability:.1%}")

            st.caption(
                f"Scores below {LOW_CONFIDENCE_THRESHOLD:.0%} are treated as low confidence."
            )

        st.markdown(
            '<div class="note" style="margin-top:1rem;">'
            "Images are processed in memory for this request. The app does not save "
            "uploads or prediction history."
            "</div>",
            unsafe_allow_html=True,
        )

with tab_how:
    st.markdown("### From mammogram to prediction")
    st.markdown(
        """
        1. **Upload** a mammogram image through the analysis tab.
        2. **Preprocess** the image: convert it to RGB, resize to 224 × 224, apply a
           sharpening filter, and normalize pixel values to 0–1.
        3. **Classify** the image with a DenseNet201 network fine-tuned on eight
           benign/malignant and density categories.
        4. **Review** the predicted class and the probability for every category.
        """
    )

    col_a, col_b = st.columns(2)
    with col_a:
        work = IMAGE_DIR / "after cropping.png"
        if work.exists():
            st.image(str(work), caption="Crop and enhancement workflow", use_container_width=True)
    with col_b:
        processing = IMAGE_DIR / "processing.png"
        if processing.exists():
            st.image(str(processing), caption="Sharpening and contrast processing", use_container_width=True)

    st.markdown("### Classification categories")
    st.markdown(
        "\n".join(f"- **{name}**" for name in CLASS_NAMES)
    )

    st.markdown("### Density scale")
    st.markdown(
        """
        - **Density 1** — almost entirely fatty
        - **Density 2** — scattered areas of fibroglandular density
        - **Density 3** — heterogeneously dense
        - **Density 4** — extremely dense
        """
    )

with tab_guide:
    st.markdown("### Tissue examples")
    benign_col, malignant_col = st.columns(2)
    benign = IMAGE_DIR / "benign.png"
    malignant = IMAGE_DIR / "malignant.png"
    with benign_col:
        if benign.exists():
            st.image(str(benign), caption="Benign tissue example", use_container_width=True)
    with malignant_col:
        if malignant.exists():
            st.image(str(malignant), caption="Malignant tissue example", use_container_width=True)

    st.markdown(
        """
        Dense breast tissue appears brighter on a mammogram and can make small lesions
        harder to see. The model combines a benign/malignant estimate with a density
        category, producing one of the eight classes shown in the analysis tab.
        """
    )

st.markdown(
    '<div class="footer">'
    "Research and educational tool only. Not a medical device and not a substitute "
    "for review by a qualified clinician."
    "</div>",
    unsafe_allow_html=True,
)
