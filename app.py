import os
import tensorflow as tf
import numpy as np
from model import load_model
from weights import download_weights
from preprocess import preprocess_numpy_array
import gradio as gr

CLASS_NAMES = [
    "Benign with Density=1", "Malignant with Density=1",
    "Benign with Density=2", "Malignant with Density=2",
    "Benign with Density=3", "Malignant with Density=3",
    "Benign with Density=4", "Malignant with Density=4",
]

CSS = """
.gradio-container { max-width: 1120px !important; margin: auto !important; }

.header-wrapper { text-align: center; padding: 1.5rem 0 0; }
.header-title {
    font-size: 2rem; font-weight: 700; color: #1e293b; margin: 0 0 0.35rem;
}
.header-accent { color: #ea580c; }
.header-subtitle {
    font-size: 1.05rem; color: #64748b; margin-bottom: 0.75rem; line-height: 1.5;
}
.header-badge {
    display: inline-block; background: #fff7ed; color: #c2410c;
    padding: 0.25rem 0.85rem; border-radius: 999px; font-size: 0.78rem;
    font-weight: 600; border: 1px solid #fed7aa; margin-bottom: 0.75rem;
}

.upload-section { padding-right: 0.5rem; }
.results-section { padding-left: 0.5rem; }

.guide-card {
    background: #f8fafc; border-radius: 14px; padding: 1.5rem;
    border: 1px solid #e2e8f0; margin-bottom: 1rem;
}
.guide-card h3 { margin: 0 0 0.5rem; color: #1e293b; font-size: 1.1rem; }
.guide-card p  { margin: 0; color: #475569; font-size: 0.9rem; line-height: 1.6; }

.pipeline-step {
    display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 1rem;
}
.pipeline-num {
    width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #f97316, #ea580c);
    color: #fff; display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85rem; flex-shrink: 0;
}
.pipeline-text h4 { margin: 0 0 0.15rem; color: #1e293b; font-size: 0.95rem; }
.pipeline-text p  { margin: 0; color: #64748b; font-size: 0.85rem; line-height: 1.5; }

.footer-disclaimer {
    text-align: center; color: #94a3b8; font-size: 0.82rem;
    margin-top: 2.5rem; padding: 1.25rem; border-top: 1px solid #e2e8f0;
}
"""


def main():
    print("Loading model...")
    model = load_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        metrics=["accuracy"],
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    )
    print("Downloading weights if needed...")
    download_weights()
    model.load_weights("weights/modeldense1.h5")
    print("Model ready.\n")

    def predict_img(img):
        if img is None:
            return {name: 0.0 for name in CLASS_NAMES}

        processed = preprocess_numpy_array(img)
        if processed is None:
            return {name: 0.0 for name in CLASS_NAMES}
        if processed.shape != (224, 224, 3):
            return {name: 0.0 for name in CLASS_NAMES}

        im = processed.reshape(-1, 224, 224, 3)
        pred = model.predict(im, verbose=0)[0]

        top_idx = np.argmax(pred)
        if pred[top_idx] < 0.3:
            return {"Low confidence — consider re-uploading": 1.0}

        return {CLASS_NAMES[i]: float(pred[i]) for i in range(8)}

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="gray"),
        css=CSS,
        title="OncologyBreastScanner",
    ) as demo:
        # ── Header ──
        gr.HTML(
            '<div class="header-wrapper">'
            '<span class="header-badge">Deep Learning &bull; DenseNet201 &bull; CBIS-DDSM</span>'
            '<h1 class="header-title">'
            'Oncology<span class="header-accent">Breast</span>Scanner'
            '</h1>'
            '<p class="header-subtitle">'
            'AI-Powered Mammogram Analysis &mdash; '
            'Breast Density Classification &amp; Malignancy Detection'
            '</p>'
            '</div>'
        )

        # ── Tabs ──
        with gr.Tabs():
            # ──── Tab 1: Analyze ────
            with gr.TabItem("Analyze"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=320, elem_classes="upload-section"):
                        input_image = gr.Image(
                            type="numpy",
                            image_mode="RGB",
                            label="Upload Mammogram",
                            height=400,
                        )
                        with gr.Row():
                            analyze_btn = gr.Button(
                                "Analyze", variant="primary", scale=2
                            )
                            clear_btn = gr.Button(
                                "Clear", variant="secondary", scale=1
                            )

                        gr.Markdown("#### Example Cases")
                        gr.Examples(
                            examples=[
                                "Test_images/Density1Malignant.jpg",
                                "Test_images/Density2Malignant.jpg",
                                "Test_images/Density3Benign.jpg",
                                "Test_images/Density4Benign.jpg",
                            ],
                            inputs=input_image,
                            label=None,
                        )

                    with gr.Column(scale=1, min_width=320, elem_classes="results-section"):
                        output_label = gr.Label(
                            label="Analysis Results",
                            num_top_classes=8,
                        )

            # ──── Tab 2: How It Works ────
            with gr.TabItem("How It Works"):
                gr.Markdown("## Processing Pipeline")
                gr.Markdown(
                    "The model preprocesses each mammogram through a series of "
                    "steps before classification to improve prediction accuracy."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML(
                            '<div class="guide-card">'
                            '<div class="pipeline-step">'
                            '<div class="pipeline-num">1</div>'
                            '<div class="pipeline-text">'
                            '<h4>Color Normalization</h4>'
                            '<p>Converts grayscale or RGBA images to standard RGB format '
                            'to ensure consistent color channels across all inputs.</p>'
                            '</div></div>'
                            '<div class="pipeline-step">'
                            '<div class="pipeline-num">2</div>'
                            '<div class="pipeline-text">'
                            '<h4>Image Resizing</h4>'
                            '<p>Resizes the mammogram to 224&times;224 pixels, '
                            'matching the input dimensions required by DenseNet201.</p>'
                            '</div></div>'
                            '<div class="pipeline-step">'
                            '<div class="pipeline-num">3</div>'
                            '<div class="pipeline-text">'
                            '<h4>Contrast Enhancement</h4>'
                            '<p>Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) '
                            'in the LAB color space to enhance subtle tissue details.</p>'
                            '</div></div>'
                            '<div class="pipeline-step">'
                            '<div class="pipeline-num">4</div>'
                            '<div class="pipeline-text">'
                            '<h4>Sharpening &amp; Normalization</h4>'
                            '<p>Applies a sharpening convolution kernel to enhance edges, '
                            'then normalizes pixel values to the [0, 1] range for inference.</p>'
                            '</div></div>'
                            '</div>'
                        )
                    with gr.Column(scale=1):
                        gr.Image(
                            value="image/processing.png",
                            label="Preprocessing Visualization",
                            show_label=True,
                            height=340,
                        )
                        gr.Image(
                            value="image/after cropping.png",
                            label="After Cropping &amp; Enhancement",
                            show_label=True,
                            height=260,
                        )

                gr.Markdown("## Model Architecture")
                gr.Markdown(
                    "The classifier uses **DenseNet201** pre-trained on ImageNet "
                    "as its backbone, with a custom classification head: "
                    "Global Average + Max Pooling &rarr; Dense(1024) &rarr; "
                    "Dense(512) &rarr; Softmax over 8 classes."
                )

            # ──── Tab 3: Visual Guide ────
            with gr.TabItem("Visual Guide"):
                gr.Markdown("## Understanding Mammogram Findings")
                gr.Markdown(
                    "The model classifies mammograms into 8 categories by combining "
                    "tissue density (BI-RADS 1&ndash;4) with pathology (benign vs malignant)."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Image(
                            value="image/benign.png",
                            label="Benign Finding",
                            show_label=True,
                        )
                        gr.Markdown(
                            "*A benign mass typically presents with smooth, "
                            "well-circumscribed borders and uniform density.*"
                        )
                    with gr.Column(scale=1):
                        gr.Image(
                            value="image/malignant.png",
                            label="Malignant Finding",
                            show_label=True,
                        )
                        gr.Markdown(
                            "*Malignant lesions often show irregular, spiculated "
                            "borders and architectural distortion.*"
                        )

                gr.Markdown("---")
                gr.Markdown("## BI-RADS Density Categories")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "| Level | Description |\n"
                            "|-------|-------------|\n"
                            "| **1** | Almost entirely fatty |\n"
                            "| **2** | Scattered fibroglandular densities |\n"
                            "| **3** | Heterogeneously dense |\n"
                            "| **4** | Extremely dense |"
                        )
                    with gr.Column(scale=1):
                        gr.Image(
                            value="image/prediction.png",
                            label="Sample Prediction Output",
                            show_label=True,
                        )

        # ── Footer ──
        gr.HTML(
            '<div class="footer-disclaimer">'
            '<strong>Research Use Only</strong> &mdash; '
            'This tool is for educational and research purposes. '
            'It is not a medical device and should not be used for clinical diagnosis. '
            'Always consult a qualified healthcare professional.'
            '</div>'
        )

        def clear_all():
            empty = {name: 0.0 for name in CLASS_NAMES}
            return None, empty

        analyze_btn.click(
            fn=predict_img,
            inputs=[input_image],
            outputs=[output_label],
        )

        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[input_image, output_label],
        )

    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))


if __name__ == "__main__":
    main()
