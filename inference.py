from weights import WEIGHTS_PATH, download_weights

CLASS_NAMES = [
    "Benign with Density=1",
    "Malignant with Density=1",
    "Benign with Density=2",
    "Malignant with Density=2",
    "Benign with Density=3",
    "Malignant with Density=3",
    "Benign with Density=4",
    "Malignant with Density=4",
]

INPUT_SHAPE = (224, 224, 3)
LOW_CONFIDENCE_THRESHOLD = 0.3
LOW_CONFIDENCE_LABEL = "Low confidence — consider re-uploading"


def load_inference_model(weights_path=WEIGHTS_PATH, weights_url=None):
    from model import build_model

    path = download_weights(dest=weights_path, url=weights_url)
    model = build_model(
        input_shape=INPUT_SHAPE,
        n_classes=len(CLASS_NAMES),
        pretrained=False,
    )
    model.load_weights(path)
    return model


def predict_array(model, image):
    if image is None:
        return None

    from preprocess import preprocess_numpy_array

    processed = preprocess_numpy_array(image)
    if processed is None or processed.shape != INPUT_SHAPE:
        return None

    batch = processed.reshape(-1, *INPUT_SHAPE)
    prediction = model.predict(batch, verbose=0)[0]
    return {name: float(prediction[i]) for i, name in enumerate(CLASS_NAMES)}


def top_prediction(scores):
    if not scores:
        return None, 0.0, True

    label = max(scores, key=scores.get)
    confidence = scores[label]
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return LOW_CONFIDENCE_LABEL, confidence, True

    return label, confidence, False
