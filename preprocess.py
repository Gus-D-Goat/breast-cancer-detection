import cv2
import numpy as np

SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])


def preprocess_mammogram(image, apply_clahe=False, apply_sharpen=True, target_size=(224, 224)):
    if image is None:
        return None

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    image = cv2.resize(image, target_size)

    if apply_clahe:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    if apply_sharpen:
        image = cv2.filter2D(image, -1, SHARPEN_KERNEL)

    return image


def normalize_for_model(image):
    image = image / 255.0
    return image


def load_and_preprocess(path, target_size=(224, 224), apply_clahe=False, apply_sharpen=True):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_mammogram(img, apply_clahe=apply_clahe, apply_sharpen=apply_sharpen, target_size=target_size)
    if img is None:
        return None
    img = normalize_for_model(img)
    return img


def preprocess_numpy_array(img_array, target_size=(224, 224), apply_clahe=False, apply_sharpen=True):
    if img_array is None or img_array.size == 0:
        return None
    if img_array.max() <= 1.0 and img_array.dtype.kind == 'f':
        img_array = (img_array * 255).astype(np.uint8)
    elif img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    img = preprocess_mammogram(img_array, apply_clahe=apply_clahe, apply_sharpen=apply_sharpen, target_size=target_size)
    if img is None:
        return None
    img = normalize_for_model(img)
    return img
