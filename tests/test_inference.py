import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class InferenceContractTests(unittest.TestCase):
    def test_class_names_match_model_outputs(self):
        from inference import CLASS_NAMES

        self.assertEqual(len(CLASS_NAMES), 8)
        self.assertEqual(len(set(CLASS_NAMES)), 8)

    def test_preprocessing_shape_and_range(self):
        try:
            import numpy as np
            from preprocess import preprocess_numpy_array
        except ImportError:
            self.skipTest("OpenCV or NumPy is not installed")

        image = np.full((300, 240, 3), 128, dtype=np.uint8)
        processed = preprocess_numpy_array(image)

        self.assertIsNotNone(processed)
        self.assertEqual(processed.shape, (224, 224, 3))
        self.assertGreaterEqual(float(processed.min()), 0.0)
        self.assertLessEqual(float(processed.max()), 1.0)

    def test_low_confidence_policy(self):
        from inference import LOW_CONFIDENCE_LABEL, top_prediction

        label, confidence, low = top_prediction({"A": 0.2, "B": 0.1})

        self.assertTrue(low)
        self.assertEqual(label, LOW_CONFIDENCE_LABEL)
        self.assertAlmostEqual(confidence, 0.2)

    def test_high_confidence_policy(self):
        from inference import top_prediction

        label, confidence, low = top_prediction({"A": 0.8, "B": 0.2})

        self.assertFalse(low)
        self.assertEqual(label, "A")
        self.assertAlmostEqual(confidence, 0.8)

    def test_saved_weight_layer_names_handles_both_layouts(self):
        try:
            import tempfile

            import h5py
            from model import LEGACY_LAYER_NAMES, saved_weight_layer_names
        except ImportError:
            self.skipTest("h5py is not installed")

        encoded = [name.encode() for name in LEGACY_LAYER_NAMES]
        with tempfile.TemporaryDirectory() as directory:
            root_layout = Path(directory) / "root.h5"
            nested_layout = Path(directory) / "nested.h5"

            with h5py.File(root_layout, "w") as handle:
                handle.attrs["layer_names"] = encoded

            with h5py.File(nested_layout, "w") as handle:
                handle.create_group("model_weights").attrs["layer_names"] = encoded

            self.assertEqual(saved_weight_layer_names(root_layout), list(LEGACY_LAYER_NAMES))
            self.assertEqual(saved_weight_layer_names(nested_layout), list(LEGACY_LAYER_NAMES))

    def test_model_predictions_on_example_images(self):
        weights_path = ROOT / "weights" / "modeldense1.h5"
        if not weights_path.exists():
            self.skipTest("Model weights are not available locally")
        try:
            import numpy as np
            from PIL import Image
            from inference import CLASS_NAMES, load_inference_model, predict_array
        except ImportError:
            self.skipTest("TensorFlow, OpenCV, or Pillow is not installed")

        model = load_inference_model(weights_path=weights_path)
        for path in sorted((ROOT / "Test_images").glob("*.jpg")):
            with self.subTest(image=path.name):
                image = np.array(Image.open(path).convert("RGB"))
                scores = predict_array(model, image)

                self.assertIsNotNone(scores)
                self.assertEqual(list(scores), CLASS_NAMES)
                self.assertAlmostEqual(sum(scores.values()), 1.0, places=4)
                for probability in scores.values():
                    self.assertGreaterEqual(probability, 0.0)
                    self.assertLessEqual(probability, 1.0)


if __name__ == "__main__":
    unittest.main()
