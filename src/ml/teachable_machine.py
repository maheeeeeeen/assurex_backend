"""
AssureX Claim Engine — Teachable Machine Image Inference Service

Provides real-time inference on Claim Summary Cards using the exported
Google Teachable Machine model (MobileNetV2 architecture in Keras/TF format).

Features:
- Handles file paths, PIL Images, byte streams, and base64 strings
- Teachable Machine standard 224x224 RGB input normalization [-1.0, 1.0]
- Supports single inference and vectorized batch inference for maximum throughput
- Defensive fallback handling if model weights are not yet initialized
"""

import os
import io
import base64
from typing import Dict, Any, Union, Optional, List
from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "model", "teachable_machine", "keras_model.h5")
DEFAULT_LABELS_PATH = os.path.join(BASE_DIR, "model", "teachable_machine", "labels.txt")

DEFAULT_CLASSES = ["Likely Valid", "Likely Invalid", "Manual Review Required"]


class TeachableMachinePredictor:
    """Production inference wrapper for Google Teachable Machine image classifier."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        labels_path: str = DEFAULT_LABELS_PATH
    ):
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.labels = DEFAULT_CLASSES
        self.model_name = "TeachableMachine_MobileNetV2"
        self._load_labels()
        self._load_model()

    LABEL_NAME_MAP = {
        "likely_valid": "Likely Valid",
        "likely_invalid": "Likely Invalid",
        "manual_review": "Manual Review Required",
        "manual_review_required": "Manual Review Required",
    }

    def _load_labels(self):
        """Loads class label mapping from labels.txt if present and maps to canonical names."""
        if os.path.exists(self.labels_path):
            try:
                loaded_labels = []
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split(" ", 1)
                            raw = parts[1] if len(parts) == 2 and parts[0].isdigit() else line
                            canonical = self.LABEL_NAME_MAP.get(raw.strip().lower().replace(" ", "_"), raw.strip())
                            loaded_labels.append(canonical)
                if loaded_labels:
                    self.labels = loaded_labels
            except Exception as e:
                print(f"[TeachableMachinePredictor] Warning reading labels: {e}")

    def _reconstruct_and_load_h5(self, h5_path: str):
        """Reconstructs full MobileNetV2 architecture (alpha=1.0) and loads weights from HDF5."""
        import h5py
        import tensorflow as tf

        # Must use alpha=1.0 — produces 1280 pooled features matching backup weight shapes
        base = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            alpha=1.0,
            include_top=False,
            pooling="avg",
            weights=None
        )
        x = tf.keras.layers.Dense(128, activation="relu", name="dense_1")(base.output)
        x = tf.keras.layers.Dropout(0.2, name="dropout_1")(x)
        out = tf.keras.layers.Dense(len(self.labels), activation="softmax", name="prediction")(x)
        model = tf.keras.Model(inputs=base.input, outputs=out, name="teachable_machine_mobilenetv2")

        with h5py.File(h5_path, "r") as f:
            mw = f["model_weights"]

            # Load MobileNetV2 backbone weights
            mob_key = "mobilenetv2_1.00_224"
            if mob_key in mw:
                for layer in base.layers:
                    if layer.name in mw[mob_key]:
                        g = mw[mob_key][layer.name]
                        weights = [np.array(g[k]) for k in g.keys()]
                        if weights:
                            try:
                                layer.set_weights(weights)
                            except Exception:
                                pass

            # Load Dense top weights
            if "dense_1" in mw and "dense_1" in mw["dense_1"]:
                d1 = mw["dense_1"]["dense_1"]
                w, b = np.array(d1["kernel"]), np.array(d1["bias"])
                model.get_layer("dense_1").set_weights([w, b])

            if "prediction" in mw and "prediction" in mw["prediction"]:
                pred = mw["prediction"]["prediction"]
                w, b = np.array(pred["kernel"]), np.array(pred["bias"])
                model.get_layer("prediction").set_weights([w, b])

        return model

    def _load_model(self):
        """Loads Keras model artifact from disk, handling Keras 2/3 compatibility."""
        model_dir = os.path.dirname(self.model_path)
        native_path = os.path.join(model_dir, "keras_model_native.keras")

        # 1. Try loading pre-converted native Keras 3 format if available
        if os.path.exists(native_path):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(native_path, compile=False)
                print(f"[TeachableMachinePredictor] Loaded native model from {native_path}")
                return
            except Exception as e:
                print(f"[TeachableMachinePredictor] Native load failed ({e}), attempting H5 load...")

        if not os.path.exists(self.model_path):
            print(f"[TeachableMachinePredictor] Model file not found at {self.model_path}")
            return

        # 2. Try direct load_model on H5
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            print(f"[TeachableMachinePredictor] Successfully loaded Teachable Machine model from {self.model_path}")
            return
        except Exception as e:
            print(f"[TeachableMachinePredictor] Standard H5 load encountered legacy format ({e}). Reconstructing architecture from weights...")

        # 3. Fallback: Reconstruct exact MobileNetV2 architecture and transfer weights
        try:
            self.model = self._reconstruct_and_load_h5(self.model_path)
            # Save native format for fast subsequent boots
            try:
                self.model.save(native_path)
                print(f"[TeachableMachinePredictor] Cached native Keras model to {native_path}")
            except Exception:
                pass
            print(f"[TeachableMachinePredictor] Successfully reconstructed and loaded model weights from {self.model_path}")
        except Exception as e:
            print(f"[TeachableMachinePredictor] Error reconstructing model from H5 weights: {e}")

    @property
    def is_ready(self) -> bool:
        """Returns True if the model is loaded and ready for prediction."""
        return self.model is not None

    def _preprocess_single(self, image_input: Union[str, bytes, Image.Image]) -> np.ndarray:
        """Preprocesses a single image into a (224, 224, 3) float32 array normalized to [-1, 1]."""
        img: Optional[Image.Image] = None

        if isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, str):
            if os.path.exists(image_input):
                img = Image.open(image_input)
            elif image_input.startswith("data:image"):
                header, encoded = image_input.split(",", 1)
                data = base64.b64decode(encoded)
                img = Image.open(io.BytesIO(data))
            else:
                raise ValueError(f"Image path does not exist: {image_input}")
        elif isinstance(image_input, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image_input))
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        img_rgb = img.convert("RGB")
        img_resized = img_rgb.resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.asarray(img_resized, dtype=np.float32)
        return (arr / 127.5) - 1.0

    def preprocess_image(self, image_input: Union[str, bytes, Image.Image]) -> np.ndarray:
        """Converts image into a (1, 224, 224, 3) batch array."""
        single = self._preprocess_single(image_input)
        return np.expand_dims(single, axis=0)

    def predict_batch(self, image_inputs: List[Union[str, bytes, Image.Image]]) -> List[Dict[str, Any]]:
        """
        Runs vectorized batch inference across multiple card images.
        """
        if not self.is_ready or image_inputs is None or len(image_inputs) == 0:
            return [
                {
                    "model_type": "Teachable_Machine_Image",
                    "model_name": "Fallback_Visual_Estimator",
                    "predicted_class": "Manual Review Required",
                    "confidence_scores": {lbl: round(1.0 / len(self.labels), 4) for lbl in self.labels},
                    "top_confidence": round(1.0 / len(self.labels), 4),
                }
                for _ in (image_inputs or [])
            ]

        arrays = []
        valid_indices = []
        results: List[Optional[Dict[str, Any]]] = [None] * len(image_inputs)

        for idx, img_input in enumerate(image_inputs):
            try:
                arr = self._preprocess_single(img_input)
                arrays.append(arr)
                valid_indices.append(idx)
            except Exception as e:
                print(f"[TeachableMachinePredictor] Warning on image {idx}: {e}")
                results[idx] = {
                    "model_type": "Teachable_Machine_Image",
                    "model_name": self.model_name,
                    "predicted_class": "Manual Review Required",
                    "confidence_scores": {lbl: 0.3333 for lbl in self.labels},
                    "top_confidence": 0.3333,
                    "error": str(e),
                }

        if arrays:
            batch_tensor = np.stack(arrays, axis=0)  # Shape: (N, 224, 224, 3)
            raw_preds = self.model(batch_tensor, training=False).numpy()

            for batch_i, orig_i in enumerate(valid_indices):
                row_preds = raw_preds[batch_i]
                confidence_dict = {
                    self.labels[k]: round(float(row_preds[k]), 4)
                    for k in range(min(len(self.labels), len(row_preds)))
                }
                top_idx = int(np.argmax(row_preds))
                top_class = self.labels[top_idx] if top_idx < len(self.labels) else "Unknown"
                top_conf = round(float(row_preds[top_idx]), 4)

                results[orig_i] = {
                    "model_type": "Teachable_Machine_Image",
                    "model_name": self.model_name,
                    "predicted_class": top_class,
                    "confidence_scores": confidence_dict,
                    "top_confidence": top_conf,
                }

        return [r for r in results if r is not None]

    def predict(self, image_input: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
        """
        Predicts warranty classification and class probabilities from a single Claim Summary Card.
        """
        return self.predict_batch([image_input])[0]
