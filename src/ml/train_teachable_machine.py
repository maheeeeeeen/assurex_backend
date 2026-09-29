"""
AssureX Claim Engine — Teachable Machine Image Classifier Training & Export

Trains a MobileNetV2-based computer vision classifier on Claim Summary Cards,
matching the exact architecture, input size (224x224 RGB), normalization ([-1, 1]),
and export format of Google Teachable Machine.

Features:
- Fast in-memory caching for zero CPU starvation
- MobileNetV2 transfer learning feature extraction
- Full end-to-end Keras model export to keras_model.h5
- labels.txt generation matching Google TM convention
- Comprehensive validation metrics and evaluation report
"""

import os
import sys
import json
import time
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CARDS_DIR = os.path.join(BASE_DIR, "sample_claims", "cards")
MODEL_DIR = os.path.join(BASE_DIR, "model", "teachable_machine")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

TRAIN_DIR = os.path.join(CARDS_DIR, "train")
VAL_DIR = os.path.join(CARDS_DIR, "val")
TEST_DIR = os.path.join(CARDS_DIR, "test")

CLASS_NAMES = ["likely_valid", "likely_invalid", "manual_review"]
DISPLAY_LABELS = ["Likely Valid", "Likely Invalid", "Manual Review Required"]


def load_card_dataset(dir_path: str, max_per_class: int = None, seed: int = 42):
    """
    Loads and pre-resizes cards to (224, 224) into memory.
    Applies Teachable Machine normalization: (pixel / 127.5) - 1.0.
    """
    random.seed(seed)
    images = []
    labels = []

    print(f"Loading cards from {dir_path} (max_per_class={max_per_class})...")
    t0 = time.time()

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_folder = os.path.join(dir_path, class_name)
        if not os.path.exists(class_folder):
            continue

        filenames = [f for f in os.listdir(class_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        random.shuffle(filenames)

        if max_per_class is not None and len(filenames) > max_per_class:
            filenames = filenames[:max_per_class]

        print(f"  [{class_name}]: loading {len(filenames)} images...")
        for fname in filenames:
            fpath = os.path.join(class_folder, fname)
            try:
                with Image.open(fpath) as img:
                    img_rgb = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                    arr = np.array(img_rgb, dtype=np.float32)
                    norm_arr = (arr / 127.5) - 1.0
                    images.append(norm_arr)
                    labels.append(class_idx)
            except Exception as e:
                print(f"    Warning: Failed to load {fpath}: {e}")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    elapsed = time.time() - t0
    print(f"Loaded {len(X)} images in {elapsed:.2f}s | Shape: {X.shape}")
    return X, y


def train_and_export():
    print("=" * 65)
    print("AssureX Claim Engine — Phase 3: Teachable Machine Image Model Training")
    print("=" * 65)

    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        raise FileNotFoundError(f"Card image directories not found at {CARDS_DIR}!")

    # 1. Load balanced training images and full validation set
    X_train, y_train = load_card_dataset(TRAIN_DIR, max_per_class=300, seed=42)
    X_val, y_val = load_card_dataset(VAL_DIR, max_per_class=None, seed=42)

    # One-hot encode targets
    Y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
    Y_val = tf.keras.utils.to_categorical(y_val, num_classes=3)

    # 2. Build MobileNetV2 base feature extractor
    print("\nInitializing MobileNetV2 backbone (ImageNet weights)...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    # Extract bottleneck features in memory for fast training
    print("Extracting feature representations for training set...")
    t_feat0 = time.time()
    feats_train = base_model.predict(X_train, batch_size=64, verbose=1)
    feats_val = base_model.predict(X_val, batch_size=64, verbose=1)
    print(f"Feature extraction done in {time.time() - t_feat0:.2f}s | Train feats: {feats_train.shape}")

    # 3. Build and train classification head
    head_inputs = layers.Input(shape=feats_train.shape[1:])
    x = layers.GlobalAveragePooling2D()(head_inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    head_outputs = layers.Dense(3, activation="softmax", name="prediction")(x)

    head_model = models.Model(head_inputs, head_outputs, name="TM_Classifier_Head")
    head_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nTraining classification head (20 epochs)...")
    t_train0 = time.time()
    history = head_model.fit(
        feats_train,
        Y_train,
        epochs=20,
        batch_size=32,
        validation_data=(feats_val, Y_val),
        verbose=1
    )
    print(f"Classification head trained in {time.time() - t_train0:.2f}s")

    # 4. Construct End-to-End Teachable Machine Model
    print("\nAssembling end-to-end Teachable Machine pipeline...")
    end_to_end_inputs = layers.Input(shape=(224, 224, 3), name="card_image_input")
    x = base_model(end_to_end_inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    # Re-use trained weights from head_model
    dense1_layer = layers.Dense(128, activation="relu", name="dense_1")
    output_layer = layers.Dense(3, activation="softmax", name="prediction")

    # Copy weights
    dense1_weights = head_model.layers[3].get_weights()
    output_weights = head_model.layers[4].get_weights()

    x = dense1_layer(x)
    end_to_end_outputs = output_layer(x)

    full_model = models.Model(inputs=end_to_end_inputs, outputs=end_to_end_outputs, name="TeachableMachine_MobileNetV2")
    full_model.layers[-2].set_weights(dense1_weights)
    full_model.layers[-1].set_weights(output_weights)

    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 5. Evaluate End-to-End Model on Full Validation Set
    print("\nEvaluating End-to-End Model on Validation Cards...")
    val_preds = full_model.predict(X_val, batch_size=32, verbose=1)
    y_pred = np.argmax(val_preds, axis=1)

    val_acc = np.mean(y_pred == y_val)
    val_f1 = f1_score(y_val, y_pred, average="weighted")
    print(f"\n==========================================")
    print(f"Final Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Final Validation F1:       {val_f1*100:.2f}%")
    print(f"==========================================")

    print("\nClassification Report (Teachable Machine Model):")
    report_str = classification_report(y_val, y_pred, target_names=DISPLAY_LABELS, digits=4)
    print(report_str)

    # 6. Save Model Artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    h5_path = os.path.join(MODEL_DIR, "keras_model.h5")
    full_model.save(h5_path)
    print(f"\nSaved Teachable Machine Model: {h5_path}")

    # Write labels.txt
    labels_path = os.path.join(MODEL_DIR, "labels.txt")
    with open(labels_path, "w") as f:
        for idx, lbl in enumerate(DISPLAY_LABELS):
            f.write(f"{idx} {lbl}\n")
    print(f"Saved labels: {labels_path}")

    # Save metadata
    meta = {
        "model_architecture": "MobileNetV2 + GlobalAveragePooling + Dense(128) + Dense(3, Softmax)",
        "input_shape": [224, 224, 3],
        "normalization": "[-1.0, 1.0]",
        "training_images_sampled": len(X_train),
        "validation_images": len(X_val),
        "val_accuracy": round(float(val_acc), 4),
        "val_weighted_f1": round(float(val_f1), 4),
        "classes": DISPLAY_LABELS,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nPhase 3 Teachable Machine Training & Export Completed Successfully!")
    return full_model, meta


if __name__ == "__main__":
    train_and_export()
