"""
Ablation Study - All Versions (V1 to V5)
------------------------------------------
This script trains five model versions for the ablation study.
Each version removes or changes one part of the proposed model,
therefore it helps to show the contribution of each component.

Run: py -3.11 ablation_study.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (classification_report, roc_auc_score)
from sklearn.preprocessing import label_binarize
import pandas as pd

# ─── PATHS ───────────────────────────────────────────────────────────────────
SPLIT_FOLDER = "E:/Journals/Journal 2026/July 26/Soil 30.6.26/Split_Dataset_Fixed"
TRAIN_FOLDER = f"{SPLIT_FOLDER}/train"
VAL_FOLDER   = f"{SPLIT_FOLDER}/validation"
TEST_FOLDER  = f"{SPLIT_FOLDER}/test"
SAVE_DIR     = r"C:\Users\sarav\Desktop\Ablation"
os.makedirs(SAVE_DIR, exist_ok=True)

IMAGE_SIZE  = (240, 240)
BATCH_SIZE  = 16
EPOCHS      = 30
NUM_CLASSES = 9

# ─── DATA GENERATORS ─────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.15
)
val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_datagen.flow_from_directory(
    TRAIN_FOLDER, target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True)

val_data = val_test_datagen.flow_from_directory(
    VAL_FOLDER, target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)

test_data = val_test_datagen.flow_from_directory(
    TEST_FOLDER, target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)

# ─── TEXTURE ATTENTION BLOCK ─────────────────────────────────────────────────
def texture_attention_block(input_tensor, filters):
    # Channel attention
    avg_pool = layers.GlobalAveragePooling2D()(input_tensor)
    cw = layers.Dense(filters, activation="relu")(avg_pool)
    cw = layers.Dense(filters, activation="sigmoid")(cw)
    cw = layers.Reshape((1, 1, filters))(cw)
    cf = layers.Multiply()([input_tensor, cw])
    # Spatial attention
    sw = layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid")(cf)
    return layers.Multiply()([cf, sw])

def channel_attention_only(input_tensor, filters):
    avg_pool = layers.GlobalAveragePooling2D()(input_tensor)
    cw = layers.Dense(filters, activation="relu")(avg_pool)
    cw = layers.Dense(filters, activation="sigmoid")(cw)
    cw = layers.Reshape((1, 1, filters))(cw)
    return layers.Multiply()([input_tensor, cw])

def spatial_attention_only(input_tensor):
    sw = layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid")(input_tensor)
    return layers.Multiply()([input_tensor, sw])

# ─── CUSTOM CLASSIFICATION LAYERS ────────────────────────────────────────────
def add_custom_layers(x):
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    return layers.Dense(NUM_CLASSES, activation="softmax")(x)

def add_simple_layers(x):
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    return layers.Dense(NUM_CLASSES, activation="softmax")(x)

# ─── GET EFFICIENTNETB0 BASE ──────────────────────────────────────────────────
def get_base():
    base = EfficientNetB0(
        input_shape=(240, 240, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    return base

# ─── BUILD MODELS ─────────────────────────────────────────────────────────────

# Version 1: EfficientNetB0 only, no attention, no custom layers
def build_v1():
    base = get_base()
    x = base.output
    x = add_simple_layers(x)
    return models.Model(base.input, x, name="V1_EfficientNetB0_Only")

# Version 2: EfficientNetB0 + custom layers, no attention
def build_v2():
    base = get_base()
    x = base.output
    x = add_custom_layers(x)
    return models.Model(base.input, x, name="V2_EfficientNetB0_CustomLayers")

# Version 3: EfficientNetB0 + attention block, simple layers
def build_v3():
    base = get_base()
    x = base.output
    x = texture_attention_block(x, filters=x.shape[-1])
    x = add_simple_layers(x)
    return models.Model(base.input, x, name="V3_EfficientNetB0_AttentionOnly")

# Version 4: EfficientNetB0 + custom layers + channel attention only
def build_v4():
    base = get_base()
    x = base.output
    x = channel_attention_only(x, filters=x.shape[-1])
    x = add_custom_layers(x)
    return models.Model(base.input, x, name="V4_EfficientNetB0_ChannelAttention")

# Version 5: EfficientNetB0 + custom layers + spatial attention only
def build_v5():
    base = get_base()
    x = base.output
    x = spatial_attention_only(x)
    x = add_custom_layers(x)
    return models.Model(base.input, x, name="V5_EfficientNetB0_SpatialAttention")

# ─── TRAIN AND EVALUATE ───────────────────────────────────────────────────────
def train_and_evaluate(model, version_name):
    print(f"\n{'='*60}")
    print(f"Training {version_name}")
    print(f"{'='*60}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    save_path = os.path.join(SAVE_DIR, f"{version_name}.keras")

    early_stop = EarlyStopping(
        monitor="val_accuracy", patience=7, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        save_path, monitor="val_accuracy", save_best_only=True)

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    # Evaluate on test set
    print(f"\nEvaluating {version_name} on test set...")
    y_pred_probs = model.predict(test_data)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_data.classes

    # Reset test data for next model
    test_data.reset()
    train_data.reset()

    # Calculate metrics
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0)

    accuracy   = (y_pred == y_true).mean() * 100
    precision  = report["macro avg"]["precision"]
    recall     = report["macro avg"]["recall"]
    f1         = report["macro avg"]["f1-score"]

    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    auc = roc_auc_score(y_true_bin, y_pred_probs, average="macro")

    print(f"Accuracy  : {accuracy:.2f}%")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"AUC       : {auc:.4f}")

    return {
        "Version"    : version_name,
        "Accuracy (%)" : round(accuracy, 2),
        "Precision"  : round(precision, 4),
        "Recall"     : round(recall, 4),
        "F1 Score"   : round(f1, 4),
        "AUC"        : round(auc, 4)
    }

# ─── RUN ALL VERSIONS ─────────────────────────────────────────────────────────
versions = [
    (build_v1, "V1_EfficientNetB0_Only"),
    (build_v2, "V2_EfficientNetB0_CustomLayers"),
    (build_v3, "V3_EfficientNetB0_AttentionOnly"),
    (build_v4, "V4_EfficientNetB0_ChannelAttention"),
    (build_v5, "V5_EfficientNetB0_SpatialAttention"),
]

results = []
for build_fn, version_name in versions:
    model = build_fn()
    result = train_and_evaluate(model, version_name)
    results.append(result)
    tf.keras.backend.clear_session()

# Add Version 6 results manually since it is already trained
results.append({
    "Version"      : "V6_Full_Proposed_Model",
    "Accuracy (%)" : 88.57,
    "Precision"    : 0.0,  # replace with real value from Table 4
    "Recall"       : 0.0,  # replace with real value from Table 4
    "F1 Score"     : 0.0,  # replace with real value from Table 4
    "AUC"          : 0.0   # replace with real value from Table 4
})

# ─── SAVE RESULTS ─────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
print("\n" + "="*60)
print("ABLATION STUDY RESULTS")
print("="*60)
print(df.to_string(index=False))

save_csv = os.path.join(SAVE_DIR, "Ablation_Study_Results.csv")
df.to_csv(save_csv, index=False)
print(f"\nAblation results saved to: {save_csv}")
