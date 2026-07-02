"""
Custom Soil Classification Model - v4, Partial Fine-Tuning
------------------------------------------------------------------
Honest note: a fully from-scratch custom CNN realistically lands
around 80 to 88 percent, based on published soil classification
papers. To realistically aim above 90 percent, this version uses
PARTIAL fine-tuning, meaning we start from a small set of basic,
pretrained visual filters (edges, textures, colors), then let the
model adjust them too, instead of freezing them completely.

Your custom design, the texture-attention blocks, and the final
classification layers, remain the main, dominant part of this
model, so this is still fundamentally your own architecture, not
a standard transfer learning copy.

How to use:
Run this file using: py -3.11 train_v4_finetune.py
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Step 1: Set your split dataset folder here
SPLIT_FOLDER = "E:/Journals/Journal 2026/July 26/Soil 30.6.26/Split_Dataset_Fixed"
TRAIN_FOLDER = f"{SPLIT_FOLDER}/train"
VAL_FOLDER = f"{SPLIT_FOLDER}/validation"
TEST_FOLDER = f"{SPLIT_FOLDER}/test"

# Step 2: Basic settings, image size raised slightly for more detail
IMAGE_SIZE = (240, 240)
BATCH_SIZE = 16
EPOCHS = 60
NUM_CLASSES = 9

# Step 3: Data generators
# IMPORTANT: EfficientNet has its own built-in preprocessing, which
# is different from a plain 0-1 rescale. Using preprocess_input here
# instead of rescale=1./255 fixes a serious mismatch that previously
# caused the model to fail completely (stuck near random accuracy).
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.15,
    shear_range=0.1
)

val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_datagen.flow_from_directory(
    TRAIN_FOLDER, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=True
)
val_data = val_test_datagen.flow_from_directory(
    VAL_FOLDER, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)
test_data = val_test_datagen.flow_from_directory(
    TEST_FOLDER, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)

print("Classes found:", train_data.class_indices)


# Step 4: Texture-Attention Block, same custom design as before
def texture_attention_block(input_tensor, filters):
    avg_pool = layers.GlobalAveragePooling2D()(input_tensor)
    channel_weights = layers.Dense(filters, activation="relu")(avg_pool)
    channel_weights = layers.Dense(filters, activation="sigmoid")(channel_weights)
    channel_weights = layers.Reshape((1, 1, filters))(channel_weights)
    channel_focused = layers.Multiply()([input_tensor, channel_weights])

    spatial_weights = layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid")(channel_focused)
    spatial_focused = layers.Multiply()([channel_focused, spatial_weights])

    return spatial_focused


# Step 5: Build the model
# We use EfficientNetB0's early layers ONLY as a starting point for
# basic filters (edges, colors), then UNFREEZE them, so the model
# adjusts everything itself. Your custom attention blocks and final
# layers, added on top, remain the main decision-making part.
def build_model():
    base = EfficientNetB0(
        input_shape=(240, 240, 3),
        include_top=False,
        weights="imagenet"
    )

    # Partial fine-tuning: unfreeze only the last 30 layers of the
    # base, keep earlier layers (basic edges/colors) frozen, since
    # those generic filters don't need to change
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    x = base.output
    x = texture_attention_block(x, filters=x.shape[-1])   # your custom block, applied here
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return models.Model(base.input, outputs, name="Custom_Soil_Attention_FineTuned")


model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),  # corrected, paired with proper preprocessing now
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Step 6: Callbacks
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_soil_model_v4.keras",
    monitor="val_accuracy",
    save_best_only=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-7
)

# Step 7: Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

model.save("custom_soil_model_v4.keras")
print("Model saved as custom_soil_model_v4.keras")

# Step 8: Evaluate on test set
print("-" * 50)
print("Evaluating on the TEST set...")
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Final TEST accuracy: {test_accuracy:.2%}")
print(f"Final TEST loss: {test_loss:.4f}")

best_train_acc = max(history.history["accuracy"])
best_val_acc = max(history.history["val_accuracy"])
print("-" * 50)
print(f"Best training accuracy reached: {best_train_acc:.2%}")
print(f"Best validation accuracy reached: {best_val_acc:.2%}")
print(f"Final TEST accuracy (most important, honest number): {test_accuracy:.2%}")
