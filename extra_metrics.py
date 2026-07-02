"""
Complete Testing Metrics Table
--------------------------------
Calculates all 8 metrics for each soil class:
Accuracy, Precision, Recall, Specificity,
F1-Score, Kappa, AUC, Mean Average Precision

Run: py -3.11 extra_metrics.py
"""

import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import (confusion_matrix, cohen_kappa_score,
                             roc_auc_score, average_precision_score)
from sklearn.preprocessing import label_binarize

MODEL_PATH  = "E:/Journals/Journal 2026/July 26/Soil 30.6.26/best_soil_model_v4.keras"
TEST_FOLDER = "E:/Journals/Journal 2026/July 26/Soil 30.6.26/Split_Dataset_Fixed/test"
SAVE_PATH   = r"C:\Users\sarav\Desktop\Tables\Table4_Testing_Metrics.csv"

IMAGE_SIZE  = (240, 240)
BATCH_SIZE  = 16
NUM_CLASSES = 9

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded.")

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_data = datagen.flow_from_directory(
    TEST_FOLDER, target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)

class_names = list(test_data.class_indices.keys())
print("Classes:", class_names)

print("\nRunning predictions...")
y_pred_probs = model.predict(test_data)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_data.classes

cm = confusion_matrix(y_true, y_pred)
y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
kappa = cohen_kappa_score(y_true, y_pred)
overall_accuracy = (y_pred == y_true).mean()

metrics_data = []

for i, class_name in enumerate(class_names):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - TP - FP - FN

    precision   = TP / (TP + FP)   if (TP + FP) > 0 else 0
    recall      = TP / (TP + FN)   if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP)   if (TN + FP) > 0 else 0
    f1          = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0)
    accuracy    = (TP + TN) / cm.sum()
    auc         = roc_auc_score(y_true_bin[:, i], y_pred_probs[:, i])
    map_score   = average_precision_score(y_true_bin[:, i], y_pred_probs[:, i])

    metrics_data.append({
        "Soil Class"            : class_name.replace("_", " "),
        "Accuracy (%)"          : round(accuracy * 100, 2),
        "Precision"             : round(precision, 4),
        "Recall"                : round(recall, 4),
        "Specificity"           : round(specificity, 4),
        "F1-Score"              : round(f1, 4),
        "Kappa"                 : round(kappa, 4),
        "AUC"                   : round(auc, 4),
        "Mean Avg Precision"    : round(map_score, 4)
    })

df = pd.DataFrame(metrics_data)

macro_row = pd.DataFrame([{
    "Soil Class"            : "Macro Average",
    "Accuracy (%)"          : round(df["Accuracy (%)"].mean(), 2),
    "Precision"             : round(df["Precision"].mean(), 4),
    "Recall"                : round(df["Recall"].mean(), 4),
    "Specificity"           : round(df["Specificity"].mean(), 4),
    "F1-Score"              : round(df["F1-Score"].mean(), 4),
    "Kappa"                 : round(kappa, 4),
    "AUC"                   : round(df["AUC"].mean(), 4),
    "Mean Avg Precision"    : round(df["Mean Avg Precision"].mean(), 4)
}])

df = pd.concat([df, macro_row], ignore_index=True)

print("\n" + "=" * 90)
print("TABLE 4: Complete Testing Metrics")
print("=" * 90)
print(df.to_string(index=False))
print("=" * 90)
print(f"\nOverall Test Accuracy   : {overall_accuracy*100:.2f}%")
print(f"Cohen Kappa Coefficient : {kappa:.4f}")

os.makedirs(r"C:\Users\sarav\Desktop\Tables", exist_ok=True)
df.to_csv(SAVE_PATH, index=False)
print(f"\nTable 4 saved to Desktop as Table4_Testing_Metrics.csv")
