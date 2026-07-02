"""
Recreate Figures 3 and 4 using original training history values
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "E:/Journals/Journal 2026/July 26/Soil 30.6.26/Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Original training history values captured from PowerShell screenshots
epochs = list(range(1, 27))

train_acc = [
    0.5037, 0.6140, 0.6680, 0.7102, 0.7353,
    0.7573, 0.7785, 0.7951, 0.8057, 0.9617,
    0.9665, 0.9702, 0.9758, 0.9774, 0.9810,
    0.9827, 0.9848, 0.9841, 0.9859, 0.9855,
    0.9868, 0.9845, 0.9884, 0.9896, 0.9891,
    0.9889
]

val_acc = [
    0.2622, 0.6018, 0.5960, 0.6760, 0.5029,
    0.7189, 0.7362, 0.7049, 0.6661, 0.8782,
    0.8883, 0.8832, 0.8934, 0.9036, 0.8934,
    0.9086, 0.8934, 0.9036, 0.8934, 0.9036,
    0.8985, 0.9036, 0.8985, 0.9036, 0.9036,
    0.8985
]

train_loss = [
    1.4497, 1.0920, 0.9189, 0.7963, 0.7308,
    0.6786, 0.6146, 0.5678, 0.5459, 0.1290,
    0.1074, 0.0915, 0.0758, 0.0727, 0.0574,
    0.0544, 0.0480, 0.0571, 0.0441, 0.0472,
    0.0425, 0.0448, 0.0374, 0.0362, 0.0362,
    0.0376
]

val_loss = [
    2.8147, 1.1172, 1.2765, 0.9910, 2.0600,
    0.8673, 0.8630, 1.0009, 1.2928, 0.5254,
    0.5517, 0.5209, 0.5458, 0.4454, 0.5424,
    0.5145, 0.5374, 0.5355, 0.5529, 0.5553,
    0.5654, 0.5583, 0.5461, 0.5779, 0.5592,
    0.5510
]

# ─── FIGURE 3: Accuracy Curve ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs, [a * 100 for a in train_acc],
        "b-o", linewidth=2, markersize=4, label="Training Accuracy")
ax.plot(epochs, [a * 100 for a in val_acc],
        "r-s", linewidth=2, markersize=4, label="Validation Accuracy")
ax.set_title("Figure 3: Training and Validation Accuracy Curve",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)
ax.set_xticks(epochs)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "Fig3_Accuracy_Curve.png"), dpi=150)
plt.close()
print("Figure 3 saved.")

# ─── FIGURE 4: Loss Curve ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs, train_loss,
        "b-o", linewidth=2, markersize=4, label="Training Loss")
ax.plot(epochs, val_loss,
        "r-s", linewidth=2, markersize=4, label="Validation Loss")
ax.set_title("Figure 4: Training and Validation Loss Curve",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(epochs)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "Fig4_Loss_Curve.png"), dpi=150)
plt.close()
print("Figure 4 saved.")

print("Done. Figures 3 and 4 saved to Results folder.")
