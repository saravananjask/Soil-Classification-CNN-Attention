import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH  = r"E:\Journals\Journal 2026\July 26\Soil 30.6.26\Results\Table_KFold_Results.csv"
SAVE_PATH = r"C:\Users\sarav\Desktop\Fig5_CV_BoxPlot.png"

df = pd.read_csv(CSV_PATH)
print("Loaded k-fold results:")
print(df)

fold_accuracies = [float(v) / 100 for v in df["Validation Accuracy (%)"]]

fig, ax = plt.subplots(figsize=(8, 7))
ax.boxplot(
    [acc * 100 for acc in fold_accuracies],
    patch_artist=True,
    widths=0.4,
    boxprops=dict(facecolor="steelblue", color="navy"),
    medianprops=dict(color="red", linewidth=2),
    whiskerprops=dict(color="navy", linewidth=1.5),
    capprops=dict(color="navy", linewidth=1.5),
)

for i, acc in enumerate(fold_accuracies):
    ax.scatter(1, acc * 100, color="darkred", zorder=5, s=60)
    ax.annotate(f"Fold {i+1}: {acc*100:.1f}%",
                xy=(1, acc * 100),
                xytext=(1.15, acc * 100),
                fontsize=9, color="darkred")

ax.set_title("Figure 5: 5-Fold Cross Validation Accuracy",
             fontsize=14, fontweight="bold")
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_xlabel("Custom Soil CNN Model", fontsize=12)
ax.set_xticks([1])
ax.set_xticklabels(["Custom Soil CNN"])
ax.set_ylim(50, 105)
ax.axhline(y=np.mean(fold_accuracies) * 100,
           color="green", linestyle="--", linewidth=1.5,
           label=f"Mean: {np.mean(fold_accuracies)*100:.1f}%")
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150)
plt.close()
print("Figure 5 saved to Desktop successfully.")
