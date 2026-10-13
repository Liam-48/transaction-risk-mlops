import mlflow
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from src.data import load_data, split_data
from src.evaluation import evaluate_at_capacity


# -----------------------------
# Configuration
# -----------------------------
MLFLOW_URI = "http://127.0.0.1:5000"
MODEL_URI = "models:/bank-fraud-risk/2"  
OUTPUT_PATH = "reports/review_capacity.png"

# Palette
PRIMARY = "#1D3557"      # deep blue
ACCENT = "#2A9D8F"       # teal
BASELINE = "#9AA5B1"     # muted gray
GRID = "#D9E2EC"
TEXT = "#243B53"

# -----------------------------
# Load model and test data
# -----------------------------
mlflow.set_tracking_uri(MLFLOW_URI)
model = mlflow.pyfunc.load_model(MODEL_URI)

df = load_data("data/Base.csv")
X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)

y_proba = np.asarray(model.predict(X_test))[:, 1]

capacities = np.linspace(0.005, 0.10, 20)
recalls = []

for capacity in capacities:
    result = evaluate_at_capacity(y_test, y_proba, capacity)
    recalls.append(result["recall"])

recalls = np.array(recalls)

# Highlight 1% review capacity
highlight_capacity = 0.01
highlight_result = evaluate_at_capacity(y_test, y_proba, highlight_capacity)
highlight_recall = highlight_result["recall"]

x = capacities * 100
y = recalls * 100
baseline_y = capacities * 100

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=180)

# Background
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Shaded advantage area
ax.fill_between(
    x, baseline_y, y,
    alpha=0.12,
    color=ACCENT
)

# Main model line
ax.plot(
    x, y,
    linewidth=3.0,
    color=PRIMARY,
    label="XGBoost model",
    zorder=3
)

# Random baseline
ax.plot(
    x, baseline_y,
    linestyle="--",
    linewidth=2.0,
    color=BASELINE,
    label="Random selection",
    zorder=2
)

# Highlight 1% operating point
ax.scatter(
    [highlight_capacity * 100],
    [highlight_recall * 100],
    s=95,
    color=ACCENT,
    edgecolor="white",
    linewidth=1.5,
    zorder=5
)

# Annotation
ax.annotate(
    "1% review capacity\nRecall: 22.1%",
    xy=(highlight_capacity * 100, highlight_recall * 100),
    xytext=(2.3, 32),
    textcoords="data",
    arrowprops=dict(
        arrowstyle="-",
        color=ACCENT,
        lw=1.6
    ),
    fontsize=11,
    color=TEXT,
    ha="left",
    va="center",
    bbox=dict(
        boxstyle="round,pad=0.45",
        fc="white",
        ec="#D8E1E8",
        lw=1.1
    )
)

# Titles
ax.set_title(
    "Fraud Detection vs. Investigation Capacity",
    fontsize=20,
    color=TEXT,
    pad=14,
    weight="semibold"
)

# Axis labels
ax.set_xlabel("Applications reviewed (%)", fontsize=12, color=TEXT, labelpad=10)
ax.set_ylabel("Fraud recall (%)", fontsize=12, color=TEXT, labelpad=10)

# Limits
ax.set_xlim(0.5, 10.5)
ax.set_ylim(0, max(y.max(), baseline_y.max()) + 5)

# Grid
ax.grid(True, axis="y", color=GRID, linewidth=0.9)
ax.grid(False, axis="x")

# Spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#BCCCDC")
ax.spines["bottom"].set_color("#BCCCDC")

# Ticks
ax.tick_params(axis="both", colors=TEXT, labelsize=11)

# Legend
leg = ax.legend(
    frameon=False,
    loc="upper left",
    fontsize=11
)
for text in leg.get_texts():
    text.set_color(TEXT)

plt.tight_layout()

Path("reports").mkdir(exist_ok=True)
plt.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor="white")
plt.show()