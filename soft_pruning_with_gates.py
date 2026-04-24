"""
Self-Pruning Neural Network using Learnable Gates

This project implements a neural network that learns to prune itself during training
by associating each weight with a learnable gate parameter. The gating mechanism,
combined with L1 regularization, enables the model to suppress unimportant connections
and induce sparsity in a differentiable manner.

Key Components:
- Soft pruning via sigmoid-gated weights
- L1 regularization to promote sparsity in gate activations
- Threshold-based hard pruning for evaluation
- Systematic experimentation across lambda (λ) and threshold values
- Visualization of sparsity–accuracy trade-offs

Outputs:
- results/results.csv : Structured experiment results
- results/plots/      : Generated analysis plots

Author: Nitin Saini
"""

import os
import shutil
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# CLEAN PREVIOUS RESULTS
# -------------------------------------------------------
# Ensures reproducibility by removing previous outputs
if os.path.exists("results"):
    shutil.rmtree("results")

os.makedirs("results/plots", exist_ok=True)

# -------------------------------------------------------
# PRUNABLE LINEAR LAYER
# -------------------------------------------------------
class PrunableLinear(nn.Module):
    """
    Linear layer with learnable gates.
    Each weight is scaled by sigmoid(gate_score).
    """

    def __init__(self, in_f, out_f, init_type="zero"):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_f))

        init_val = 0.0 if init_type == "zero" else -0.5
        self.gate_scores = nn.Parameter(torch.ones(out_f, in_f) * init_val)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        return F.linear(x, self.weight * gates, self.bias)

# -------------------------------------------------------
# MODEL
# -------------------------------------------------------
class Net(nn.Module):
    def __init__(self, init_type="zero"):
        super().__init__()

        self.layers = nn.ModuleList([
            PrunableLinear(3072, 2048, init_type),
            PrunableLinear(2048, 1024, init_type),
            PrunableLinear(1024, 512, init_type),
            PrunableLinear(512, 256, init_type),
            PrunableLinear(256, 128, init_type),
            PrunableLinear(128, 10, init_type),
        ])

    def forward(self, x):
        x = x.view(x.size(0), -1)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)

        return x

# -------------------------------------------------------
# LOSS FUNCTION
# -------------------------------------------------------
def compute_loss(model, outputs, targets, lam):
    """
    Total Loss = CrossEntropy + λ * L1(gates)
    """
    ce_loss = F.cross_entropy(outputs, targets)

    sparsity_loss = 0
    count = 0

    for m in model.modules():
        if isinstance(m, PrunableLinear):
            sparsity_loss += torch.sigmoid(m.gate_scores).mean()
            count += 1

    sparsity_loss /= count
    return ce_loss + lam * sparsity_loss

# -------------------------------------------------------
# EVALUATION
# -------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total

def evaluate_hard(model, loader, threshold):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)

            for i, layer in enumerate(model.layers):
                gates = torch.sigmoid(layer.gate_scores)
                mask = (gates > threshold).float()
                x = F.linear(x, layer.weight * mask, layer.bias)

                if i < len(model.layers) - 1:
                    x = F.relu(x)

            pred = x.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total

def compute_sparsity(model, threshold):
    total, pruned = 0, 0

    for m in model.modules():
        if isinstance(m, PrunableLinear):
            g = torch.sigmoid(m.gate_scores)
            total += g.numel()
            pruned += (g < threshold).sum().item()

    return pruned / total

# -------------------------------------------------------
# DATA
# -------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("./data", train=True, download=True, transform=transform),
    batch_size=128, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("./data", train=False, download=True, transform=transform),
    batch_size=128, shuffle=False)

# -------------------------------------------------------
# TRAINING
# -------------------------------------------------------
def train_model(lam, init, epochs=20):
    model = Net(init).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(epochs):
        model.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = compute_loss(model, outputs, y, lam)
            loss.backward()
            optimizer.step()

        print(f"[TRAIN] init={init} λ={lam} epoch {ep+1}/{epochs}")

    return model

# -------------------------------------------------------
# EXPERIMENT LOOP
# -------------------------------------------------------
lambdas = [1e-3, 5e-3, 1e-2, 3e-2, 5e-2]
inits = ["zero", "neg05"]
thresholds = [0.2, 0.25, 0.3, 0.35, 0.4]

results = []

for init in inits:
    for lam in lambdas:

        print("\n" + "="*50)
        print(f"RUN: init={init}, lambda={lam}")
        print("="*50)

        model = train_model(lam, init)

        soft_acc = evaluate(model, test_loader)
        print(f"[RESULT] Soft Accuracy: {soft_acc:.4f}")

        for th in thresholds:
            sp = compute_sparsity(model, th)
            hard_acc = evaluate_hard(model, test_loader, th)

            print(f"  threshold={th:.2f} | sparsity={sp:.4f} | hard_acc={hard_acc:.4f}")

            results.append({
                "init": init,
                "lambda": lam,
                "threshold": th,
                "soft_acc": soft_acc,
                "hard_acc": hard_acc,
                "sparsity": sp
            })

# -------------------------------------------------------
# SAVE RESULTS
# -------------------------------------------------------
with open("results/results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)



# -------------------------------------------------------
# VISUAL ANALYSIS OF EXPERIMENT RESULTS
# -------------------------------------------------------
# The following plots are generated using the collected
# experiment results. These plots help analyze:
# 1. Effect of lambda on sparsity
# 2. Accuracy vs sparsity trade-off
# 3. Sensitivity to pruning threshold
# 4. Distribution of learned gate values

import matplotlib.ticker as mticker
import numpy as np

# =======================================================
# PLOT 1: Lambda vs Sparsity (with Hard Accuracy Annotation)
# =======================================================
# Shows how lambda controls sparsity.
# Each curve corresponds to a threshold value.
# Staggered annotation offsets prevent label overlap.

fig, ax = plt.subplots(figsize=(9, 5.5))

filtered = [
    r for r in results
    if r["init"] == "neg05"
    and round(r["threshold"], 2) in [0.2, 0.25, 0.3, 0.35]
]

target_lambdas = [0.001, 0.005, 0.01, 0.03, 0.05]

style = {
    0.20: {"color": "#185FA5", "marker": "o", "linestyle": "-",   "label": "th=0.20"},
    0.25: {"color": "#1D9E75", "marker": "^", "linestyle": "--",  "label": "th=0.25"},
    0.30: {"color": "#BA7517", "marker": "s", "linestyle": ":",   "label": "th=0.30"},
    0.35: {"color": "#A32D2D", "marker": "D", "linestyle": "-.",  "label": "th=0.35"},
}

# Staggered vertical offsets: alternating above/below per curve
# so annotations from adjacent curves never collide
v_offsets_by_th = {
    0.20: [ 10,  10,  10,  10,  10],
    0.25: [-14, -14, -14, -14, -14],
    0.30: [ 12, -14,  12, -14,  12],
    0.35: [-16,  12, -16,  12, -16],
}

for th in [0.20, 0.25, 0.30, 0.35]:
    lam_to_row = {
        round(r["lambda"], 4): r
        for r in filtered
        if round(r["threshold"], 2) == th
    }

    x, y, acc = [], [], []
    for lam in target_lambdas:
        key = round(lam, 4)
        if key in lam_to_row:
            x.append(lam)
            y.append(lam_to_row[key]["sparsity"])
            acc.append(lam_to_row[key]["hard_acc"])

    s = style[th]
    ax.plot(x, y, color=s["color"], marker=s["marker"],
            linestyle=s["linestyle"], linewidth=2,
            markersize=7, label=s["label"])

    offsets = v_offsets_by_th[th]
    for i, (xi, yi, ai) in enumerate(zip(x, y, acc)):
        voff = offsets[i] if i < len(offsets) else 10
        ax.annotate(
            f"{ai:.2f}",
            xy=(xi, yi),
            xytext=(0, voff),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=s["color"],
            arrowprops=dict(arrowstyle="-", color=s["color"], lw=0.6,
                            shrinkA=0, shrinkB=3) if abs(voff) > 11 else None,
        )

ax.set_xlabel("Lambda (λ)", fontsize=11)
ax.set_ylabel("Sparsity", fontsize=11)
ax.set_title("Lambda vs Sparsity (Init = neg05)\n"
             "Annotations show Hard Accuracy at each point", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend(title="Threshold")
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"{val:g}"))
plt.tight_layout()
plt.savefig("results/plots/lambda_vs_sparsity.png", dpi=150)
plt.close()


# =======================================================
# PLOT 2: Accuracy vs Sparsity (Labeled with bounding boxes)
# =======================================================
# Primary trade-off plot.
# Each label shows threshold + accuracy in a white-backed box
# with per-curve directional offsets to avoid overlap.

fig, ax = plt.subplots(figsize=(9, 6))

target_lambdas_p2 = [0.03, 0.05]
line_styles = {0.03: ("o", "-",  "#185FA5"), 0.05: ("s", "--", "#A32D2D")}

# Per-(lambda, threshold) offsets (dx_pts, dy_pts) chosen to avoid collision
label_offsets = {
    (0.03, 0.20): (-18,  8),
    (0.03, 0.25): (-18, -14),
    (0.03, 0.30): (-18,  8),
    (0.03, 0.35): ( 12,  8),
    (0.03, 0.40): (-30,  8),
    (0.05, 0.20): ( 12,  8),
    (0.05, 0.25): ( 12, -14),
    (0.05, 0.30): ( 12,  8),
    (0.05, 0.35): ( 12, -14),
    (0.05, 0.40): ( 12,  8),
}

for lam in target_lambdas_p2:
    mk, ls, col = line_styles[lam]
    pts = sorted(
        [(r["sparsity"], r["hard_acc"], r["threshold"])
         for r in results
         if r["init"] == "neg05" and r["lambda"] == lam],
        key=lambda p: p[0]
    )
    ax.plot([p[0] for p in pts], [p[1] for p in pts],
            marker=mk, linestyle=ls, color=col,
            linewidth=2, markersize=7, label=f"λ={lam}", zorder=3)

    for (sp, acc, th) in pts:
        if round(th, 2) == 0.40:
            ax.scatter(sp, acc, color="red", s=120, zorder=5)
        dx, dy = label_offsets.get((lam, round(th, 2)), (0, 10))
        ax.annotate(
            f"th={th:.2f}\n{acc:.3f}",
            xy=(sp, acc),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=col,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col,
                      alpha=0.75, lw=0.6),
            arrowprops=dict(arrowstyle="-", color=col, lw=0.7,
                            shrinkA=0, shrinkB=3),
        )

ax.set_xlabel("Sparsity", fontsize=11)
ax.set_ylabel("Hard Accuracy", fontsize=11)
ax.set_title("Accuracy vs Sparsity (Init = neg05)\n"
             "Red dot = threshold 0.40 (breaking point)", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend(title="Lambda")
plt.tight_layout()
plt.savefig("results/plots/accuracy_vs_sparsity.png", dpi=150)
plt.close()


# =======================================================
# PLOT 3: Threshold vs Accuracy (Sensitivity Analysis)
# =======================================================
# Shows how pruning threshold affects performance.
# Alternating offsets separate the two λ curves' annotations.
# Shaded region highlights the safe operating zone.

fig, ax = plt.subplots(figsize=(8, 5))

filtered3 = [
    r for r in results
    if r["init"] == "neg05"
    and round(r["lambda"], 3) in [0.03, 0.05]
]

target_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40]

style3 = {
    0.03: {"color": "#185FA5", "marker": "o", "linestyle": "-",  "label": "λ=0.03"},
    0.05: {"color": "#A32D2D", "marker": "s", "linestyle": "--", "label": "λ=0.05"},
}

# Alternating offsets: one curve annotated above, the other below at each x
v_offsets = {
    0.03: [ 10, -15,  10, -15,  10],
    0.05: [-15,  10, -15,  10, -15],
}

for lam in [0.03, 0.05]:
    th_to_row = {
        round(r["threshold"], 2): r
        for r in filtered3
        if round(r["lambda"], 3) == lam
    }
    x, y = [], []
    for th in target_thresholds:
        key = round(th, 2)
        if key in th_to_row:
            x.append(th)
            y.append(th_to_row[key]["hard_acc"])

    s = style3[lam]
    ax.plot(x, y, color=s["color"], marker=s["marker"],
            linestyle=s["linestyle"], linewidth=2,
            markersize=7, label=s["label"])

    offsets = v_offsets[lam]
    for i, (xi, yi) in enumerate(zip(x, y)):
        voff = offsets[i]
        ax.annotate(
            f"{yi:.3f}",
            xy=(xi, yi),
            xytext=(0, voff),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=s["color"],
            fontweight="semibold",
        )

# Highlight stable accuracy zone and cliff edge
ax.axvspan(0.245, 0.355, alpha=0.07, color="green", label="Sweet spot: moderate pruning (10%→45%)")
ax.axvline(x=0.35, linestyle=":", color="grey", alpha=0.6, linewidth=1.2,
           label="Cliff edge (~0.35)")

ax.set_xlabel("Threshold", fontsize=11)
ax.set_ylabel("Hard Accuracy", fontsize=11)
ax.set_title("Threshold vs Accuracy (Init = neg05)\n"
             "Accuracy collapses sharply beyond threshold 0.35", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend(title="Lambda", loc="lower left")
plt.tight_layout()
plt.savefig("results/plots/threshold_vs_accuracy.png", dpi=150)
plt.close()


# =======================================================
# PLOT 4: Gate Distribution (Model Behavior) — Enhanced
# =======================================================
# Shows distribution of learned gate values for the final model.
# Enhanced with:
#   - Twin y-axis showing cumulative pruning % (orange curve)
#   - Vertical lines for each threshold with % pruned annotation
#   - Mean gate line
#   - Stats summary box

all_gates_np = []
for m in model.modules():
    if isinstance(m, PrunableLinear):
        gates = torch.sigmoid(m.gate_scores).detach().cpu().numpy().flatten()
        all_gates_np.extend(gates)

all_gates_np = np.array(all_gates_np)

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

# Histogram on primary axis
counts, bin_edges, _ = ax1.hist(
    all_gates_np, bins=60, color="#3A7DC9", alpha=0.80,
    edgecolor="none", label="Gate count"
)

# Cumulative % on twin axis
cumulative = np.cumsum(counts) / counts.sum() * 100
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
ax2.plot(bin_centres, cumulative, color="#E05A2B", linewidth=2,
         label="Cumulative %", zorder=5)

# Threshold vertical lines with % pruned annotation
th_colors = {0.30: "#185FA5", 0.35: "#BA7517", 0.40: "#A32D2D"}
for th, tc in th_colors.items():
    pct = (all_gates_np < th).mean() * 100
    ax1.axvline(x=th, linestyle="--", color=tc, linewidth=1.4,
                label=f"th={th}  →  {pct:.1f}% pruned")
    ax1.text(th + 0.003, counts.max() * 0.82,
             f"th={th}\n{pct:.1f}%",
             fontsize=8, color=tc, va="top")

# Mean gate line
mean_g = float(np.mean(all_gates_np))
ax1.axvline(x=mean_g, linestyle=":", color="black", linewidth=1.2)
ax1.text(mean_g + 0.005, counts.max() * 0.55,
         f"mean={mean_g:.3f}", fontsize=8, color="black")

# Stats box
stats_text = (
    f"Total gates: {len(all_gates_np):,}\n"
    f"Mean: {mean_g:.3f}   Median: {np.median(all_gates_np):.3f}\n"
    f"Pruned @0.30: {(all_gates_np < 0.30).mean()*100:.1f}%\n"
    f"Pruned @0.35: {(all_gates_np < 0.35).mean()*100:.1f}%\n"
    f"Pruned @0.40: {(all_gates_np < 0.40).mean()*100:.1f}%"
)
ax1.text(0.02, 0.97, stats_text,
         transform=ax1.transAxes, fontsize=8.5, verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#aaaaaa", alpha=0.9))

ax1.set_xlabel("Gate Value  σ(gate_score)", fontsize=11)
ax1.set_ylabel("Frequency (count)", fontsize=11)
ax2.set_ylabel("Cumulative Gates Pruned (%)", fontsize=11, color="#E05A2B")
ax2.tick_params(axis="y", colors="#E05A2B")
ax2.set_ylim(0, 105)
ax1.set_title(
    "Gate Distribution — Final Model\n"
    "Orange curve: cumulative % of gates pruned at each threshold",
    fontsize=12
)
ax1.grid(True, linestyle="--", alpha=0.40)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5,
           loc="center right", framealpha=0.9)

plt.tight_layout()
plt.savefig("results/plots/gate_distribution.png", dpi=150)
plt.close()


print("\nAll results and plots saved in /results/plots")
