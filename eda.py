"""
eda.py — Exploratory Data Analysis for California Housing dataset.

Generates plots saved to plots/eda/ directory:
    - Feature distributions (histograms)
    - Correlation heatmap
    - Feature vs Target scatter plots
    - Target distribution

Run with:
    python eda.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = "data.csv"
OUTPUT_DIR = "plots/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_DESCRIPTIONS = {
    "MedInc":     "Median Income (10k USD)",
    "HouseAge":   "Median House Age (years)",
    "AveRooms":   "Avg Rooms per Household",
    "AveBedrms":  "Avg Bedrooms per Household",
    "Population": "Block Population",
    "AveOccup":   "Avg Household Occupancy",
    "Latitude":   "Latitude",
    "Longitude":  "Longitude",
    "target":     "Median House Value (100k USD)",
}

sns.set_theme(style="whitegrid", palette="muted")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"\n{'='*55}")
    print(f"  Dataset: {path}")
    print(f"  Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Missing: {df.isnull().sum().sum()} values")
    print(f"{'='*55}\n")
    print(df.describe().round(3).to_string())
    print()
    return df


# ── Plot 1: Target Distribution ───────────────────────────────────────────────
def plot_target_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Target Variable — Median House Value (100k USD)", fontsize=13)

    axes[0].hist(df["target"], bins=50, color="steelblue", edgecolor="white")
    axes[0].set_title("Distribution")
    axes[0].set_xlabel("House Value (100k USD)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(df["target"].mean(), color="red", linestyle="--", label=f"Mean: {df['target'].mean():.2f}")
    axes[0].axvline(df["target"].median(), color="orange", linestyle="--", label=f"Median: {df['target'].median():.2f}")
    axes[0].legend()

    axes[1].boxplot(df["target"], vert=True, patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.6))
    axes[1].set_title("Boxplot")
    axes[1].set_ylabel("House Value (100k USD)")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR}/01_target_distribution.png")


# ── Plot 2: Feature Distributions ─────────────────────────────────────────────
def plot_feature_distributions(df: pd.DataFrame) -> None:
    features = [c for c in df.columns if c != "target"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Feature Distributions", fontsize=13)
    axes = axes.flatten()

    for i, col in enumerate(features):
        axes[i].hist(df[col], bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        axes[i].set_title(FEATURE_DESCRIPTIONS.get(col, col), fontsize=9)
        axes[i].set_xlabel(col, fontsize=8)
        axes[i].set_ylabel("Count", fontsize=8)
        axes[i].tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR}/02_feature_distributions.png")


# ── Plot 3: Correlation Heatmap ───────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df.corr().round(2)

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # show lower triangle
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5,
        ax=ax, cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Matrix", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR}/03_correlation_heatmap.png")

    # Print top correlations with target
    target_corr = corr["target"].drop("target").sort_values(ascending=False)
    print("\n  Feature correlations with target:")
    for feat, val in target_corr.items():
        bar = "█" * int(abs(val) * 20)
        print(f"    {feat:12s} {val:+.3f}  {bar}")
    print()


# ── Plot 4: Feature vs Target Scatter ─────────────────────────────────────────
def plot_feature_vs_target(df: pd.DataFrame) -> None:
    features = [c for c in df.columns if c != "target"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Feature vs Target (Median House Value)", fontsize=13)
    axes = axes.flatten()

    for i, col in enumerate(features):
        sample = df.sample(min(2000, len(df)), random_state=42)  # subsample for speed
        axes[i].scatter(sample[col], sample["target"],
                        alpha=0.2, s=5, color="steelblue")
        # Trend line
        z = np.polyfit(sample[col], sample["target"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(sample[col].min(), sample[col].max(), 100)
        axes[i].plot(x_line, p(x_line), color="red", linewidth=1.5)

        axes[i].set_xlabel(col, fontsize=8)
        axes[i].set_ylabel("Target", fontsize=8)
        axes[i].set_title(FEATURE_DESCRIPTIONS.get(col, col), fontsize=9)
        axes[i].tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_feature_vs_target.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR}/04_feature_vs_target.png")


# ── Plot 5: Geographic Heatmap ─────────────────────────────────────────────────
def plot_geographic(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 10))
    sc = ax.scatter(
        df["Longitude"], df["Latitude"],
        c=df["target"], cmap="YlOrRd",
        alpha=0.4, s=3
    )
    plt.colorbar(sc, ax=ax, label="House Value (100k USD)")
    ax.set_title("Geographic Distribution of House Values\n(California)", fontsize=13)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_geographic_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR}/05_geographic_heatmap.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nRunning EDA on California Housing dataset...")
    df = load_data(DATA_PATH)

    print("Generating plots:")
    plot_target_distribution(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_feature_vs_target(df)
    plot_geographic(df)

    print(f"\n All EDA plots saved to {OUTPUT_DIR}/\n")