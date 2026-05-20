"""
Generate every figure produced by the benchmark.

Main figures:
  Figure 1   UMAP grid (uncorrected vs. five correction methods,
             two rows: by batch / by cell-type).
  Figure 2   Metric heat-map + Overall-score barplot.

Supplementary figures (S1 and S2 are saved by 01_data_preparation.py):
  S3   per-method silhouette curves (cell-type silhouette by group)
  S4   runtime barplot
  S5   per-cell-type batch silhouette (sensitivity to rare types)
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import (
    DATA_DIR, FIGURES_DIR, METRICS_DIR,
    load_embedding, load_preprocessed, set_plot_style,
)

set_plot_style()
SEED = 42

METHOD_ORDER = [
    "Uncorrected", "Seurat CCA", "Harmony", "scVI", "Scanorama", "ComBat",
]
METHOD_KEYS = {
    "Uncorrected": "X_pca",
    "Seurat CCA":  "X_seurat",
    "Harmony":     "X_harmony",
    "scVI":        "X_scvi",
    "Scanorama":   "X_scanorama",
    "ComBat":      "X_combat",
}
METHOD_FILES = {
    "Seurat CCA": "seurat",
    "Harmony":    "harmony",
    "scVI":       "scvi",
    "Scanorama":  "scanorama",
    "ComBat":     "combat",
}


def load_all(adata: sc.AnnData) -> sc.AnnData:
    """Load every saved embedding and (re-)compute UMAP for each method."""
    print("\nLoading embeddings and recomputing UMAPs...")
    for method in METHOD_ORDER:
        key = METHOD_KEYS[method]
        if key not in adata.obsm:
            try:
                emb = load_embedding(METHOD_FILES[method])
            except FileNotFoundError:
                print(f"  [skip] {method}: embedding file missing")
                continue
            adata.obsm[key] = emb.astype(np.float32)

        sc.pp.neighbors(adata, use_rep=key, n_neighbors=15, random_state=SEED)
        sc.tl.umap(adata, random_state=SEED, min_dist=0.4)
        adata.obsm[f"X_umap_{key}"] = adata.obsm["X_umap"].copy()
        print(f"  {method:<12s}  UMAP({adata.obsm[f'X_umap_{key}'].shape})")
    return adata


def figure1_umap_grid(adata: sc.AnnData) -> None:
    print("\nFigure 1: UMAP grid")
    methods = [m for m in METHOD_ORDER
                if f"X_umap_{METHOD_KEYS[m]}" in adata.obsm]
    n = len(methods)

    fig, axes = plt.subplots(
        2, n, figsize=(2.4 * n, 5.8),
        gridspec_kw=dict(wspace=0.06, hspace=0.08),
    )

    batch_pal = dict(zip(
        adata.obs["batch"].cat.categories,
        sns.color_palette("Set2", adata.obs["batch"].nunique()),
    ))
    ct_pal = dict(zip(
        adata.obs["celltype"].cat.categories,
        sns.color_palette("tab10", adata.obs["celltype"].nunique()),
    ))

    def normalize_umap(arr: np.ndarray) -> np.ndarray:
        """Rescale a 2-D embedding to the unit square so panels fill axes."""
        a = arr.astype(np.float32).copy()
        for d in range(2):
            lo, hi = a[:, d].min(), a[:, d].max()
            a[:, d] = (a[:, d] - lo) / (hi - lo + 1e-9)
        return a

    for j, name in enumerate(methods):
        umap = normalize_umap(adata.obsm[f"X_umap_{METHOD_KEYS[name]}"])

        ax_b = axes[0, j]
        for b in adata.obs["batch"].cat.categories:
            m = (adata.obs["batch"] == b).values
            ax_b.scatter(umap[m, 0], umap[m, 1], s=1.6, alpha=0.7,
                         c=[batch_pal[b]], rasterized=True,
                         linewidths=0, label=b)
        ax_b.set_xticks([]); ax_b.set_yticks([])
        ax_b.set_xlim(-0.04, 1.04); ax_b.set_ylim(-0.04, 1.04)
        ax_b.set_box_aspect(1)
        ax_b.set_title(name, fontsize=11, fontweight="bold")
        if j == 0:
            ax_b.set_ylabel("Batch", fontsize=11, fontweight="bold")
        for s in ax_b.spines.values():
            s.set_visible(False)

        ax_c = axes[1, j]
        for ct in adata.obs["celltype"].cat.categories:
            m = (adata.obs["celltype"] == ct).values
            ax_c.scatter(umap[m, 0], umap[m, 1], s=1.6, alpha=0.7,
                         c=[ct_pal[ct]], rasterized=True,
                         linewidths=0, label=ct)
        ax_c.set_xticks([]); ax_c.set_yticks([])
        ax_c.set_xlim(-0.04, 1.04); ax_c.set_ylim(-0.04, 1.04)
        ax_c.set_box_aspect(1)
        if j == 0:
            ax_c.set_ylabel("Cell type", fontsize=11, fontweight="bold")
        for s in ax_c.spines.values():
            s.set_visible(False)

    batch_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                    color=c, label=b, markersize=6)
        for b, c in batch_pal.items()
    ]
    ct_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                    color=c, label=ct, markersize=6)
        for ct, c in ct_pal.items()
    ]
    axes[0, -1].legend(handles=batch_handles, frameon=False,
                        bbox_to_anchor=(1.02, 0.5), loc="center left",
                        fontsize=8, title="Batch", title_fontsize=9)
    axes[1, -1].legend(handles=ct_handles, frameon=False,
                        bbox_to_anchor=(1.02, 0.5), loc="center left",
                        fontsize=8, title="Cell type", title_fontsize=9)

    fig.tight_layout()
    out_png = os.path.join(FIGURES_DIR, "Figure1_UMAP_comparison.png")
    out_pdf = os.path.join(FIGURES_DIR, "Figure1_UMAP_comparison.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf,             bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_png}")


def figure2_metrics() -> None:
    print("\nFigure 2: metric heatmap + Overall")
    path = os.path.join(METRICS_DIR, "benchmark_summary.csv")
    df = pd.read_csv(path, index_col=0).reindex(METHOD_ORDER).dropna(how="all")

    metric_cols = [
        "Silhouette_batch", "kBET", "Graph_connectivity",
        "Silhouette_celltype", "ARI", "NMI",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    nice = {
        "Silhouette_batch":    "Silhouette\n(batch)",
        "kBET":                "kBET",
        "Graph_connectivity":  "Graph\nconnectivity",
        "Silhouette_celltype": "Silhouette\n(cell type)",
        "ARI":                 "ARI",
        "NMI":                 "NMI",
    }

    fig = plt.figure(figsize=(11, 5.0))
    gs  = fig.add_gridspec(
        1, 5,
        width_ratios=[1.05, 1.6, 0.10, 1.4, 0.05],
        wspace=0.30,
    )
    ax_h = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[0, 3])

    heat = df[metric_cols].copy()
    heat.columns = [nice[c] for c in metric_cols]
    sns.heatmap(
        heat, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0, vmax=1, cbar=True, ax=ax_h,
        linewidths=0.4, linecolor="white",
        annot_kws={"fontsize": 9},
        cbar_kws={"label": "Score"},
    )
    n_batch = 3
    n_total = len(metric_cols)
    ax_h.add_patch(plt.Rectangle(
        (0, 0), n_batch, len(heat), fill=False, ec="#0d4d8c", lw=2,
    ))
    ax_h.add_patch(plt.Rectangle(
        (n_batch, 0), n_total - n_batch, len(heat),
        fill=False, ec="#7d2d62", lw=2,
    ))
    ax_h.text(
        n_batch / 2, -0.18, "Batch\nremoval",
        ha="center", va="bottom",
        color="#0d4d8c", fontsize=9, fontweight="bold",
        linespacing=0.95,
        transform=ax_h.transData, clip_on=False,
    )
    ax_h.text(
        n_batch + (n_total - n_batch) / 2, -0.18,
        "Biological\nconservation",
        ha="center", va="bottom",
        color="#7d2d62", fontsize=9, fontweight="bold",
        linespacing=0.95,
        transform=ax_h.transData, clip_on=False,
    )
    ax_h.set_yticklabels(ax_h.get_yticklabels(), rotation=0)
    ax_h.set_xlabel("")
    ax_h.set_title("(a) Per-metric scores", fontsize=11, fontweight="bold",
                    pad=42, loc="left")

    df_sorted = df.sort_values("Overall", ascending=True)
    bars = ax_b.barh(
        df_sorted.index, df_sorted["Overall"],
        color=sns.color_palette("viridis", len(df_sorted)),
        edgecolor="white",
    )
    for bar, v in zip(bars, df_sorted["Overall"]):
        ax_b.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                   f"{v:.3f}", va="center", fontsize=9)
    ax_b.set_xlim(0, max(0.95, df_sorted["Overall"].max() * 1.15))
    ax_b.set_xlabel("Overall score  (0.4*Batch + 0.6*Bio)")
    ax_b.set_title("(b) Aggregate ranking", fontsize=11, fontweight="bold",
                    pad=10)
    for s in ("top", "right"):
        ax_b.spines[s].set_visible(False)

    out_png = os.path.join(FIGURES_DIR, "Figure2_metrics_comparison.png")
    out_pdf = os.path.join(FIGURES_DIR, "Figure2_metrics_comparison.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf,             bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_png}")


def supp_silhouette(adata: sc.AnnData) -> None:
    print("\nSupp. Fig. S3: silhouette curves")
    methods = [m for m in METHOD_ORDER if METHOD_KEYS[m] in adata.obsm]
    n = len(methods)

    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 3.6), sharey=True)
    ct_pal = dict(zip(
        adata.obs["celltype"].cat.categories,
        sns.color_palette("tab10", adata.obs["celltype"].nunique()),
    ))

    for ax, name in zip(axes, methods):
        emb = adata.obsm[METHOD_KEYS[name]]
        sil = silhouette_samples(emb, adata.obs["celltype"].astype(str).values)
        y0 = 0
        for ct in adata.obs["celltype"].cat.categories:
            m = (adata.obs["celltype"] == ct).values
            vals = np.sort(sil[m])
            y = np.arange(y0, y0 + len(vals))
            ax.fill_betweenx(y, 0, vals, color=ct_pal[ct], alpha=0.8)
            y0 += len(vals) + 5

        avg = float(np.mean(sil))
        ax.axvline(avg, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{name}\n(mean = {avg:.3f})", fontsize=9)
        ax.set_xlabel("Silhouette")
        ax.set_yticks([])
        ax.set_xlim(-0.4, 1.0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=ct)
        for ct, c in ct_pal.items()
    ]
    axes[-1].legend(
        handles=handles, frameon=False, bbox_to_anchor=(1.02, 0.5),
        loc="center left", fontsize=7, title="Cell type", title_fontsize=8,
    )
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "FigureS3_silhouette.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")


def supp_runtime() -> None:
    print("\nSupp. Fig. S4: runtime barplot")
    path = os.path.join(METRICS_DIR, "runtimes.csv")
    if not os.path.exists(path):
        print("  no runtime CSV")
        return
    df = pd.read_csv(path)
    name_map = {
        "Harmony": "Harmony", "scVI": "scVI",
        "Scanorama": "Scanorama", "ComBat": "ComBat",
        "Combat": "ComBat", "Seurat CCA": "Seurat CCA",
    }
    df["method"] = df["method"].map(lambda m: name_map.get(m, m))
    df = df.groupby("method", as_index=False)["runtime_seconds"].mean()
    df = (df.set_index("method")
            .reindex([m for m in METHOD_ORDER if m != "Uncorrected"])
            .dropna()
            .reset_index())

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    colors = sns.color_palette("viridis", len(df))
    bars = ax.bar(df["method"], df["runtime_seconds"], color=colors,
                   edgecolor="white")
    for bar, v in zip(bars, df["runtime_seconds"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v:.1f}s", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Wall-clock runtime (s)")
    ax.set_title("Runtime per method", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "FigureS4_runtime.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")


def supp_per_celltype_batch_mixing(adata: sc.AnnData) -> None:
    print("\nSupp. Fig. S5: per-cell-type batch mixing")
    methods = [m for m in METHOD_ORDER if METHOD_KEYS[m] in adata.obsm]
    rows = []
    cts = sorted(
        adata.obs["celltype"].cat.categories,
        key=lambda c: -(adata.obs["celltype"] == c).sum(),
    )
    for ct in cts:
        mask = (adata.obs["celltype"] == ct).values
        if mask.sum() < 15:
            continue
        b_sub = adata.obs.loc[mask, "batch"].astype(str).values
        if len(np.unique(b_sub)) < 2:
            continue
        for name in methods:
            try:
                s = silhouette_score(
                    adata.obsm[METHOD_KEYS[name]][mask], b_sub,
                )
                rows.append({
                    "Cell type":   ct,
                    "Method":      name,
                    "Batch mixing": 1 - (s + 1) / 2,
                    "n_cells":     int(mask.sum()),
                })
            except Exception:
                continue
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    palette = sns.color_palette("viridis", len(methods))
    sns.barplot(
        data=df, x="Cell type", y="Batch mixing", hue="Method",
        hue_order=methods, ax=ax, palette=palette, edgecolor="white",
    )
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-cell-type batch mixing  (higher = better)",
                  fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
               frameon=False, fontsize=8)
    ax.tick_params(axis="x", rotation=20)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "FigureS5_celltype_mixing.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")


def main() -> None:
    print("=" * 60)
    print("  STEP 5: VISUALISATION")
    print("=" * 60)

    combined = os.path.join(DATA_DIR, "pancreas_with_embeddings.h5ad")
    if os.path.exists(combined):
        adata = sc.read_h5ad(combined)
    else:
        adata = load_preprocessed()

    adata = load_all(adata)

    figure1_umap_grid(adata)
    figure2_metrics()
    supp_silhouette(adata)
    supp_runtime()
    supp_per_celltype_batch_mixing(adata)

    print("\n[OK] all figures written to", FIGURES_DIR)


if __name__ == "__main__":
    main()
