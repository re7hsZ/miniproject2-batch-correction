"""
Download the scIB human pancreas benchmark and run standard preprocessing.

Data source
-----------
Zenodo mirror of the SCIB-cleaned pancreas integration benchmark
(Luecken et al., 2022, Nature Methods):
  human_pancreas_norm_complexBatch.h5ad  (16 382 cells, 19 093 genes, 9 batches)

Workflow
--------
1.  Download / load raw benchmark h5ad
2.  Per-batch QC violin plot                      (-> results/figures/FigureS1_QC.png)
3.  Library-size normalisation + log1p
4.  Batch-aware HVG selection (Seurat v3 flavour, raw counts)
5.  Scale + 50-PC PCA + variance scree            (-> results/figures/FigureS2_PCA_variance.png)
6.  Neighbour graph + UMAP + Leiden (uncorrected baseline)
7.  Export CSV files for the Seurat (R) script    (-> data/pancreas_counts.csv,
                                                      data/pancreas_metadata.csv)

Output: data/pancreas_preprocessed.h5ad
"""

from __future__ import annotations

import os
import sys
import warnings
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import scanpy as sc

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import DATA_DIR, FIGURES_DIR, set_plot_style

set_plot_style()

ZENODO_URL = (
    "https://zenodo.org/api/records/18336458/files/"
    "human_pancreas_norm_complexBatch.h5ad/content"
)
RAW_FILENAME = "human_pancreas_norm_complexBatch.h5ad"


def download_raw_dataset(dest: str) -> None:
    """Download the scIB pancreas benchmark from Zenodo if not already present."""
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000_000:
        print(f"  Raw dataset already present -> {dest}")
        return

    tmp = dest + ".part"
    print(f"  Downloading scIB pancreas benchmark from Zenodo ...")
    print(f"  URL: {ZENODO_URL}")

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = block_num * block_size
        pct = min(100.0, 100.0 * done / total_size)
        print(f"\r  Progress: {pct:5.1f}%", end="", flush=True)

    urlretrieve(ZENODO_URL, tmp, reporthook=_progress)
    print()
    os.replace(tmp, dest)
    print(f"  Saved -> {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")


def load_raw_dataset(path: str) -> sc.AnnData:
    """Load the benchmark h5ad and harmonise obs column names for the pipeline."""
    adata = sc.read_h5ad(path)
    print(f"  Loaded raw AnnData: {adata.shape}")

    if "counts" not in adata.layers:
        raise ValueError("Expected raw counts in adata.layers['counts'].")

    adata.obs["batch"] = adata.obs["tech"].astype(str).astype("category")
    adata.obs["celltype"] = adata.obs["celltype"].astype(str).astype("category")

    adata.X = adata.layers["counts"].copy()
    print(f"  Batches ({adata.obs['batch'].nunique()}): "
          f"{adata.obs['batch'].value_counts().to_dict()}")
    print(f"  Cell types ({adata.obs['celltype'].nunique()}): "
          f"{adata.obs['celltype'].value_counts().to_dict()}")
    return adata


def preprocess(adata: sc.AnnData) -> sc.AnnData:
    """Standard Scanpy preprocessing + uncorrected baseline (UMAP + Leiden)."""
    print("\n" + "=" * 60)
    print("  PREPROCESSING")
    print("=" * 60)

    adata.layers["counts"] = adata.X.copy()

    print("\n1. Quality Control")
    print(f"   Before filtering: {adata.shape[0]} cells, {adata.shape[1]} genes")

    sc.pp.calculate_qc_metrics(
        adata, percent_top=None, log1p=False, inplace=True,
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    sc.pl.violin(adata, "n_genes_by_counts", groupby="batch",
                 ax=axes[0], show=False, rotation=30, stripplot=False)
    axes[0].set_title("Genes / cell")
    sc.pl.violin(adata, "total_counts", groupby="batch",
                 ax=axes[1], show=False, rotation=30, stripplot=False)
    axes[1].set_title("Total UMI / cell")
    axes[1].set_yscale("log")
    sc.pl.violin(adata, "n_genes_by_counts", groupby="batch",
                 ax=axes[2], show=False, rotation=30, stripplot=False)
    axes[2].set_title("Detected genes")
    for ax in axes:
        ax.set_xlabel("")
    fig.suptitle("Per-batch QC metrics (pre-filtering)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "FigureS1_QC.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("   Saved Fig. S1.")

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=10)
    print(f"   After filtering : {adata.shape[0]} cells, {adata.shape[1]} genes")
    for b, c in adata.obs["batch"].value_counts().items():
        print(f"     {b:<26s} {c:>5d}")

    print("\n2. Normalisation + log1p")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()

    print("\n3. Highly-variable gene selection (Seurat v3, batch-aware)")
    sc.pp.highly_variable_genes(
        adata, layer="counts", n_top_genes=2000,
        flavor="seurat_v3", batch_key="batch", subset=False,
    )
    n_hvg = int(adata.var["highly_variable"].sum())
    print(f"   Selected {n_hvg} HVGs.")

    adata_hvg = adata[:, adata.var["highly_variable"]].copy()

    print("\n4. Scale + PCA")
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=50, svd_solver="arpack", random_state=42)
    print(f"   X_pca shape: {adata_hvg.obsm['X_pca'].shape}")

    var_ratio = adata_hvg.uns["pca"]["variance_ratio"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(range(1, len(var_ratio) + 1), var_ratio,
           color="#4c72b0", edgecolor="white")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance ratio")
    ax.set_title("PCA variance explained (top 50 PCs)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "FigureS2_PCA_variance.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n5. Neighbours, UMAP (uncorrected) + Leiden")
    sc.pp.neighbors(adata_hvg, n_pcs=30, n_neighbors=15, random_state=42)
    sc.tl.umap(adata_hvg, random_state=42, min_dist=0.4)
    adata_hvg.obsm["X_umap_uncorrected"] = adata_hvg.obsm["X_umap"].copy()
    sc.tl.leiden(
        adata_hvg, resolution=0.8, random_state=42,
        key_added="leiden_uncorrected", flavor="igraph",
        n_iterations=2, directed=False,
    )
    n_clu = adata_hvg.obs["leiden_uncorrected"].nunique()
    print(f"   Leiden (uncorrected) -> {n_clu} clusters")
    return adata_hvg


def export_for_seurat(adata: sc.AnnData) -> None:
    """Export raw counts on HVGs + metadata for the Seurat (R) integration."""
    import pandas as pd
    import scipy.sparse as sp

    print("\n6. Exporting CSVs for the Seurat (R) script")
    counts = adata.layers["counts"]
    if sp.issparse(counts):
        counts = counts.toarray()
    df_counts = pd.DataFrame(
        counts.T, index=adata.var_names, columns=adata.obs_names,
    )
    p_counts = os.path.join(DATA_DIR, "pancreas_counts.csv")
    df_counts.to_csv(p_counts)
    print(f"   Counts (genes x cells): {df_counts.shape} -> {p_counts}")

    p_meta = os.path.join(DATA_DIR, "pancreas_metadata.csv")
    adata.obs[["batch", "celltype"]].to_csv(p_meta)
    print(f"   Metadata: {adata.obs.shape[0]} rows -> {p_meta}")


def main() -> sc.AnnData:
    print("=" * 60)
    print("  STEP 1: DATA PREPARATION (scIB human pancreas benchmark)")
    print("=" * 60)

    raw_path = os.path.join(DATA_DIR, RAW_FILENAME)
    download_raw_dataset(raw_path)
    adata_raw = load_raw_dataset(raw_path)

    adata = preprocess(adata_raw)

    out_path = os.path.join(DATA_DIR, "pancreas_preprocessed.h5ad")
    adata.write(out_path)
    print(f"\n[OK] Preprocessed AnnData saved -> {out_path}")
    print(f"     Shape : {adata.shape}")
    print(f"     obsm  : {list(adata.obsm.keys())}")
    print(f"     layers: {list(adata.layers.keys())}")

    export_for_seurat(adata)
    return adata


if __name__ == "__main__":
    main()
