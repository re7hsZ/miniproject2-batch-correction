"""Shared paths, plotting style, timing decorator and small I/O helpers."""

import os
import time
from functools import wraps

import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt


# Project paths
PROJECT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR    = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR    = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR    = os.path.join(RESULTS_DIR, "metrics")
EMBEDDINGS_DIR = os.path.join(RESULTS_DIR, "embeddings")

for d in [DATA_DIR, FIGURES_DIR, METRICS_DIR, EMBEDDINGS_DIR]:
    os.makedirs(d, exist_ok=True)


def set_plot_style() -> None:
    """Apply a publication-quality matplotlib + scanpy plotting style."""
    plt.rcParams.update({
        "figure.dpi":     150,
        "savefig.dpi":    300,
        "savefig.bbox":   "tight",
        "font.size":      10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.figsize": (8, 6),
        "font.family":    "sans-serif",
    })
    sc.settings.set_figure_params(
        dpi=150, dpi_save=300, frameon=False, fontsize=10,
    )


def timer(method_name: str):
    """Time a function and append the runtime to results/metrics/runtimes.csv."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"\n{'=' * 60}")
            print(f"  Running: {method_name}")
            print(f"{'=' * 60}")
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            minutes, seconds = divmod(elapsed, 60)
            print(f"  [OK] {method_name} completed in "
                  f"{int(minutes)}m {seconds:.1f}s")
            print(f"{'=' * 60}\n")

            log = os.path.join(METRICS_DIR, "runtimes.csv")
            new_row = pd.DataFrame(
                [[method_name, elapsed]],
                columns=["method", "runtime_seconds"],
            )
            if os.path.exists(log):
                existing = pd.read_csv(log)
                if method_name in existing["method"].values:
                    existing.loc[
                        existing["method"] == method_name,
                        "runtime_seconds",
                    ] = elapsed
                    out_df = existing
                else:
                    out_df = pd.concat([existing, new_row], ignore_index=True)
            else:
                out_df = new_row
            out_df.to_csv(log, index=False)
            return result
        return wrapper
    return decorator


def load_preprocessed() -> sc.AnnData:
    """Load the preprocessed AnnData object produced by 01_data_preparation.py."""
    path = os.path.join(DATA_DIR, "pancreas_preprocessed.h5ad")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Preprocessed data not found at {path}. "
            "Run 01_data_preparation.py first."
        )
    print(f"Loading preprocessed data from {path}")
    adata = sc.read_h5ad(path)
    print(f"  Shape: {adata.shape}")
    print(f"  Batches: {adata.obs['batch'].value_counts().to_dict()}")
    print(f"  Cell types: {adata.obs['celltype'].nunique()} types")
    return adata


def save_embedding(adata: sc.AnnData, key: str, method_name: str) -> None:
    """Persist a corrected embedding to results/embeddings/{method}_embedding.csv."""
    emb = adata.obsm[key]
    df = pd.DataFrame(
        emb,
        index=adata.obs_names,
        columns=[f"{method_name}_{i + 1}" for i in range(emb.shape[1])],
    )
    path = os.path.join(EMBEDDINGS_DIR, f"{method_name}_embedding.csv")
    df.to_csv(path)
    print(f"  Saved embedding to {path} (shape: {emb.shape})")


def load_embedding(method_name: str):
    """Load a saved embedding from results/embeddings/{method}_embedding.csv."""
    path = os.path.join(EMBEDDINGS_DIR, f"{method_name}_embedding.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding not found: {path}")
    df = pd.read_csv(path, index_col=0)
    return df.values
