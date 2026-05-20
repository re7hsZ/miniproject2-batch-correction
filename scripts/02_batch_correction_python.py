"""
Run the four Python batch-correction methods and persist their embeddings:

  1. Harmony     - iterative soft-clustering in PCA space.
  2. scVI        - conditional VAE on raw counts.
  3. Scanorama   - MNN + SVD panoramic stitching in HVG space.
  4. ComBat      - empirical-Bayes correction on log-normalised expression.

Outputs:
  results/embeddings/{harmony,scvi,scanorama,combat}_embedding.csv
  data/pancreas_with_embeddings.h5ad
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import scanpy as sc

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import (
    DATA_DIR, load_preprocessed, save_embedding, set_plot_style, timer,
)

set_plot_style()
SEED = 42


@timer("Harmony")
def run_harmony(adata: sc.AnnData) -> sc.AnnData:
    """Harmony (Korsunsky et al., 2019, Nature Methods)."""
    from harmony import harmonize

    print("  Running harmony-pytorch on X_pca ...")
    Z = harmonize(
        adata.obsm["X_pca"], adata.obs,
        batch_key="batch",
        max_iter_harmony=20,
        random_state=SEED,
    )
    adata.obsm["X_harmony"] = np.asarray(Z, dtype=np.float32)
    save_embedding(adata, "X_harmony", "harmony")
    print(f"  X_harmony shape: {adata.obsm['X_harmony'].shape}")
    return adata


@timer("scVI")
def run_scvi(adata: sc.AnnData) -> sc.AnnData:
    """scVI (Lopez et al., 2018, Nature Methods)."""
    import scvi
    scvi.settings.seed = SEED

    a = adata.copy()
    a.X = a.layers["counts"].copy()

    scvi.model.SCVI.setup_anndata(a, batch_key="batch")
    model = scvi.model.SCVI(
        a, n_layers=2, n_hidden=128, n_latent=30,
        dropout_rate=0.1, gene_likelihood="nb",
    )
    model.train(
        max_epochs=120,
        early_stopping=True, early_stopping_patience=10,
        batch_size=256,
        plan_kwargs={"lr": 1e-3},
        accelerator="cpu",
    )
    adata.obsm["X_scvi"] = model.get_latent_representation().astype(np.float32)
    save_embedding(adata, "X_scvi", "scvi")
    print(f"  X_scvi shape: {adata.obsm['X_scvi'].shape}")
    return adata


@timer("Scanorama")
def run_scanorama(adata: sc.AnnData) -> sc.AnnData:
    """Scanorama (Hie et al., 2019, Nature Biotechnology)."""
    import scanorama

    batches = list(adata.obs["batch"].cat.categories)
    splits  = [adata[adata.obs["batch"] == b].copy() for b in batches]

    scanorama.integrate_scanpy(splits, dimred=50, seed=SEED)

    Z = np.zeros((adata.n_obs, 50), dtype=np.float32)
    for sub in splits:
        Z[adata.obs_names.get_indexer(sub.obs_names)] = sub.obsm["X_scanorama"]

    adata.obsm["X_scanorama"] = Z
    save_embedding(adata, "X_scanorama", "scanorama")
    print(f"  X_scanorama shape: {adata.obsm['X_scanorama'].shape}")
    return adata


@timer("ComBat")
def run_combat(adata: sc.AnnData) -> sc.AnnData:
    """ComBat empirical-Bayes correction (Johnson, Li and Rabinovic, 2007)."""
    a = adata.copy()
    a.X = a.layers["lognorm"].copy()
    sc.pp.combat(a, key="batch")

    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=50, svd_solver="arpack", random_state=SEED)
    adata.obsm["X_combat"] = a.obsm["X_pca"].astype(np.float32)
    save_embedding(adata, "X_combat", "combat")
    print(f"  X_combat shape: {adata.obsm['X_combat'].shape}")
    return adata


def main() -> sc.AnnData:
    print("=" * 60)
    print("  STEP 2: PYTHON BATCH-CORRECTION METHODS")
    print("=" * 60)

    adata = load_preprocessed()

    adata = run_harmony(adata)
    adata = run_scvi(adata)
    adata = run_scanorama(adata)
    adata = run_combat(adata)

    for k in list(adata.uns.keys()):
        if "neighbors" in k.lower():
            del adata.uns[k]
    for k in list(adata.obsp.keys()):
        del adata.obsp[k]

    out_path = os.path.join(DATA_DIR, "pancreas_with_embeddings.h5ad")
    adata.write(out_path)
    print(f"\n[OK] Saved AnnData with embeddings -> {out_path}")
    print(f"     obsm: {sorted(adata.obsm.keys())}")
    return adata


if __name__ == "__main__":
    main()
