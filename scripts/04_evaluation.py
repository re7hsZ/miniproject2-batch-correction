"""
Evaluate every batch-corrected embedding with a panel of metrics.

  Batch removal       Silhouette_batch (1 - (s + 1) / 2)   higher = better
                      kBET acceptance rate                  higher = better
                      Graph connectivity                    higher = better

  Bio-conservation    Silhouette_celltype ((s + 1) / 2)    higher = better
                      ARI(Leiden vs ground truth)           higher = better
                      NMI(Leiden vs ground truth)           higher = better

Aggregated scores follow the scIB convention (Luecken et al., 2022):
  Batch       = mean(Silhouette_batch, kBET, Graph_connectivity)
  Bio         = mean(Silhouette_celltype, ARI, NMI)
  Overall     = 0.4 * Batch + 0.6 * Bio
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import chisquare
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import (
    DATA_DIR, METRICS_DIR, load_embedding, load_preprocessed,
)

SEED = 42


def silhouette_batch(emb: np.ndarray, batch: np.ndarray) -> float:
    s = silhouette_score(
        emb, batch,
        sample_size=min(5000, len(batch)),
        random_state=SEED,
    )
    return float(1 - (s + 1) / 2)


def silhouette_celltype(emb: np.ndarray, ct: np.ndarray) -> float:
    s = silhouette_score(
        emb, ct,
        sample_size=min(5000, len(ct)),
        random_state=SEED,
    )
    return float((s + 1) / 2)


def kbet(emb: np.ndarray, batch: np.ndarray,
         k: int = 25, n_samples: int = 500) -> float:
    rng = np.random.default_rng(SEED)
    global_freq = pd.Series(batch).value_counts(normalize=True).sort_index()
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)

    idx = rng.choice(len(batch), min(n_samples, len(batch)), replace=False)
    accepted = 0
    expected = global_freq.values * k
    valid = expected > 0
    for i in idx:
        _, ne = nn.kneighbors(emb[i:i + 1])
        local = batch[ne[0, 1:]]
        lf = (pd.Series(local).value_counts(normalize=True)
                .reindex(global_freq.index, fill_value=0).values)
        if valid.sum() > 1:
            _, p = chisquare(lf[valid] * k, f_exp=expected[valid])
            if p > 0.05:
                accepted += 1
    return accepted / len(idx)


def graph_connectivity(emb: np.ndarray, celltypes: np.ndarray,
                       k: int = 15) -> float:
    """
    Per-celltype graph connectivity (Luecken et al., 2022): the average
    fraction of cells of each celltype that fall in the largest connected
    component of its k-NN sub-graph.
    """
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    _, indices = nn.kneighbors(emb)
    n = emb.shape[0]
    rows = np.repeat(np.arange(n), k)
    cols = indices[:, 1:].ravel()
    G = sp.csr_matrix((np.ones_like(cols), (rows, cols)), shape=(n, n))
    G = G + G.T

    scores = []
    for ct in np.unique(celltypes):
        mask = np.where(celltypes == ct)[0]
        if mask.size < 5:
            continue
        sub = G[mask][:, mask]
        n_comp, labels = sp.csgraph.connected_components(sub, directed=False)
        if n_comp == 0:
            continue
        biggest = np.bincount(labels).max()
        scores.append(biggest / mask.size)
    return float(np.mean(scores))


def leiden_labels(adata: sc.AnnData, emb: np.ndarray,
                  resolution: float = 0.8) -> np.ndarray:
    adata.obsm["_eval_emb"] = emb
    sc.pp.neighbors(adata, use_rep="_eval_emb", n_neighbors=15,
                     random_state=SEED)
    sc.tl.leiden(adata, resolution=resolution, random_state=SEED,
                  key_added="_eval_leiden", flavor="igraph",
                  n_iterations=2, directed=False)
    return adata.obs["_eval_leiden"].values


def evaluate(adata: sc.AnnData, emb: np.ndarray) -> Dict[str, float]:
    batch = adata.obs["batch"].astype(str).values
    ct    = adata.obs["celltype"].astype(str).values

    m: Dict[str, float] = {
        "Silhouette_batch":    silhouette_batch(emb, batch),
        "kBET":                kbet(emb, batch),
        "Graph_connectivity":  graph_connectivity(emb, ct),
        "Silhouette_celltype": silhouette_celltype(emb, ct),
    }
    pred = leiden_labels(adata, emb)
    m["ARI"] = float(adjusted_rand_score(ct, pred))
    m["NMI"] = float(normalized_mutual_info_score(ct, pred))
    m["Batch_score"] = float(np.mean([
        m["Silhouette_batch"], m["kBET"], m["Graph_connectivity"],
    ]))
    m["Bio_score"]   = float(np.mean([
        m["Silhouette_celltype"], m["ARI"], m["NMI"],
    ]))
    m["Overall"]     = 0.4 * m["Batch_score"] + 0.6 * m["Bio_score"]
    return m


def get_embedding(adata: sc.AnnData, name: str) -> Tuple[str, np.ndarray]:
    """Resolve the AnnData / on-disk embedding for a given method name."""
    key_map = {
        "Uncorrected": "X_pca",
        "Harmony":     "X_harmony",
        "scVI":        "X_scvi",
        "Scanorama":   "X_scanorama",
        "ComBat":      "X_combat",
        "Seurat CCA":  "X_seurat",
    }
    obsm_key = key_map[name]
    if obsm_key in adata.obsm:
        return obsm_key, np.asarray(adata.obsm[obsm_key])

    file_map = {
        "Harmony":    "harmony",
        "scVI":       "scvi",
        "Scanorama":  "scanorama",
        "ComBat":     "combat",
        "Seurat CCA": "seurat",
    }
    arr = load_embedding(file_map[name])
    adata.obsm[obsm_key] = arr.astype(np.float32)
    return obsm_key, arr


def main() -> None:
    print("=" * 60)
    print("  STEP 4: BENCHMARK EVALUATION")
    print("=" * 60)

    adata_path = os.path.join(DATA_DIR, "pancreas_with_embeddings.h5ad")
    if os.path.exists(adata_path):
        print(f"Loading combined embeddings: {adata_path}")
        adata = sc.read_h5ad(adata_path)
    else:
        adata = load_preprocessed()

    methods = [
        "Uncorrected", "Seurat CCA", "Harmony", "scVI", "Scanorama", "ComBat",
    ]
    rows: Dict[str, Dict[str, float]] = {}
    for name in methods:
        try:
            print(f"\n-> {name}")
            _, emb = get_embedding(adata, name)
            rows[name] = evaluate(adata, emb)
            for k, v in rows[name].items():
                print(f"    {k:<22s} {v:.4f}")
        except Exception as e:
            print(f"  WARN: {name} skipped ({e}).")

    df = pd.DataFrame(rows).T
    df.index.name = "Method"
    cols = [
        "Silhouette_batch", "kBET", "Graph_connectivity",
        "Silhouette_celltype", "ARI", "NMI",
        "Batch_score", "Bio_score", "Overall",
    ]
    df = df[[c for c in cols if c in df.columns]]
    out_csv = os.path.join(METRICS_DIR, "benchmark_summary.csv")
    df.to_csv(out_csv)

    print("\n" + "=" * 60)
    print("  BENCHMARK SUMMARY (sorted by Overall)")
    print("=" * 60)
    print(df.sort_values("Overall", ascending=False).round(4).to_string())
    print(f"\nSaved -> {out_csv}")


if __name__ == "__main__":
    main()
