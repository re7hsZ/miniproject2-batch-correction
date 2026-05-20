# Benchmarking Batch-Effect Correction Methods for scRNA-seq Integration

CMML3 Miniproject 2 — code repository for reproducing the benchmark.

**Repository:** https://github.com/re7hsZ/miniproject2-batch-correction

The pipeline benchmarks **five** widely used batch-effect correction tools on the
**scIB human pancreas integration benchmark** (Luecken et al., 2022, *Nature Methods*):

| Method | Correction level | Architecture | Implementation |
|--------|------------------|--------------|----------------|
| **ComBat** | gene-expression matrix | parametric empirical Bayes | `scanpy.pp.combat` |
| **Harmony** | PCA latent space | iterative soft-clustering | `harmony-pytorch` |
| **scVI** | VAE latent space | conditional VAE (deep learning) | `scvi-tools` |
| **Scanorama** | k-NN graph + SVD | mutual nearest neighbours + panorama | `scanorama` |
| **Seurat CCA** | CCA latent space | canonical correlation anchors | Seurat 5 (R) |

Each embedding is evaluated with the **scIB-style metric panel** (batch silhouette,
kBET, graph connectivity, cell-type silhouette, ARI, NMI) and aggregated as
`0.4 * Batch + 0.6 * Bio`.

---

## Project layout

This repository tracks **only the source files needed to run the pipeline**.
The `data/` and `results/` folders are created automatically when the
scripts are executed and are excluded from version control.

```
miniproject2/
|-- scripts/
|   |-- 01_data_preparation.py        # download scIB pancreas + preprocessing
|   |-- 02_batch_correction_python.py # Harmony / scVI / Scanorama / ComBat
|   |-- 03_run_seurat.py              # Python entry point for the Seurat step
|   |-- 03_seurat_integration.R       # Seurat CCA (invoked by 03_run_seurat.py)
|   |-- 04_evaluation.py              # six-metric benchmark
|   |-- 05_visualization.py           # main and supplementary figures
|   `-- utils.py                      # shared paths and helpers
|-- environment.yml
|-- requirements.txt
|-- README.md
`-- .gitignore

# Folders created at runtime (not tracked in the repo):
# data/        downloaded benchmark h5ad + preprocessed AnnData and CSVs for R
# results/
#   |-- embeddings/   {harmony,scvi,scanorama,combat,seurat}_embedding.csv
#   |-- metrics/      benchmark_summary.csv, runtimes.csv
#   `-- figures/      Figure1_*, Figure2_*, FigureS1_*, ..., FigureS5_*
```

---

## Quick start (full pipeline, about 20 min on a laptop)

```bash
# 1) Create the environment (conda is the easiest way to get R + Seurat)
conda env create -f environment.yml
conda activate scrna_batch

# OR, Python side only (install R + Seurat 5.x separately)
pip install -r requirements.txt

# 2) Run the pipeline
python scripts/01_data_preparation.py        # download + preprocess  (~2 min first run)
python scripts/02_batch_correction_python.py # Harmony/scVI/Scanorama/ComBat (~3 min)
python scripts/03_run_seurat.py              # Seurat CCA via R 4.x   (~9 min)
python scripts/04_evaluation.py              # benchmark metrics      (~2 min)
python scripts/05_visualization.py           # main + supplementary figures (~2 min)
```

Step 1 downloads `human_pancreas_norm_complexBatch.h5ad` (~316 MB) from Zenodo
(the scIB-cleaned pancreas benchmark mirror) on first run.

After the pipeline finishes you will find:

- `results/metrics/benchmark_summary.csv` - the master results table
- `results/metrics/runtimes.csv`          - per-method wall-clock times
- `results/figures/Figure1_UMAP_comparison.png/.pdf`     - main Figure 1
- `results/figures/Figure2_metrics_comparison.png/.pdf`  - main Figure 2
- `results/figures/FigureS1_QC.png`                      - Supp. Fig. S1
- `results/figures/FigureS2_PCA_variance.png`            - Supp. Fig. S2
- `results/figures/FigureS3_silhouette.png`              - Supp. Fig. S3
- `results/figures/FigureS4_runtime.png`                 - Supp. Fig. S4
- `results/figures/FigureS5_celltype_mixing.png`         - Supp. Fig. S5

---

## Notes for Windows users

The conda-managed R distribution on some Windows installations triggers an
`mingw-w64` pseudo-relocation crash when loading large R packages (Seurat, sp).
`scripts/03_run_seurat.py` therefore prefers a **standalone R installation**:

1. `C:\Program Files\R\R-4.6.0\bin\Rscript.exe`
2. `C:\Program Files\R\R-4.5.3\bin\Rscript.exe`
3. `C:\Program Files\R\R-4.4.0\bin\Rscript.exe`
4. fallback to whatever `Rscript` is on `PATH`

Make sure that Seurat 5.x is installed in that R: `install.packages("Seurat")`.

---

## Configuration

All random seeds are pinned to **42**.

**Dataset:** scIB human pancreas benchmark (16 382 cells, 19 093 genes, 9 sequencing
technologies, 14 annotated cell types). Source:
[Zenodo record 18336458](https://zenodo.org/records/18336458)
(`human_pancreas_norm_complexBatch.h5ad`).

---

## Acknowledgements

- The benchmark dataset follows the **scIB** pancreas integration task (Luecken et al.,
  2022, *Nature Methods*).
- The metric panel and aggregation follow the same scIB benchmark framework.
- All correction algorithms are used unchanged from their original publications.
