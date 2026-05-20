# Seurat v5 anchor-based CCA integration on the scIB human pancreas benchmark.
# Reads raw counts (HVGs) + metadata exported by 01_data_preparation.py,
# performs CCA anchor integration, and saves the integrated PCA embedding for
# downstream evaluation in Python.
#
# This script is normally invoked through scripts/03_run_seurat.py, which
# auto-locates a working standalone R 4.x installation with Seurat 5.x.
# Tested with R 4.6.0 + Seurat 5.5.0.

options(future.globals.maxSize = 4 * 1024^3)
suppressMessages({
  library(Seurat)
})
set.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
project_dir <- normalizePath(file.path(getwd()))
data_dir    <- file.path(project_dir, "data")
out_dir     <- file.path(project_dir, "results", "embeddings")
metrics_dir <- file.path(project_dir, "results", "metrics")
dir.create(out_dir,     showWarnings = FALSE, recursive = TRUE)
dir.create(metrics_dir, showWarnings = FALSE, recursive = TRUE)

counts_file <- file.path(data_dir, "pancreas_counts.csv")
meta_file   <- file.path(data_dir, "pancreas_metadata.csv")
out_file    <- file.path(out_dir,  "seurat_embedding.csv")
runtime_log <- file.path(metrics_dir, "runtimes.csv")

cat("\n[Seurat ", as.character(packageVersion("Seurat")), "] CCA Integration\n", sep = "")
cat("  counts: ", counts_file, "\n")
cat("  meta  : ", meta_file,   "\n")

t0 <- Sys.time()

# ── Load data ─────────────────────────────────────────────────────────────────
counts <- read.csv(counts_file, row.names = 1, check.names = FALSE)
meta   <- read.csv(meta_file,   row.names = 1)
counts <- as.matrix(counts)
storage.mode(counts) <- "integer"
cat("  Matrix: ", dim(counts)[1], "genes x", dim(counts)[2], "cells\n")

stopifnot(all(colnames(counts) == rownames(meta)))

# ── Build per-batch Seurat objects ────────────────────────────────────────────
batches <- as.character(unique(meta$batch))
sobj_list <- lapply(batches, function(b) {
  cells <- rownames(meta)[meta$batch == b]
  s <- CreateSeuratObject(counts = counts[, cells], meta.data = meta[cells, ])
  s <- NormalizeData(s,  verbose = FALSE)
  s <- FindVariableFeatures(s, selection.method = "vst",
                             nfeatures = 2000, verbose = FALSE)
  s
})
names(sobj_list) <- batches

# ── Anchor-based integration (CCA) ───────────────────────────────────────────
features <- SelectIntegrationFeatures(sobj_list, nfeatures = 2000, verbose = FALSE)
cat("  Finding anchors (CCA, 30 dims)…\n")
anchors  <- FindIntegrationAnchors(
  object.list      = sobj_list,
  anchor.features  = features,
  dims             = 1:30,
  reduction        = "cca",
  verbose          = FALSE,
)
cat("  Integrating data…\n")
integrated <- IntegrateData(anchorset = anchors, dims = 1:30, verbose = FALSE)

DefaultAssay(integrated) <- "integrated"
integrated <- ScaleData(integrated, verbose = FALSE)
integrated <- RunPCA(integrated, npcs = 50, verbose = FALSE)

# ── Export embeddings in the original cell order ──────────────────────────────
emb <- Embeddings(integrated, "pca")
emb <- emb[rownames(meta), , drop = FALSE]
write.csv(emb, file = out_file, quote = FALSE)
cat("  Saved embedding (", dim(emb)[1], "x", dim(emb)[2], ") -> ", out_file, "\n", sep = "")

elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("  Seurat CCA total runtime: %.1f s\n", elapsed))

# ── Append to runtime log (compatible with Python utils.timer) ───────────────
new_row <- data.frame(method = "Seurat CCA", runtime_seconds = elapsed,
                       stringsAsFactors = FALSE)
if (file.exists(runtime_log)) {
  existing <- read.csv(runtime_log, stringsAsFactors = FALSE)
  if ("Seurat CCA" %in% existing$method) {
    existing$runtime_seconds[existing$method == "Seurat CCA"] <- elapsed
    write.csv(existing, runtime_log, row.names = FALSE, quote = FALSE)
  } else {
    write.csv(rbind(existing, new_row), runtime_log, row.names = FALSE, quote = FALSE)
  }
} else {
  write.csv(new_row, runtime_log, row.names = FALSE, quote = FALSE)
}

cat("\n[OK] Seurat CCA integration complete.\n")
