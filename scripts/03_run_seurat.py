"""
Python entry point for the Seurat (R) integration step of the pipeline.

This wrapper locates a working standalone R installation and delegates to
`scripts/03_seurat_integration.R`, which reads the CSV files exported by
`01_data_preparation.py` and writes the integrated PCA embedding to
`results/embeddings/seurat_embedding.csv`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import PROJECT_DIR


R_CANDIDATES = [
    r"C:\Program Files\R\R-4.6.0\bin\Rscript.exe",
    r"C:\Program Files\R\R-4.5.3\bin\Rscript.exe",
    r"C:\Program Files\R\R-4.4.0\bin\Rscript.exe",
]


def find_rscript() -> str:
    for path in R_CANDIDATES:
        if os.path.exists(path):
            return path
    found = shutil.which("Rscript")
    if found is None:
        raise FileNotFoundError(
            "No usable Rscript was found. Install R 4.x and Seurat 5.x, or "
            "add the conda-managed R to PATH."
        )
    return found


def main() -> None:
    rscript = find_rscript()
    r_file  = os.path.join(PROJECT_DIR, "scripts", "03_seurat_integration.R")

    print(f"Rscript : {rscript}")
    print(f"R file  : {r_file}")
    print(f"cwd     : {PROJECT_DIR}")

    result = subprocess.run(
        [rscript, r_file],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.stdout:
        print("\n----- R STDOUT -----")
        print(result.stdout)
    if result.stderr:
        print("\n----- R STDERR -----")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Seurat R script failed with exit code {result.returncode}."
        )

    print("\n[OK] Seurat CCA integration complete.")


if __name__ == "__main__":
    main()
