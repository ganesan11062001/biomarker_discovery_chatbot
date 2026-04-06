#!/usr/bin/env Rscript
# proteomics_qc.R
# Quality control for proteomics intensity matrices.
# Input (JSON arg): data_path, data_type, missing_threshold, cv_cutoff,
#                   outlier_sd, output_dir
# Output (JSON to stdout): QC metrics and paths

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
params <- fromJSON(args[1])

data_path        <- params$data_path
data_type        <- params$data_type %||% "generic"
missing_thresh   <- as.numeric(params$missing_threshold %||% 0.30)
cv_cutoff        <- if (!is.null(params$cv_cutoff) && !is.na(params$cv_cutoff))
                      as.numeric(params$cv_cutoff) else NULL
outlier_sd       <- as.numeric(params$outlier_sd %||% 3.0)
output_dir       <- params$output_dir %||% "outputs"

`%||%` <- function(a, b) if (!is.null(a)) a else b

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ── Load data ────────────────────────────────────────────────────────────────
# Rows = proteins, columns = samples
mat <- read.csv(data_path, row.names = 1, check.names = FALSE)
mat <- as.matrix(mat)

proteins_total  <- nrow(mat)
samples_total   <- ncol(mat)

# ── 1. Protein-level missing value filter ────────────────────────────────────
missing_frac <- rowMeans(is.na(mat))
mat <- mat[missing_frac <= missing_thresh, , drop = FALSE]

# ── 2. Sample-level missing value filter (>50% missing → drop sample) ────────
sample_missing <- colMeans(is.na(mat))
mat <- mat[, sample_missing <= 0.50, drop = FALSE]
samples_retained <- ncol(mat)

# ── 3. CV cutoff (optional) ───────────────────────────────────────────────────
cv_removed <- 0
if (!is.null(cv_cutoff)) {
  row_cv <- apply(mat, 1, function(x) {
    x <- x[!is.na(x)]
    if (length(x) < 2) return(Inf)
    sd(x) / abs(mean(x)) * 100
  })
  mat <- mat[row_cv <= cv_cutoff, , drop = FALSE]
  cv_removed <- proteins_total - nrow(mat)
}

proteins_retained <- nrow(mat)

# ── 4. Outlier sample detection (PCA distance) ───────────────────────────────
outliers_removed <- 0
mat_complete <- mat[complete.cases(mat), , drop = FALSE]
if (nrow(mat_complete) >= 5 && ncol(mat_complete) >= 3) {
  pca <- prcomp(t(mat_complete), scale. = TRUE)
  pc_scores <- pca$x[, 1:min(2, ncol(pca$x)), drop = FALSE]
  center <- colMeans(pc_scores)
  dists <- apply(pc_scores, 1, function(x) sqrt(sum((x - center)^2)))
  thresh <- mean(dists) + outlier_sd * sd(dists)
  outlier_samples <- names(dists[dists > thresh])
  if (length(outlier_samples) > 0) {
    mat <- mat[, !colnames(mat) %in% outlier_samples, drop = FALSE]
    outliers_removed <- length(outlier_samples)
    samples_retained <- ncol(mat)
  }
}

# ── Save filtered data ────────────────────────────────────────────────────────
filtered_path <- file.path(output_dir, paste0("qc_filtered_", basename(data_path)))
write.csv(mat, filtered_path)

# ── QC report ─────────────────────────────────────────────────────────────────
qc_report <- list(
  proteins_total    = proteins_total,
  proteins_retained = proteins_retained,
  samples_total     = samples_total,
  samples_retained  = samples_retained,
  outliers_removed  = outliers_removed,
  missing_threshold = missing_thresh,
  cv_cutoff         = if (!is.null(cv_cutoff)) cv_cutoff else "not applied"
)
report_path <- file.path(output_dir, "qc_report.json")
writeLines(toJSON(qc_report, auto_unbox = TRUE), report_path)

# ── Return result ─────────────────────────────────────────────────────────────
qc_passed <- (proteins_retained / proteins_total) >= 0.50

result <- list(
  qc_passed          = qc_passed,
  filtered_data_path = filtered_path,
  qc_report_path     = report_path,
  proteins_total     = proteins_total,
  proteins_retained  = proteins_retained,
  samples_retained   = samples_retained,
  outliers_removed   = outliers_removed,
  missing_threshold  = missing_thresh,
  cv_cutoff          = if (!is.null(cv_cutoff)) cv_cutoff else NA
)

cat(toJSON(result, auto_unbox = TRUE))
