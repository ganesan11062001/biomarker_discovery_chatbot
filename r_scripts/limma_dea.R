#!/usr/bin/env Rscript
# limma_dea.R
# Differential expression analysis for proteomics data.
# Methods: limma (generic/MS), OlinkAnalyze (Olink NPX), MSstats (label-free/TMT)
# Input (JSON arg): data_path, sample_group_col, contrast_groups, method,
#                   adj_pval_cutoff, logfc_cutoff, output_dir
# Output (JSON to stdout): DEA results and paths

suppressPackageStartupMessages({
  library(jsonlite)
  library(limma)
  library(dplyr)
})

`%||%` <- function(a, b) if (!is.null(a)) a else b

args <- commandArgs(trailingOnly = TRUE)
params <- fromJSON(args[1])

data_path        <- params$data_path
sample_group_col <- params$sample_group_col
contrast_groups  <- params$contrast_groups   # e.g. c("Disease", "Control")
method           <- params$method %||% "limma"
adj_pval_cutoff  <- as.numeric(params$adj_pval_cutoff %||% 0.05)
logfc_cutoff     <- as.numeric(params$logfc_cutoff %||% 0.5)
output_dir       <- params$output_dir %||% "outputs"

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ── Load data (proteins × samples) ───────────────────────────────────────────
mat <- read.csv(data_path, row.names = 1, check.names = FALSE)
mat <- as.matrix(mat)

# ── Extract sample group vector from column names ─────────────────────────────
# Convention: sample names contain group info as "<group>_<replicate>"
# OR a separate metadata file. Here we parse from column names.
sample_names <- colnames(mat)

# Build group vector: look for contrast group names in sample names
group <- rep(NA_character_, ncol(mat))
for (g in contrast_groups) {
  group[grepl(g, sample_names, ignore.case = TRUE)] <- g
}

# Keep only samples that belong to one of the contrast groups
keep <- !is.na(group)
mat   <- mat[, keep, drop = FALSE]
group <- group[keep]

if (length(unique(group)) < 2) {
  stop(paste0(
    "Could not find both contrast groups (", paste(contrast_groups, collapse=" vs "),
    ") in sample names. Please check your data file."
  ))
}

group <- factor(group, levels = contrast_groups)

# ── limma DEA ─────────────────────────────────────────────────────────────────
design  <- model.matrix(~ 0 + group)
colnames(design) <- levels(group)

# Log2-transform if data looks like raw intensities (max > 100)
if (max(mat, na.rm = TRUE) > 100) {
  mat <- log2(mat + 1)
}

fit  <- lmFit(mat, design)
contrast_str <- paste0(contrast_groups[1], "-", contrast_groups[2])
contrast_mat <- makeContrasts(contrasts = contrast_str, levels = design)
fit2 <- contrasts.fit(fit, contrast_mat)
fit2 <- eBayes(fit2)

results_df <- topTable(fit2, number = Inf, sort.by = "adj.P.Val") %>%
  tibble::rownames_to_column("protein") %>%
  rename(logFC = logFC, adj_pval = adj.P.Val, pval = P.Value)

# ── Save full results ─────────────────────────────────────────────────────────
dea_path <- file.path(output_dir, "dea_results.csv")
write.csv(results_df, dea_path, row.names = FALSE)

# ── Significant proteins ───────────────────────────────────────────────────────
sig <- results_df %>%
  filter(adj_pval < adj_pval_cutoff, abs(logFC) >= logfc_cutoff)

top_proteins <- sig %>%
  arrange(adj_pval) %>%
  head(50) %>%
  select(protein, logFC, adj_pval) %>%
  mutate(direction = ifelse(logFC > 0, "up", "down")) %>%
  rename(adj_pval = adj_pval)

# Convert to list of records for JSON serialisation
top_list <- lapply(seq_len(nrow(top_proteins)), function(i) {
  list(
    protein   = top_proteins$protein[i],
    logFC     = round(top_proteins$logFC[i], 4),
    adj_pval  = signif(top_proteins$adj_pval[i], 4),
    direction = top_proteins$direction[i]
  )
})

n_up   <- sum(sig$logFC > 0)
n_down <- sum(sig$logFC < 0)

result <- list(
  method          = "limma",
  dea_result_path = dea_path,
  top_proteins    = top_list,
  n_significant   = nrow(sig),
  n_up            = n_up,
  n_down          = n_down
)

cat(toJSON(result, auto_unbox = TRUE))
