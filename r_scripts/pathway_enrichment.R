#!/usr/bin/env Rscript
# pathway_enrichment.R
# KEGG and GO enrichment using clusterProfiler.
# Input (JSON arg): protein_list, dea_result_path, kegg_org, orgdb,
#                   pval_cutoff, output_dir
# Output (JSON to stdout): enrichment results and paths

suppressPackageStartupMessages({
  library(jsonlite)
  library(clusterProfiler)
  library(dplyr)
})

`%||%` <- function(a, b) if (!is.null(a)) a else b

args <- commandArgs(trailingOnly = TRUE)
params <- fromJSON(args[1])

protein_list  <- params$protein_list          # Gene symbols or UniProt IDs
dea_path      <- params$dea_result_path
kegg_org      <- params$kegg_org %||% "hsa"   # e.g. "hsa" for human
orgdb         <- params$orgdb %||% "org.Hs.eg.db"
pval_cutoff   <- as.numeric(params$pval_cutoff %||% 0.05)
output_dir    <- params$output_dir %||% "outputs"

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Load organism annotation DB dynamically
library(orgdb, character.only = TRUE)
org_db <- get(orgdb)

# ── Map gene symbols to Entrez IDs ────────────────────────────────────────────
entrez_ids <- tryCatch(
  bitr(protein_list, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org_db),
  error = function(e) {
    # Fallback: try UNIPROT → ENTREZID
    bitr(protein_list, fromType = "UNIPROT", toType = "ENTREZID", OrgDb = org_db)
  }
)

gene_list <- unique(entrez_ids$ENTREZID)

# ── KEGG enrichment ───────────────────────────────────────────────────────────
kegg_res <- tryCatch(
  enrichKEGG(gene = gene_list, organism = kegg_org, pvalueCutoff = pval_cutoff),
  error = function(e) NULL
)

kegg_df <- if (!is.null(kegg_res) && nrow(as.data.frame(kegg_res)) > 0) {
  as.data.frame(kegg_res) %>%
    select(Description, pvalue, p.adjust, Count, GeneRatio) %>%
    rename(pathway = Description, p_adjust = p.adjust, gene_count = Count) %>%
    arrange(p_adjust)
} else {
  data.frame()
}

# ── GO enrichment (Biological Process) ───────────────────────────────────────
go_res <- tryCatch(
  enrichGO(
    gene = gene_list, OrgDb = org_db, ont = "BP",
    pAdjustMethod = "BH", pvalueCutoff = pval_cutoff, readable = TRUE
  ),
  error = function(e) NULL
)

go_df <- if (!is.null(go_res) && nrow(as.data.frame(go_res)) > 0) {
  as.data.frame(go_res) %>%
    select(Description, pvalue, p.adjust, Count) %>%
    rename(pathway = Description, p_adjust = p.adjust, gene_count = Count) %>%
    arrange(p_adjust)
} else {
  data.frame()
}

# ── Save results ──────────────────────────────────────────────────────────────
enrichment_path <- file.path(output_dir, "enrichment_results.csv")

all_results <- bind_rows(
  if (nrow(kegg_df) > 0) mutate(kegg_df, source = "KEGG") else NULL,
  if (nrow(go_df)   > 0) mutate(go_df,   source = "GO_BP") else NULL
)

if (nrow(all_results) > 0) {
  write.csv(all_results, enrichment_path, row.names = FALSE)
}

# ── Top pathways for JSON output ──────────────────────────────────────────────
top_kegg <- if (nrow(kegg_df) > 0) head(kegg_df, 10) else data.frame()
top_pathways <- lapply(seq_len(nrow(top_kegg)), function(i) {
  list(
    pathway    = top_kegg$pathway[i],
    p_adjust   = signif(top_kegg$p_adjust[i], 4),
    gene_count = top_kegg$gene_count[i],
    source     = "KEGG"
  )
})

result <- list(
  enrichment_result_path = enrichment_path,
  top_pathways           = top_pathways,
  n_kegg_significant     = nrow(kegg_df[kegg_df$p_adjust < pval_cutoff, ]),
  n_go_significant       = nrow(go_df[go_df$p_adjust < pval_cutoff, ])
)

cat(toJSON(result, auto_unbox = TRUE))
