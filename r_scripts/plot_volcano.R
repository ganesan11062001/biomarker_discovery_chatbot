#!/usr/bin/env Rscript
# plot_volcano.R
# Generates volcano plot, heatmap, and (optionally) enrichment dot plot.
# Input (JSON arg): dea_result_path, enrichment_result_path, top_proteins,
#                   contrast_groups, disease_program, output_dir
# Output (JSON to stdout): list of generated plot paths

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(dplyr)
})

`%||%` <- function(a, b) if (!is.null(a)) a else b

args <- commandArgs(trailingOnly = TRUE)
params <- fromJSON(args[1])

dea_path         <- params$dea_result_path
enrichment_path  <- params$enrichment_result_path %||% ""
top_proteins     <- params$top_proteins %||% character(0)
contrast_groups  <- params$contrast_groups %||% c("Group1", "Group2")
disease_program  <- params$disease_program %||% ""
output_dir       <- params$output_dir %||% "outputs"

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

plot_paths <- character(0)

# ── Load DEA results ──────────────────────────────────────────────────────────
dea <- read.csv(dea_path)

# Ensure required columns exist
if (!all(c("logFC", "adj_pval") %in% colnames(dea))) {
  # Try alternative column names
  if ("adj.P.Val" %in% colnames(dea)) dea <- rename(dea, adj_pval = adj.P.Val)
  if ("P.Value"   %in% colnames(dea)) dea <- rename(dea, pval     = P.Value)
}

dea <- dea %>%
  mutate(
    neg_log10_p = -log10(adj_pval + 1e-300),
    significance = case_when(
      adj_pval < 0.05 & logFC >  0.5 ~ "Up",
      adj_pval < 0.05 & logFC < -0.5 ~ "Down",
      TRUE                            ~ "NS"
    )
  )

# ── Volcano plot ──────────────────────────────────────────────────────────────
label_df <- dea %>%
  filter(significance != "NS") %>%
  arrange(adj_pval) %>%
  head(15)

title_str <- if (length(contrast_groups) >= 2) {
  paste0(contrast_groups[1], " vs ", contrast_groups[2],
         if (nchar(disease_program) > 0) paste0("  (", disease_program, ")") else "")
} else "Differential Expression"

volcano <- ggplot(dea, aes(x = logFC, y = neg_log10_p, colour = significance)) +
  geom_point(alpha = 0.6, size = 1.5) +
  scale_colour_manual(values = c("Up" = "#C0392B", "Down" = "#2980B9", "NS" = "grey60")) +
  geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed", colour = "grey40") +
  geom_hline(yintercept = -log10(0.05),  linetype = "dashed", colour = "grey40") +
  ggrepel::geom_text_repel(
    data = label_df,
    aes(label = protein), size = 3, max.overlaps = 15, colour = "black"
  ) +
  labs(
    title    = title_str,
    x        = expression(log[2]~FC),
    y        = expression(-log[10]~"adj. p-value"),
    colour   = NULL
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "top")

volcano_path <- file.path(output_dir, "volcano_plot.png")
ggsave(volcano_path, plot = volcano, width = 7, height = 6, dpi = 150)
plot_paths <- c(plot_paths, volcano_path)

# ── Heatmap (top proteins) ────────────────────────────────────────────────────
# Only if pheatmap is available
if (requireNamespace("pheatmap", quietly = TRUE) && length(top_proteins) > 0) {
  # Re-load the processed (QC-filtered) data if accessible
  # Fall back to using DEA logFC values as a proxy heatmap
  heatmap_data <- dea %>%
    filter(protein %in% top_proteins) %>%
    select(protein, logFC) %>%
    tibble::column_to_rownames("protein") %>%
    as.matrix()

  if (nrow(heatmap_data) > 1) {
    heatmap_path <- file.path(output_dir, "heatmap.png")
    png(heatmap_path, width = 600, height = max(400, nrow(heatmap_data) * 20))
    pheatmap::pheatmap(
      heatmap_data,
      cluster_cols  = FALSE,
      color         = colorRampPalette(c("#2980B9", "white", "#C0392B"))(50),
      main          = "Top DEA Proteins – logFC",
      fontsize_row  = 8
    )
    dev.off()
    plot_paths <- c(plot_paths, heatmap_path)
  }
}

# ── Enrichment dot plot ───────────────────────────────────────────────────────
if (nchar(enrichment_path) > 0 && file.exists(enrichment_path)) {
  enrich_df <- read.csv(enrichment_path) %>%
    filter(source == "KEGG") %>%
    head(15) %>%
    mutate(pathway = stringr::str_wrap(pathway, 40))

  if (nrow(enrich_df) > 0) {
    dotplot <- ggplot(enrich_df, aes(x = -log10(p_adjust + 1e-300),
                                      y = reorder(pathway, -p_adjust),
                                      size = gene_count,
                                      colour = -log10(p_adjust + 1e-300))) +
      geom_point() +
      scale_colour_gradient(low = "#85C1E9", high = "#1A5276") +
      labs(title = "KEGG Pathway Enrichment",
           x = expression(-log[10]~"adj. p-value"), y = NULL,
           size = "Gene count") +
      theme_bw(base_size = 11) +
      theme(axis.text.y = element_text(size = 8))

    dot_path <- file.path(output_dir, "enrichment_dotplot.png")
    ggsave(dot_path, plot = dotplot, width = 8, height = 6, dpi = 150)
    plot_paths <- c(plot_paths, dot_path)
  }
}

# ── Biomarker ranking table ───────────────────────────────────────────────────
ranking <- dea %>%
  filter(significance != "NS") %>%
  arrange(adj_pval) %>%
  select(protein, logFC, adj_pval, significance)

ranking_path <- file.path(output_dir, "biomarker_ranking.csv")
write.csv(ranking, ranking_path, row.names = FALSE)

result <- list(
  plot_paths   = as.list(plot_paths),
  report_path  = ranking_path
)

cat(toJSON(result, auto_unbox = TRUE))
