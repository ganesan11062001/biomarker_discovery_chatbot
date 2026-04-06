options(repos = c(CRAN = "https://cloud.r-project.org"))

message("Installing CRAN packages...")
cran_pkgs <- c("jsonlite", "dplyr", "ggplot2", "ggrepel", "pheatmap", "stringr", "tibble")
install.packages(cran_pkgs, quiet = TRUE)

message("Installing BiocManager...")
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

message("Installing Bioconductor packages (this takes 10-20 min)...")
bioc_pkgs <- c("limma", "clusterProfiler", "org.Hs.eg.db", "org.Mm.eg.db", "OlinkAnalyze", "DEP", "MSstats")
BiocManager::install(bioc_pkgs, ask = FALSE, update = FALSE)

message("All packages installed.")
