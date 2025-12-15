# Libraries (suppress startup messages and warnings)
pkgs <- c("arrow", "dplyr", "tidyr", "RColorBrewer", "grid", "ComplexHeatmap", "circlize", "magrittr")
invisible(lapply(pkgs, function(p) {
  suppressPackageStartupMessages(
    suppressWarnings(
      library(p, character.only = TRUE, quietly = TRUE, warn.conflicts = FALSE)
    )
  )
}))

# Set plate to process: "redo" or "original"
plate_to_process <- "redo"

# Set output directory based on plate_to_process
if(plate_to_process == "redo") {
  output_dir <- file.path(".", "figures", "heatmaps", "redo_DMSO_plate")
} else if(plate_to_process == "original") {
  output_dir <- file.path(".", "figures", "heatmaps", "original_DMSO_plate")
} else {
  stop("Unknown plate_to_process value")
}

# Create directory if it doesn't exist
if(!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Build resolved base directory (one level up from notebook working dir)
base_dir <- normalizePath(file.path(getwd(), ".."))
sc_dir <- file.path(base_dir, "3.preprocessing_profiles", "data", "single_cell_profiles")

fname <- switch(
  plate_to_process,
  redo = "CARD-CelIns-CX7_251110170001_sc_feature_selected.parquet",
  original = "CARD-CelIns-CX7_251023130003_sc_feature_selected.parquet",
  stop("Unknown plate_to_process: ", plate_to_process)
)

data_path <- tryCatch(normalizePath(file.path(sc_dir, fname), mustWork = TRUE),
                      error = function(e) stop(sprintf("Could not find parquet %s in %s: %s", fname, sc_dir, e$message)))

# Read single-cell parquet
single_cell_data <- arrow::read_parquet(data_path)
print(sprintf("Loaded single-cell data from: %s", data_path))
dim(single_cell_data)
head(single_cell_data)

# Aggregate to well-level using median
aggregate_df <- single_cell_data %>%
  group_by(Metadata_Plate, Metadata_Well, Metadata_heart_number, Metadata_treatment, Metadata_cell_type) %>%
  summarize(across(where(is.numeric), ~median(.x, na.rm = TRUE)), .groups = "drop")

print(sprintf("Aggregated well-level data shape: %d x %d", nrow(aggregate_df), ncol(aggregate_df)))
head(aggregate_df)

# Loop over heart numbers
for (heart_num in unique(aggregate_df$Metadata_heart_number)) {
  
  heart_df <- aggregate_df %>% filter(Metadata_heart_number == heart_num)
  if (nrow(heart_df) == 0) {
    message(sprintf("No rows found for Heart #%s, skipping...", heart_num))
    next
  }
  
  # Loop over treatments
  for (treatment in unique(heart_df$Metadata_treatment)) {
    
    df <- heart_df %>% filter(Metadata_treatment == treatment)
    if (nrow(df) == 0) next
    
    # Separate metadata and feature columns
    metadata_cols <- names(df)[grepl('^Metadata_', names(df))]
    feature_cols <- setdiff(names(df), metadata_cols)
    
    # Use all features
    mat <- df[feature_cols]
    mat[!is.finite(as.matrix(mat))] <- 0
    mat_z <- scale(as.matrix(mat), center = TRUE, scale = TRUE)
    mat_z[is.na(mat_z)] <- 0
    
    # Row labels are just wells
    row_labels <- df$Metadata_Well
    rownames(mat_z) <- row_labels
    
    # Row annotation for wells
    row_ann_df <- data.frame(Well = as.character(df$Metadata_Well))
    rownames(row_ann_df) <- row_labels
    
    # Colors for wells
    wells <- unique(row_ann_df$Well)
    num_wells <- length(wells)
    palette_base <- colorRampPalette(RColorBrewer::brewer.pal(min(12, max(3, num_wells)), "Set3"))(num_wells)
    annotation_colors <- list(Well = setNames(palette_base, wells))
    
    row_ha <- rowAnnotation(
      Well = row_ann_df$Well,
      col = annotation_colors,
      show_annotation_name = TRUE
    )
    
    # Define correlation-based distance function
    dist_cor <- function(x) {
      as.dist(1 - cor(t(x), use = "pairwise.complete.obs"))
    }
    
    # Correlation-based clustering
    hc_rows <- hclust(dist_cor(mat_z), method = "average")
    hc_cols <- hclust(dist_cor(t(mat_z)), method = "average")
    
    # Heatmap
    ht <- Heatmap(
      mat_z,
      name = "Z-score",
      show_row_names = FALSE,
      show_column_names = FALSE,
      cluster_rows = hc_rows,
      cluster_columns = hc_cols,
      col = colorRamp2(
        breaks = c(min(mat_z), 0, max(mat_z)),
        colors = c("#ca0020", "white", "#0571b0")  # red → white → blue
      ),
      heatmap_legend_param = list(title = "Z-score"),
      column_title = sprintf(
        "Heart #%s: Well-level feature heatmap (n = %d)",
        heart_num, length(feature_cols)
      ),
      column_title_gp = gpar(fontsize = 14, fontface = "bold")
    )
    
    # Save to file in output_dir (sanitize treatment for filename)
    safe_treatment <- gsub("[^A-Za-z0-9_\\-]", "_", as.character(treatment))
    output_file <- file.path(
      output_dir,
      sprintf("heart_%s_%s_well_level_heatmap.png", heart_num, safe_treatment)
    )
    png(output_file, width = 2800, height = 2000, res = 400)
    draw(ht + row_ha, heatmap_legend_side = "right", annotation_legend_side = "right")
    dev.off()
    
    message(sprintf(
      "Saved heatmap for Heart #%s, Treatment '%s' (%d features) to %s",
      heart_num, treatment, length(feature_cols), output_file
    ))
  }
}

# Identify metadata vs feature columns
metadata_cols <- names(aggregate_df)[grepl('^Metadata_', names(aggregate_df))]
feature_cols <- setdiff(names(aggregate_df), metadata_cols)

# Use all features
mat <- as.data.frame(aggregate_df[feature_cols], stringsAsFactors = FALSE)
mat[!is.finite(as.matrix(mat))] <- 0

# Z-score each column (feature)
mat_z <- scale(as.matrix(mat), center = TRUE, scale = TRUE)
mat_z[is.na(mat_z)] <- 0

# Row labels
row_labels <- paste0("H", aggregate_df$Metadata_heart_number, "_", seq_len(nrow(aggregate_df)))
rownames(mat_z) <- row_labels

# Row annotation data.frame
row_ann_df <- data.frame(
  Metadata_heart_number = as.character(aggregate_df$Metadata_heart_number),
  Metadata_cell_type = as.character(aggregate_df$Metadata_cell_type),
  Metadata_treatment = as.character(aggregate_df$Metadata_treatment),
  stringsAsFactors = FALSE
)
rownames(row_ann_df) <- row_labels

# Helper for generating palettes
get_palette <- function(n, brewer_name, brewer_max) {
  if (n <= 0) return(character(0))
  if (n <= brewer_max) return(RColorBrewer::brewer.pal(max(3, n), brewer_name)[seq_len(n)])
  colorRampPalette(RColorBrewer::brewer.pal(brewer_max, brewer_name))(n)
}

# Create color mappings
heart_lv <- sort(unique(row_ann_df$Metadata_heart_number))
celltype_lv <- sort(unique(row_ann_df$Metadata_cell_type))
treatment_lv <- sort(unique(row_ann_df$Metadata_treatment))

heart_colors <- setNames(get_palette(length(heart_lv), "Set3", 12), heart_lv)
celltype_colors <- setNames(get_palette(length(celltype_lv), "Dark2", 8), celltype_lv)
treatment_colors <- setNames(get_palette(length(treatment_lv), "Paired", 12), treatment_lv)

# Turn annotation columns into factors
row_ann_df$Metadata_heart_number <- factor(row_ann_df$Metadata_heart_number, levels = names(heart_colors))
row_ann_df$Metadata_cell_type <- factor(row_ann_df$Metadata_cell_type, levels = names(celltype_colors))
row_ann_df$Metadata_treatment <- factor(row_ann_df$Metadata_treatment, levels = names(treatment_colors))

# Row annotation object
row_ha <- rowAnnotation(
  Heart = row_ann_df$Metadata_heart_number,
  CellType = row_ann_df$Metadata_cell_type,
  Treatment = row_ann_df$Metadata_treatment,
  col = list(
    Heart = heart_colors,
    CellType = celltype_colors,
    Treatment = treatment_colors
  ),
  show_annotation_name = TRUE,
  annotation_name_gp = gpar(fontsize = 12)
)

# Define correlation-based distance function
dist_cor <- function(x) {
  cor_mat <- cor(t(x), use = "pairwise.complete.obs")
  as.dist(1 - cor_mat)
}

# Hierarchical clustering for rows and columns
hc_rows <- hclust(dist_cor(mat_z), method = "average")
hc_cols <- hclust(dist_cor(t(mat_z)), method = "average")

# --- Heatmap with clustering ---
ht_clustered <- Heatmap(
  mat_z,
  name = "Z-score",
  show_row_names = FALSE,
  show_column_names = FALSE,
  cluster_rows = as.dendrogram(hc_rows),
  cluster_columns = as.dendrogram(hc_cols),
  col = colorRamp2(
    breaks = c(min(mat_z, na.rm = TRUE), 0, max(mat_z, na.rm = TRUE)),
    colors = c("#ca0020", "white", "#0571b0")
  ),
  heatmap_legend_param = list(title = "Z-score"),
  right_annotation = row_ha,
  column_title = sprintf("All Hearts: Feature Heatmap (Clustered, n = %d)", length(feature_cols)),
  column_title_gp = gpar(fontsize = 16, fontface = "bold")
)

# --- Heatmap without clustering ---
ht_unclustered <- Heatmap(
  mat_z,
  name = "Z-score",
  show_row_names = FALSE,
  show_column_names = FALSE,
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  col = colorRamp2(
    breaks = c(min(mat_z, na.rm = TRUE), 0, max(mat_z, na.rm = TRUE)),
    colors = c("#ca0020", "white", "#0571b0")
  ),
  heatmap_legend_param = list(title = "Z-score"),
  right_annotation = row_ha,
  column_title = sprintf("All Hearts: Feature Heatmap (Unclustered, n = %d)", length(feature_cols)),
  column_title_gp = gpar(fontsize = 16, fontface = "bold")
)

# Ensure output directory exists
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# Save clustered version
output_file_clustered <- file.path(output_dir, "all_hearts_clustered_heatmap.png")
png(filename = output_file_clustered, width = 5500, height = 4000, res = 400)
draw(ht_clustered, heatmap_legend_side = "right", annotation_legend_side = "right", merge_legend = TRUE)
dev.off()
message(sprintf("Saved clustered heatmap to %s", output_file_clustered))

# Save unclustered version
output_file_unclustered <- file.path(output_dir, "all_hearts_unclustered_heatmap.png")
png(filename = output_file_unclustered, width = 5500, height = 4000, res = 400)
draw(ht_unclustered, heatmap_legend_side = "right", annotation_legend_side = "right", merge_legend = TRUE)
dev.off()
message(sprintf("Saved unclustered heatmap to %s", output_file_unclustered))

