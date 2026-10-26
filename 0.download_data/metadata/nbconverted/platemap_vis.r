suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(platetools))


# Shape (filled) for each heart failure subtype, consistent across all platemaps
heart_failure_type_shapes <- c("Non-failing" = 21, "DCM" = 22, "ICM" = 23, "HLHS" = 24)
# Squares, diamonds, and triangles are scaled to look the same size as circles
heart_failure_type_sizes <- c("Non-failing" = 16.5, "DCM" = 15, "ICM" = 13.5, "HLHS" = 11.5)
# Sizes of the shapes in the legend so they look the same size as each other
heart_failure_type_legend_sizes <- c("Non-failing" = 6, "DCM" = 5.4, "ICM" = 4.4, "HLHS" = 3.8)

# Plot a 96-well platemap where the well shape and color show the heart failure subtype
# and the label (well_label column) is shown inside each well
plot_heart_number_platemap <- function(platemap_df, plate, colors, caption) {
    # Only keep the subtypes on this plate (in a consistent order)
    subtypes <- intersect(names(heart_failure_type_shapes), unique(platemap_df$heart_failure_type))

    platemap_df <- platemap_df %>%
        dplyr::mutate(
            heart_failure_type = factor(heart_failure_type, levels = subtypes),
            well_col = as.integer(well_col),
            well_row = factor(well_row, levels = rev(LETTERS[1:8]))
        )

    # Wells on the 96-well plate without cells (shown as an empty outline)
    empty_wells_df <- expand.grid(well_row = LETTERS[1:8], well_col = 1:12) %>%
        dplyr::mutate(well_row = factor(well_row, levels = rev(LETTERS[1:8]))) %>%
        dplyr::anti_join(platemap_df, by = c("well_row", "well_col"))

    ggplot() +
        geom_point(
            data = empty_wells_df,
            aes(x = well_col, y = well_row),
            shape = 21,
            size = 16.5,
            color = "#d9d9d6",
            fill = NA,
            stroke = 0.5
        ) +
        geom_point(
            data = platemap_df,
            aes(
                x = well_col,
                y = well_row,
                fill = heart_failure_type,
                shape = heart_failure_type,
                size = heart_failure_type
            ),
            color = "#4d4d4b",
            stroke = 0.6
        ) +
        geom_text(
            data = platemap_df,
            aes(x = well_col, y = well_row, label = well_label),
            color = "white",
            fontface = "bold",
            size = 3.6
        ) +
        scale_fill_manual(name = "Heart failure subtype", values = colors[subtypes]) +
        scale_shape_manual(name = "Heart failure subtype", values = heart_failure_type_shapes[subtypes]) +
        scale_size_manual(name = "Heart failure subtype", values = heart_failure_type_sizes[subtypes]) +
        scale_x_continuous(breaks = 1:12, position = "top", limits = c(0.5, 12.5), expand = c(0, 0)) +
        scale_y_discrete(expand = expansion(add = 0.6), drop = FALSE) +
        coord_fixed() +
        labs(
            title = paste("Platemap layout for plate", plate),
            caption = caption,
            x = NULL,
            y = NULL
        ) +
        theme_minimal(base_size = 12) +
        theme(
            plot.title = element_text(size = 14, face = "bold", hjust = 0),
            plot.title.position = "plot",
            plot.caption = element_text(size = 10, color = "#555555", hjust = 1),
            panel.grid = element_blank(),
            panel.border = element_rect(color = "#999999", fill = NA, linewidth = 0.6),
            axis.text = element_text(size = 12, color = "#555555"),
            legend.position = "right",
            legend.title = element_text(size = 11, face = "bold"),
            legend.text = element_text(size = 11),
            legend.key.height = unit(0.9, "cm")
        ) +
        guides(
            # Draw the legend shapes at the same size as each other
            size = "none",
            fill = guide_legend(override.aes = list(size = unname(heart_failure_type_legend_sizes[subtypes])))
        )
}

platemap_dir <- file.path("platemaps")
platemap_files <- list.files(platemap_dir, pattern = "_platemap\\.csv$", full.names = TRUE)
print(platemap_files)

output_fig_dir <- file.path("platemap_figures")
if (!dir.exists(output_fig_dir)) {
    dir.create(output_fig_dir)
}
platemap_suffix <- "_platemap_figure.png"

# Define output figure paths
output_platemap_files <- list()
for (platemap_file in platemap_files) {
    # Extract plate name and remove suffix 
    plate <- basename(platemap_file)
    plate <- stringr::str_remove(plate, "_platemap.csv") 
    
    output_platemap_files[[plate]] <- file.path(output_fig_dir, paste0(plate, platemap_suffix))
}

print(output_platemap_files)


# Load in all platemap CSV files
platemap_dfs <- list()
for (plate in names(output_platemap_files)) {
    # Find the umap file associated with the plate
    platemap_file <- platemap_files[stringr::str_detect(platemap_files, plate)]
    
    # Load in the umap data
    df <- readr::read_csv(
    platemap_file,
    col_types = readr::cols(.default = "c")
)

    # for plotting replace NaN as "Non-failing" for heart failure type and show healthy hearts as non-failing
    df$heart_failure_type[is.na(df$heart_failure_type)] <- "Non-failing"
    df$cell_type[df$cell_type == "Healthy"] <- "Non-failing"


    platemap_dfs[[plate]] <- df 
}

print(platemap_dfs)


# Consistent color mapping for heart failure subtype across all platemaps
heart_failure_type_colors <- c(
    "Non-failing" = "#009E73",  # green (Okabe-Ito palette)
    "DCM" = "#D55E00",      # vermillion
    "ICM" = "#0072B2",      # blue
    "HLHS" = "#CC79A7"      # pink-purple
)

for (plate in names(platemap_dfs)) {
 {
    # output for each plate
    output_file <- output_platemap_files[[plate]]
    output_file <- paste0(output_file)
    
    platemap <-
        platetools::raw_map(
            data = platemap_dfs[[plate]]$heart_failure_type,
            well = platemap_dfs[[plate]]$well_position,
            plate = 96,
            size = 8
        ) +
        ggtitle(paste("Platemap layout for plate", plate)) +
        theme(plot.title = element_text(hjust=0.5, size = 10, face = "bold", margin = margin(b = -5))) +
        ggplot2::geom_point(aes(shape = platemap_dfs[[plate]]$cell_type)) +
        ggplot2::scale_shape_discrete(name = "Heart condition") +
        ggplot2::scale_fill_manual(name = "Heart failure subtype", values = heart_failure_type_colors) +
        theme(
            legend.position = "right",
            legend.title = element_text(size = 9),
            legend.text = element_text(size = 8),
        ) +
        guides(
        shape = guide_legend(override.aes = list(size = 2)),
        fill = guide_legend(override.aes = list(size = 5))
    )

    ggsave(
        output_file,
        platemap,
        dpi = 500,
        height = 3.5,
        width = 6
    )
    }
}


# Platemap with the well shape showing the heart failure subtype and the heart number in each well
plate <- "nf_heart_failure_subtypes"

# Female hearts (all other hearts are male)
female_hearts <- c("32", "64")

# Label with the heart number and sex (M = male, F = female)
platemap_df <- platemap_dfs[[plate]] %>%
    dplyr::mutate(well_label = paste0(heart_number, ifelse(heart_number %in% female_hearts, "F", "M")))

heart_number_platemap <- plot_heart_number_platemap(
    platemap_df,
    plate,
    heart_failure_type_colors,
    caption = "Label in well = heart number and sex (M = male, F = female)"
)

ggsave(
    file.path(output_fig_dir, paste0(plate, "_platemap_figure_heart_numbers.png")),
    heart_number_platemap,
    dpi = 500,
    height = 5.5,
    width = 10.5,
    bg = "white"
)

heart_number_platemap

# Platemap with the well shape showing the heart failure subtype and the heart number in each well
plate <- "hlhs_heart_failure_subtypes"

# Label with the heart number (HLHS hearts are labeled with letters)
platemap_df <- platemap_dfs[[plate]] %>%
    dplyr::mutate(well_label = heart_number)

heart_number_platemap <- plot_heart_number_platemap(
    platemap_df,
    plate,
    heart_failure_type_colors,
    caption = "Label in well = heart number"
)

ggsave(
    file.path(output_fig_dir, paste0(plate, "_platemap_figure_heart_numbers.png")),
    heart_number_platemap,
    dpi = 500,
    height = 5.5,
    width = 10.5,
    bg = "white"
)

heart_number_platemap
