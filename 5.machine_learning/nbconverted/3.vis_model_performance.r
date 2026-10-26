suppressPackageStartupMessages({
    library(ggplot2)
    library(tidyr)
    library(dplyr)
    library(arrow)
    library(ggridges)
    library(RColorBrewer)
})


# Set the plate to process with the PLATE_TO_TRAIN environment variable (defaults to the subtyping plate)
plate_to_process <- Sys.getenv("PLATE_TO_TRAIN", "CARD-CelIns-CX7_260814100001")

output_dir <- file.path("figures", plate_to_process)
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}


pr_results <- read_parquet(file.path("performance_metrics", plate_to_process, "all_models_pr_curve_results.parquet"))

head(pr_results)


# Size the figure from the number of panels (facet_wrap layout) so each panel is about square
panel_layout <- ggplot2::wrap_dims(length(unique(pr_results$model_name)))
width = 4.5 * panel_layout[2] + 3
height = 4.5 * panel_layout[1] + 1
options(repr.plot.width = width, repr.plot.height = height)  # Adjust width and height as desired

# # Filter the data for Train and Test only
# filtered_data <- pr_results[pr_results$dataset %in% c("train", "test"), ]

# Create the ggplot for PR curves
all_models_pr_curves <- ggplot(pr_results, aes(x = recall, y = precision, color = dataset, linetype = model_type)) +
    geom_line(linewidth = 1.15) +
    facet_wrap(model_name ~ .) +
    labs(
        x = "Recall",
        y = "Precision",
        color = "Data Type",
        linetype = "Model Type"
    ) +
    theme_bw() +
    theme(
        text = element_text(size = 16),  # Increase font size for all text
        axis.title = element_text(size = 18),  # Increase font size for axis titles
        axis.text = element_text(size = 14),  # Increase font size for axis text
        legend.title = element_text(size = 16),  # Increase font size for legend title
        legend.text = element_text(size = 14),  # Increase font size for legend text
        strip.text = element_text(size = 16),  # Increase font size for facet labels
        aspect.ratio = 1,  # Square panels so the PR curves are not flattened
        panel.spacing = unit(1.5, "lines")  # Space between panels so the axis labels do not overlap
    )

# Save the plot to the output directory
ggsave(file.path(output_dir, "all_models_pr_curves.png"), all_models_pr_curves, dpi = 500, height = height, width = width)

all_models_pr_curves


probability_results <- read_parquet(file.path("performance_metrics", plate_to_process, "all_models_predicted_probabilities.parquet"))

# Show healthy hearts as non-failing in the plots
probability_results$actual_label[probability_results$actual_label == "Healthy"] <- "Non-failing"

head(probability_results)


# Print unique Metadata_heart_number values and counts
unique_hearts <- sort(unique(probability_results$Metadata_heart_number))
cat("Unique Metadata_heart_number values (n =", length(unique_hearts), "):\n")
print(unique_hearts)

cat("\nCounts per Metadata_heart_number:\n")
print(table(probability_results$Metadata_heart_number))


# Filter the data for test data only
test_performance_df <- probability_results[probability_results$dataset %in% c("test"), ]

height <- 8
width <- 16
options(repr.plot.width = width, repr.plot.height = height)

ridge_plot_test <- ggplot(test_performance_df, aes(x = predicted_probability, y = actual_label, fill = actual_label)) +
  geom_density_ridges( alpha = 0.7, scale = 2, rel_min_height = 0.01, bandwidth = 0.1) +
  scale_fill_brewer(palette = "Dark2") +  # Color by heart type
  scale_x_continuous(breaks = seq(0, 1, 0.5)) +
  facet_grid(model_type ~ model_name, scales = "free_y") + 
  labs(x = "Probability of non-failing prediction", y = "Heart Type") +  # Update x-axis label
  theme_bw() +
  theme(
    legend.position = "none",
    axis.text = element_text(size = 20),
    axis.text.x = element_text(size = 20),
    axis.title = element_text(size = 24),
    strip.text = element_text(size = 18),
    strip.background = element_rect(
      colour = "black",
      fill = "#fdfff4"
    )
  )

# Save the plot to the output directory
ggsave(file.path(output_dir, "prob_ridge_plot_testing_data.png"), ridge_plot_test, dpi = 500, height = height, width = width)

ridge_plot_test


# Filter the data for test data only
test_performance_df <- probability_results[probability_results$dataset %in% c("test"), ]

height <- 8
width <- 16
options(repr.plot.width = width, repr.plot.height = height)

ridge_plot_test <- ggplot(test_performance_df, aes(x = predicted_probability, y = factor(Metadata_heart_number), fill = actual_label)) +
  geom_density_ridges( alpha = 0.7, scale = 2, rel_min_height = 0.01, bandwidth = 0.1) +
  scale_fill_brewer(palette = "Dark2") +  # Color by heart type
  scale_x_continuous(breaks = seq(0, 1, 0.5)) +
  facet_grid(model_type ~ model_name, scales = "free_y") + 
  labs(x = "Probability of non-failing prediction", y = "Heart Number") +  # Updated y-axis label
  theme_bw() +
  theme(
    legend.position = "none",
    axis.text = element_text(size = 20),
    axis.text.x = element_text(size = 20),
    axis.title = element_text(size = 24),
    strip.text = element_text(size = 18),
    strip.background = element_rect(
      colour = "black",
      fill = "#fdfff4"
    )
  )

# Save the plot to the output directory
ggsave(file.path(output_dir, "prob_ridge_plot_per_heart_testing_data.png"), ridge_plot_test, dpi = 500, height = height, width = width)

ridge_plot_test


head(probability_results)


# Compute accuracy per heart, per split, per model
accuracy_per_heart <- probability_results %>%
    mutate(
    predicted_binary = ifelse(predicted_probability >= 0.5, 1L, 0L),
    actual_binary = as.integer(actual_label == "Non-failing")
) %>%
    group_by(model_name, model_type, dataset, Metadata_heart_number) %>%
    summarize(
        accuracy = mean(predicted_binary == actual_binary, na.rm = TRUE),
        n = n(),
        .groups = "drop"
    ) %>%
    # Ensure desired order for the dataset factor so bars appear in train, test, holdout well, holdout female hearts order
    mutate(dataset = factor(dataset, levels = c("train", "test", "holdout_well", "holdout_female_hearts")))

# Plot accuracy per heart per dataset
height <- 6
width <- 16
options(repr.plot.width = width, repr.plot.height = height)

accuracy_barplot <- ggplot(
    accuracy_per_heart,
    aes(x = factor(Metadata_heart_number), y = accuracy, fill = dataset)
) +
    geom_col(position = "dodge") +
    facet_grid(model_type ~ model_name, scales = "free_x", space = "free") +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 1)) +
    labs(
        x = "Heart Number",
        y = "Accuracy",
        fill = "Dataset"
    ) +
    theme_bw() +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
        axis.text.y = element_text(size = 12),
        axis.title = element_text(size = 14),
        strip.text = element_text(size = 12),
        legend.position = "bottom"
    )

# Save plot
ggsave(file.path(output_dir, "accuracy_per_heart_by_model.png"), accuracy_barplot, dpi = 600, width = width, height = height)

accuracy_barplot


# Load in PR curve results for multi-class model
multi_class_pr_results <- read_parquet(file.path("performance_metrics", plate_to_process, "multi_class_pr_results.parquet"))

head(multi_class_pr_results)


# Define mapping from integer → string (from the multi-class label encoder in 1.train_models)
label_mapping <- read.csv(file.path("performance_metrics", plate_to_process, "multi_class_label_mapping.csv"))
class_label <- setNames(label_mapping$class_label, as.character(label_mapping$encoded_label))
# Show healthy hearts as non-failing in the plots
class_label[class_label == "Healthy"] <- "Non-failing"

# Update multi_class_pr_results if it exists
if (exists("multi_class_pr_results") && "class_label" %in% names(multi_class_pr_results)) {
  multi_class_pr_results$class_label <- class_label[as.character(multi_class_pr_results$class_label)]
}

# Print if the mapping was successful
head(multi_class_pr_results)


# Size the figure from the number of panels (facet_wrap layout) so each panel is about square
panel_layout <- ggplot2::wrap_dims(length(unique(multi_class_pr_results$class_label)))
width = 4.5 * panel_layout[2] + 3
height = 4.5 * panel_layout[1] + 1
options(repr.plot.width = width, repr.plot.height = height)  # Adjust width and height as desired

# Create the ggplot for PR curves
multi_class_pr_results_plot <- ggplot(multi_class_pr_results, aes(x = recall, y = precision, color = dataset, linetype = model_type)) +
    geom_line(linewidth = 1.15) +
    facet_wrap(class_label ~ .) +
    labs(
        x = "Recall",
        y = "Precision",
        color = "Data Type",
        linetype = "Model Type"
    ) +
    theme_bw() +
    theme(
        text = element_text(size = 16),  # Increase font size for all text
        axis.title = element_text(size = 18),  # Increase font size for axis titles
        axis.text = element_text(size = 14),  # Increase font size for axis text
        legend.title = element_text(size = 16),  # Increase font size for legend title
        legend.text = element_text(size = 14),  # Increase font size for legend text
        strip.text = element_text(size = 16),  # Increase font size for facet labels
        aspect.ratio = 1,  # Square panels so the PR curves are not flattened
        panel.spacing = unit(1.5, "lines")  # Space between panels so the axis labels do not overlap
    )

# Save the plot to the output directory
ggsave(file.path(output_dir, "pr_curves_multiclass.png"), multi_class_pr_results_plot, dpi = 500, height = height, width = width)

multi_class_pr_results_plot


# Load in the accuracy results for multi-class model
multi_class_accuracy_results <- read_parquet(file.path("performance_metrics", plate_to_process, "multi_class_heart_accuracy.parquet"))

# Print the first few rows of the accuracy results
head(multi_class_accuracy_results)


# Ensure desired order for dataset
multi_class_accuracy_results <- multi_class_accuracy_results %>%
    mutate(dataset = factor(dataset, levels = c("train", "test", "holdout_well", "holdout_female_hearts")))

# Plot accuracy per heart
height <- 6
width <- 16
options(repr.plot.width = width, repr.plot.height = height)

accuracy_barplot <- ggplot(
        multi_class_accuracy_results,
        aes(x = factor(heart_number), y = accuracy, fill = dataset)
) +
        geom_col(position = "dodge") +
        facet_wrap(model_type ~ .) +
        scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 1)) +
        labs(
                x = "Heart Number",
                y = "Accuracy",
                fill = "Dataset"
        ) +
        theme_bw() +
        theme(
                axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
                axis.text.y = element_text(size = 12),
                axis.title = element_text(size = 14),
                strip.text = element_text(size = 12),
                legend.position = "bottom"
        )

# Save plot
ggsave(file.path(output_dir, "accuracy_per_heart_multiclass.png"), accuracy_barplot, dpi = 600, width = width, height = height)

accuracy_barplot


# Load in the predicted probabilities per single cell for the multi-class model
multi_class_probability_results <- read_parquet(file.path("performance_metrics", plate_to_process, "multi_class_cell_probabilities.parquet"))

# Order of the subtypes for plotting (Non-failing first, then the failing subtypes)
subtype_order <- c("Non-failing", sort(setdiff(class_label, "Non-failing")))

# Reshape to one row per single cell and subtype probability, and map the encoded labels to subtypes
multi_class_test_probabilities <- multi_class_probability_results %>%
    filter(dataset == "test") %>%
    pivot_longer(
        cols = starts_with("proba_class_"),
        names_to = "predicted_class",
        names_prefix = "proba_class_",
        values_to = "predicted_probability"
    ) %>%
    mutate(
        true_label = factor(class_label[as.character(true_label)], levels = subtype_order),
        predicted_class = factor(
            paste("Probability of", class_label[predicted_class]),
            levels = paste("Probability of", subtype_order)
        )
    )

head(multi_class_test_probabilities)

height <- 8
width <- 16
options(repr.plot.width = width, repr.plot.height = height)

multi_class_ridge_plot_test <- ggplot(multi_class_test_probabilities, aes(x = predicted_probability, y = true_label, fill = true_label)) +
  geom_density_ridges(alpha = 0.7, scale = 2, rel_min_height = 0.01, bandwidth = 0.1) +
  scale_fill_brewer(palette = "Dark2") +  # Color by heart type
  scale_x_continuous(breaks = seq(0, 1, 0.5)) +
  facet_grid(model_type ~ predicted_class, scales = "free_y") +
  labs(x = "Predicted probability", y = "Heart Type") +
  theme_bw() +
  theme(
    legend.position = "none",
    axis.text = element_text(size = 20),
    axis.text.x = element_text(size = 20),
    axis.title = element_text(size = 24),
    strip.text = element_text(size = 18),
    strip.background = element_rect(
      colour = "black",
      fill = "#fdfff4"
    )
  )

# Save the plot to the output directory
ggsave(file.path(output_dir, "prob_ridge_plot_multiclass_testing_data.png"), multi_class_ridge_plot_test, dpi = 500, height = height, width = width)

multi_class_ridge_plot_test

height <- 10
width <- 16
options(repr.plot.width = width, repr.plot.height = height)

# Order hearts by subtype so hearts from the same subtype are next to each other
heart_order <- multi_class_test_probabilities %>%
    distinct(Metadata_heart_number, true_label) %>%
    arrange(true_label, Metadata_heart_number) %>%
    pull(Metadata_heart_number)

multi_class_ridge_plot_per_heart <- ggplot(
    multi_class_test_probabilities,
    aes(x = predicted_probability, y = factor(Metadata_heart_number, levels = heart_order), fill = true_label)
) +
  geom_density_ridges(alpha = 0.7, scale = 2, rel_min_height = 0.01, bandwidth = 0.1) +
  scale_fill_brewer(palette = "Dark2") +  # Color by heart type
  scale_x_continuous(breaks = seq(0, 1, 0.5)) +
  facet_grid(model_type ~ predicted_class, scales = "free_y") +
  labs(x = "Predicted probability", y = "Heart Number", fill = "Heart Type") +
  theme_bw() +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 18),
    legend.title = element_text(size = 20),
    axis.text = element_text(size = 20),
    axis.text.x = element_text(size = 20),
    axis.title = element_text(size = 24),
    strip.text = element_text(size = 18),
    strip.background = element_rect(
      colour = "black",
      fill = "#fdfff4"
    )
  )

# Save the plot to the output directory
ggsave(file.path(output_dir, "prob_ridge_plot_multiclass_per_heart_testing_data.png"), multi_class_ridge_plot_per_heart, dpi = 500, height = height, width = width)

multi_class_ridge_plot_per_heart
