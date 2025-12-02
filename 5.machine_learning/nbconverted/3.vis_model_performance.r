suppressPackageStartupMessages({
    library(ggplot2)
    library(tidyr)
    library(dplyr)
    library(arrow)
    library(ggridges)
    library(RColorBrewer)
})

output_dir <- "figures"
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

pr_results <- read_parquet("./performance_metrics/all_models_pr_curve_results.parquet")

head(pr_results)

width = 14
height = 5
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
        strip.text = element_text(size = 16)  # Increase font size for facet labels
    )

# Save the plot to the output directory
ggsave(file.path(output_dir, "all_models_pr_curves.png"), all_models_pr_curves, dpi = 500, height = height, width = width)

all_models_pr_curves

probability_results <- read_parquet("./performance_metrics/all_models_predicted_probabilities.parquet")

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

ridge_plot_test <- ggplot(test_performance_df, aes(x = predicted_probability, y = actual_label, fill = Metadata_treatment)) +
  geom_density_ridges(aes(fill = Metadata_treatment), alpha = 0.7, scale = 2, rel_min_height = 0.01, bandwidth = 0.1) +
  scale_fill_manual(values = c("DMSO" = brewer.pal(8, "Dark2")[7])) +  # Only include DMSO color
  scale_x_continuous(breaks = seq(0, 1, 0.5)) +
  facet_grid(model_type ~ model_name, scales = "free_y") + 
  labs(x = "Probability of healthy prediction", y = "Heart Type") +  # Update x-axis label
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

ridge_plot_test <- ggplot(test_performance_df, aes(x = predicted_probability, y = factor(Metadata_heart_number), fill = Metadata_treatment)) +
  geom_density_ridges(aes(fill = Metadata_treatment), alpha = 0.7, scale = 2, rel_min_height = 0.01, bandwidth = 0.1) +
  scale_fill_manual(values = c("DMSO" = brewer.pal(8, "Dark2")[7])) +  # Only include DMSO color
  scale_x_continuous(breaks = seq(0, 1, 0.5)) +
  facet_grid(model_type ~ model_name, scales = "free_y") + 
  labs(x = "Probability of healthy prediction", y = "Heart Number") +  # Updated y-axis label
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
    actual_binary = as.integer(actual_label == "Healthy"),
    dataset_modified = case_when(
        Metadata_heart_number == 2 & dataset == "holdout" & Metadata_treatment == "None" ~ "holdout_media",
        TRUE ~ dataset
    )
) %>%
    group_by(model_name, model_type, dataset_modified, Metadata_heart_number, Metadata_treatment) %>%
    summarize(
        accuracy = mean(predicted_binary == actual_binary, na.rm = TRUE),
        n = n(),
        .groups = "drop"
    ) %>%
    # Ensure desired order for the dataset factor so bars appear in train, test, holdout, holdout_media order
    mutate(dataset_modified = factor(dataset_modified, levels = c("train", "test", "holdout", "holdout_media")))

# Plot accuracy per heart with modified dataset
height <- 6
width <- 16
options(repr.plot.width = width, repr.plot.height = height)

accuracy_barplot <- ggplot(
    accuracy_per_heart,
    aes(x = factor(Metadata_heart_number), y = accuracy, fill = dataset_modified)
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
multi_class_pr_results <- read_parquet("./performance_metrics/multi_class_pr_results.parquet")

head(multi_class_pr_results)

# Define mapping from integer → string
class_label <- c("0" = "DCM", "1" = "HCM", "2" = "Healthy")

# Update multi_class_pr_results if it exists
if (exists("multi_class_pr_results") && "class_label" %in% names(multi_class_pr_results)) {
  multi_class_pr_results$class_label <- class_label[as.character(multi_class_pr_results$class_label)]
}

# Print if the mapping was successful
head(multi_class_pr_results)

width = 14
height = 5
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
        strip.text = element_text(size = 16)  # Increase font size for facet labels
    )

# Save the plot to the output directory
ggsave(file.path(output_dir, "pr_curves_multiclass.png"), multi_class_pr_results_plot, dpi = 500, height = height, width = width)

multi_class_pr_results_plot

# Load in the accuracy results for multi-class model
multi_class_accuracy_results <- read_parquet("./performance_metrics/multi_class_heart_accuracy.parquet")

# Print the first few rows of the accuracy results
head(multi_class_accuracy_results)

# For heart_number 2: label as "holdout" if DMSO treated, "holdout_media" if None treatment; otherwise keep dataset
multi_class_accuracy_results <- multi_class_accuracy_results %>%
    mutate(
        dataset_modified = case_when(
            heart_number == 2 & dataset == "holdout" & treatment == "DMSO" ~ "holdout",
            heart_number == 2 & dataset == "holdout" & (treatment == "None" | is.na(treatment)) ~ "holdout_media",
            TRUE ~ dataset
        )
    )

# Ensure desired order for dataset_modified
multi_class_accuracy_results <- multi_class_accuracy_results %>%
    mutate(dataset_modified = factor(dataset_modified, levels = c("train", "test", "holdout", "holdout_media")))

# Plot accuracy per heart
height <- 6
width <- 16
options(repr.plot.width = width, repr.plot.height = height)

accuracy_barplot <- ggplot(
        multi_class_accuracy_results,
        aes(x = factor(heart_number), y = accuracy, fill = dataset_modified)
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

