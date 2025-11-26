#Data structure
#i <- instance_id
#Ouput[[i]] gives the instance and within it all the methods
#and their respective datasets and models

library(xtable)
library(ggplot2)
library(RColorBrewer)
library(viridis)
library(patchwork)

# Define a color-blind friendly palette
cb_palette <- c(
  "#E69F00", "#56B4E9", "#009E73",
  "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
)
#---------------------------

num_instances <- 1

# Create a list to hold all instances
vd <- vector("list", num_instances)

# Loop over instance files
for (i in 1:num_instances) {
  file_path <- paste0("/Users/niyati/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/n_vars=60/instance_", i, ".RData")
  vd[[i]] <- readRDS(file_path)
}

prbocs_data <- list()
prbocs_ga_data <- list()
prbocs_vb <- list()
prbocs_vb_ga <- list()
bocs_ga <- list()
bocs_sa <- list()
bocs_sdp <- list()

#Extract data for each instance
for(i in 1){
  prbocs_data[[i]] <- vd[[i]]$PRBOCS$prbocs_result$data
  # prbocs_ga_data[[i]] <- vd[[i]]$PRBOCS_GA$prbocsga_result$data
  prbocs_vb[[i]] <- vd[[i]]$PRBOCS_VB$prbocs_vb_result$data
  prbocs_vb_ga[[i]] <- vd[[i]]$PRBOCS_VB_GA$prbocs_vb_ga_result$data 
  # bocs_ga[[i]] <- vd[[i]]$BOCS_GA$bocsga_result$data 
  bocs_sa[[i]] <- vd[[i]]$BOCS_SA$bocssa_result$data
  # bocs_sdp[[i]] <- vd[[i]]$BOCS_SDP$bocssdp_result$data
}

summary_list <- list(
  PRBOCS = prbocs_data,
  # PRBOCS_GA = prbocs_ga_data,
  PRBOCS_VB = prbocs_vb,
  PRBOCS_VB_GA = prbocs_vb_ga,
  # BOCS_GA = bocs_ga,
  BOCS_SA = bocs_sa
  # BOCS_SDP = bocs_sdp
)

# --- Function to compute rowwise mean & CI ---
# 
# Check for NAs/NULL data
lapply(summary_list, function(method_list) {
  sapply(method_list, function(entry) {
    is_valid <- !is.null(entry) && is.data.frame(entry) && "y" %in% names(entry)
    if (is_valid) {
      nrow(entry)
    } else {
      NA  # or return 0
    }
  })
})

get_mean_df <- function(method_list, method_name) {
  valid_list <- Filter(function(x) {
    !is.null(x) && is.data.frame(x) && "y" %in% names(x)
  }, method_list)
  
  if (length(valid_list) == 0) return(NULL)
  
  num_rows <- nrow(prbocs_data[[1]])
  # num_rows <- nrow(method_list[[1]])
  y_means <- numeric(num_rows)
  y_lowers <- numeric(num_rows)
  y_uppers <- numeric(num_rows)
  
  for (j in 1:num_rows) {
    y_vals <- sapply(method_list, function(df) df$y[j])
    y_means[j] <- mean(y_vals)
    stderr <- sd(y_vals) / sqrt(length(y_vals))
    error_margin <- qt(0.975, df = length(y_vals) - 1) * stderr
    y_lowers[j] <- y_means[j] - error_margin
    y_uppers[j] <- y_means[j] + error_margin
  }
  
  data.frame(
    row = 1:num_rows,
    mean_y = y_means,
    ci_lower = y_lowers,
    ci_upper = y_uppers,
    method = method_name
  )
}

# --- Create combined summary dataframe ---
all_results <- do.call(rbind, lapply(names(summary_list), function(name) {
  df <- get_mean_df(summary_list[[name]], name)
  if (!is.null(df)) df else NULL
}))

all_results$method <- factor(all_results$method, levels = unique(all_results$method))

# Colorblind-friendly palette
cb_palette <- viridis::viridis(length(unique(all_results$method)))

# --- Thesis-style color palette ---
# Define thesis color palette
cb_palette <- c(
  "#1B9E77",  # green
  "#D95F02",  # orange
  "#7570B3",  # violet
  "#E7298A"   # magenta
)

# === Plot 1: Mean Objective Value Across Methods ===
mean_plot <- ggplot(all_results, aes(x = row, y = mean_y, color = method)) +
  geom_line(size = 0.7, alpha = 0.9, lineend = "round") +  # thinner, crisp lines
  scale_color_manual(values = cb_palette) +
  labs(
    x = "Iteration", y = "Mean Objective Value"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.title = element_blank(),
    legend.position = "right",
    legend.text = element_text(size = 11),
    plot.title = element_text(size = 16, hjust = 0.5),
    axis.text = element_text(size = 12),
    panel.grid.minor = element_blank()
  )

# === Plot 2: Best Objective Value Across Methods ===
# Compute running best (assuming you want to maximize the objective)
best_results <- all_results %>%
  group_by(method) %>%
  arrange(row) %>%
  mutate(best_y = cummin(mean_y))  # use cummin(mean_y) if minimizing

best_plot <- ggplot(best_results, aes(x = row, y = best_y, color = method)) +
  geom_line(size = 0.7, alpha = 0.9, lineend = "round") +
  scale_color_manual(values = cb_palette) +
  labs(
    x = "Iteration", y = "Best Objective Value"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.title = element_blank(),
    legend.position = "right",
    legend.text = element_text(size = 11),
    plot.title = element_text(size = 16, hjust = 0.5),
    axis.text = element_text(size = 12),
    panel.grid.minor = element_blank()
  )

# === Save both as LaTeX-ready PDFs ===
ggsave(
  filename = "/Users/niyati/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/n_vars=60/mean_objective_methods.pdf",
  plot = mean_plot,
  device = cairo_pdf,
  width = 11, height = 6, units = "in",
  dpi = 1200, bg = "white"
)

ggsave(
  filename = "/Users/niyati/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/n_vars=60/best_objective_maxsat60.pdf",
  plot = best_plot,
  device = cairo_pdf,
  width = 11, height = 6, units = "in",
  dpi = 1200, bg = "white"
)


summary_table <- all_results %>%
  group_by(method) %>%
  summarise(
    Mean = mean(mean_y),
    SD = sd(mean_y),
    Best = max(mean_y),
    Worst = min(mean_y)
  )

# Round for presentation
summary_table <- summary_table %>%
  mutate(across(where(is.numeric), ~ round(., 2)))

# Generate LaTeX table
xt <- xtable(summary_table,
             caption = "Summary of objective values across optimization methods.",
             label = "tab:summary_methods",
             align = c("l", "l", "r", "r", "r", "r"))

print(
  xt,
  include.rownames = FALSE,
  sanitize.text.function = identity,
  comment = FALSE,
  booktabs = TRUE,
  caption.placement = "top"
)

# --- Grid of CI plots (one per method) ---
# Define methods
methods <- names(summary_list)

# Generate individual CI plots
ci_plots <- lapply(seq_along(methods), function(i) {
  method_name <- methods[i]
  method_data <- summary_list[[i]]
  df <- get_mean_df(method_data, method_name)
  
  ggplot(df, aes(x = row, y = mean_y)) +
    geom_line(color = cb_palette[i], size = 0.6) +  # thinner line for print
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), 
                fill = cb_palette[i], alpha = 0.2) +  # lighter fill
    labs(title = method_name, x = "Iteration", y = "Mean y") +
    theme_minimal(base_size = 9) +  # slightly smaller base font
    theme(
      plot.title = element_text(size = 9, face = "bold", hjust = 0.5),
      axis.title = element_text(size = 8),
      axis.text = element_text(size = 7),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.2, colour = "grey85")
    )
})

# Arrange plots
ci_grid_plot <- wrap_plots(ci_plots, ncol = 3)

# Save high-quality vector PDF for printing
ggsave(
  filename = "/Users/niyati/Projects:Codes/PhD-Compute/R code/TestPrblms/Ising/n_vars=24 old/ci_plots_grid_ins10.pdf",
  plot = ci_grid_plot, 
  device = cairo_pdf,   # ensures vector output
  width = 11,           # A4 width
  height = 6,           # nice proportion for 3x2 grid
  units = "in",
  dpi = 1200,           # high-resolution if raster fallback occurs
  bg = "white"
)



#-----------------------------
# Computaional efficiency
#-----------------------------

prbocs_time <- list()
prbocs_ga_time <- list()
prbocs_vb_time <- list()
prbocs_vb_ga_time <- list()
bocs_ga_time <- list()
bocs_sa_time <- list()
bocs_sdp_time <- list()

#Extract time for each instance
for(i in 1:num_instances){
  prbocs_time[[i]] <- vd[[i]]$PRBOCS$time_taken
  prbocs_ga_time[[i]] <- vd[[i]]$PRBOCS_GA$time_taken
  prbocs_vb_time[[i]] <- vd[[i]]$PRBOCS_VB$time_taken
  prbocs_vb_ga_time[[i]] <- vd[[i]]$PRBOCS_VB_GA$time_taken
  bocs_ga_time[[i]] <- vd[[i]]$BOCS_GA$time_taken
  bocs_sa_time[[i]] <- vd[[i]]$BOCS_SA$time_taken
  bocs_sdp_time[[i]] <- vd[[i]]$BOCS_SDP$time_taken
}

time_list <- list(
  PRBOCS = prbocs_time,
  # PRBOCS_GA = prbocs_ga_time,
  PRBOCS_VB = prbocs_vb_time,
  PRBOCS_VB_GA = prbocs_vb_ga_time,
  # BOCS_GA = bocs_ga_time,
  BOCS_SA = bocs_sa_time,
  # BOCS_SDP = bocs_sdp_time
)

# Convert difftime to numeric hours
time_to_hours <- function(x) as.numeric(x, units = "hours")
# Convert difftime to numeric minutes
time_to_minutes <- function(x) as.numeric(x, units = "mins")

# time_df <- do.call(rbind, lapply(names(time_list), function(name) {
#   times <- sapply(time_list[[name]], time_to_minutes)
#   mean_time <- mean(times)
#   sd_time <- sd(times)
#   data.frame(
#     Method = name,
#     Time = sprintf("%.2f ± %.2f", mean_time, sd_time),
#     check.names = FALSE
#   )
# }))

time_df <- do.call(rbind, lapply(names(time_list), function(name) {
  times <- sapply(time_list[[name]], time_to_hours)
  mean_time <- mean(times)
  data.frame(
    Method = name,
    Time = sprintf("%.2f", mean_time),
    check.names = FALSE
  )
}))

time_df <- as.data.frame(time_df)

library(xtable)

print(
  xtable(time_df,
         caption = "Computation Time (in hours) for Each Method",
         label = "tab:time_results"),
  include.rownames = FALSE,
  sanitize.text.function = identity  # Keep ± symbol
)

# Create a data frame in long format for the boxplot
boxplot_data <- do.call(rbind, lapply(names(time_list), function(name) {
  times <- sapply(time_list[[name]], time_to_minutes)
  data.frame(
    Method = name,
    Time_minutes = times
  )
}))

# Create the boxplot with log scale
p <- ggplot(boxplot_data, aes(x = Method, y = Time_minutes, fill = Method)) +
  geom_boxplot() +
  theme_minimal(base_size = 14) +
  labs(title = "Execution Time Comparison (Log Scale)",
       x = "Method",
       y = "Time (minutes, log scale)") +
  theme(legend.position = "none",
        plot.title = element_text(size = 16, face = "bold"),
        axis.title = element_text(size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 12)) +
  scale_fill_manual(values = cb_palette) +
  scale_y_log10() +  # Apply log10 scale to y-axis
  annotation_logticks(sides = "l")  # Add log scale ticks to left side

# Save the plot to PDF
pdf("execution_time_boxplot.pdf", width = 10, height = 7)
print(p)
dev.off()

# Create the main boxplot with log scale and better handling of outliers
p1 <- ggplot(boxplot_data, aes(x = Method, y = Time_minutes, fill = Method)) +
  geom_boxplot(outlier.shape = 1, outlier.size = 2, outlier.alpha = 0.7) +
  theme_minimal(base_size = 14) +
  labs(title = "Execution Time Comparison (Log Scale)",
       x = "Method",
       y = "Time (minutes, log scale)") +
  theme(legend.position = "none",
        plot.title = element_text(size = 16, face = "bold"),
        axis.title = element_text(size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 12)) +
  scale_fill_manual(values = cb_palette) +
  scale_y_log10() +
  annotation_logticks(sides = "l")

# Create a second plot with a focused y-range to better compare methods
# Determine reasonable bounds: from 25th percentile of min method to 75th percentile of max method
summary_stats <- boxplot_data %>% 
  group_by(Method) %>% 
  summarize(
    q1 = quantile(Time_minutes, 0.25),
    q3 = quantile(Time_minutes, 0.75)
  )

min_y <- min(summary_stats$q1) * 0.8  # Lower bound: 80% of lowest Q1
max_y <- max(summary_stats$q3) * 1.2  # Upper bound: 120% of highest Q3

p2 <- ggplot(boxplot_data, aes(x = Method, y = Time_minutes, fill = Method)) +
  geom_boxplot(outlier.shape = 1, outlier.size = 2, outlier.alpha = 0.7) +
  theme_minimal(base_size = 14) +
  labs(title = "Execution Time Comparison (Focused Range)",
       x = "Method",
       y = "Time (minutes, log scale)") +
  theme(legend.position = "none",
        plot.title = element_text(size = 16, face = "bold"),
        axis.title = element_text(size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 12)) +
  scale_fill_manual(values = cb_palette) +
  scale_y_log10(limits = c(min_y, max_y)) +
  annotation_logticks(sides = "l")

# Add points to see the actual distribution of values
p3 <- ggplot(boxplot_data, aes(x = Method, y = Time_minutes, fill = Method)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.6, size = 2, aes(color = Method)) +
  theme_minimal(base_size = 14) +
  labs(title = "Execution Time Comparison with Individual Data Points",
       x = "Method",
       y = "Time (minutes, log scale)") +
  theme(legend.position = "none",
        plot.title = element_text(size = 16, face = "bold"),
        axis.title = element_text(size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 12)) +
  scale_fill_manual(values = cb_palette) +
  scale_color_manual(values = cb_palette) +
  scale_y_log10() +
  annotation_logticks(sides = "l")

# Make sure gridExtra is installed for arranging plots
if(!requireNamespace("gridExtra", quietly = TRUE)) {
  install.packages("gridExtra")
}
library(gridExtra)

# Save all plots to PDF
pdf("execution_time_comparison.pdf", width = 10, height = 18)
grid.arrange(p1, p2, p3, ncol = 1)
dev.off()

# Display plots in R
grid.arrange(p1, p2, p3, ncol = 1)

# Create a summary table
time_summary <- boxplot_data %>%
  group_by(Method) %>%
  summarize(
    Mean = mean(Time_minutes),
    SD = sd(Time_minutes),
    Median = median(Time_minutes),
    Min = min(Time_minutes),
    Max = max(Time_minutes),
    Q1 = quantile(Time_minutes, 0.25),
    Q3 = quantile(Time_minutes, 0.75)
  )

# Print the summary table
print(time_summary)


