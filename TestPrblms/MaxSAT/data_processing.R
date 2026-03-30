#Data structure
library(xtable)
library(ggplot2)
library(RColorBrewer)
library(viridis)
library(patchwork)
library(dplyr)

# Define a color-blind friendly palette
cb_palette1 <- c(
  "#E69F00", "#56B4E9", "#009E73",
  "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
)

#---------------------------
num_instances <- 1

vd <- vector("list", num_instances)

for (i in 1:num_instances) {
  file_path <- paste0("/Users/niyati/Projects:Codes/VaR-CoBO/TestPrblms/MaxSAT/instance_", i, ".RData")
  vd[[i]] <- readRDS(file_path)
}

prbocs_data    <- list()
prbocs_ga_data <- list()
prbocs_vb      <- list()
prbocs_vb_ga   <- list()
bocs_ga        <- list()
bocs_sa        <- list()
bocs_sdp       <- list()

for(i in 1){
  prbocs_data[[i]]  <- vd[[i]]$PRBOCS$prbocs_result$data
  prbocs_vb[[i]]    <- vd[[i]]$PRBOCS_VB$prbocs_vb_result$data
  prbocs_vb_ga[[i]] <- vd[[i]]$PRBOCS_VB_GA$prbocs_vb_ga_result$data
  bocs_sa[[i]]      <- vd[[i]]$BOCS_SA$bocssa_result$data
}

# UPDATED: correct names and order to match paper
summary_list <- list(
  `BOCS-SA`      = bocs_sa,
  `PRCBO-BFGS`   = prbocs_data,
  `VaR-CBO-BFGS` = prbocs_vb,
  `VaR-CBO-GA`   = prbocs_vb_ga
)

# --- Check for NAs/NULL data ---
lapply(summary_list, function(method_list) {
  sapply(method_list, function(entry) {
    is_valid <- !is.null(entry) && is.data.frame(entry) && "y" %in% names(entry)
    if (is_valid) nrow(entry) else NA
  })
})

get_mean_df <- function(method_list, method_name) {
  valid_list <- Filter(function(x) {
    !is.null(x) && is.data.frame(x) && "y" %in% names(x)
  }, method_list)
  
  if (length(valid_list) == 0) return(NULL)
  
  num_rows <- nrow(prbocs_data[[1]])
  y_means  <- numeric(num_rows)
  y_lowers <- numeric(num_rows)
  y_uppers <- numeric(num_rows)
  
  for (j in 1:num_rows) {
    y_vals <- sapply(method_list, function(df) df$y[j])
    y_means[j]  <- mean(y_vals)
    stderr       <- sd(y_vals) / sqrt(length(y_vals))
    error_margin <- qt(0.975, df = length(y_vals) - 1) * stderr
    y_lowers[j]  <- y_means[j] - error_margin
    y_uppers[j]  <- y_means[j] + error_margin
  }
  
  data.frame(
    row      = 1:num_rows,
    mean_y   = y_means,
    ci_lower = y_lowers,
    ci_upper = y_uppers,
    method   = method_name
  )
}

# --- Create combined summary dataframe ---
all_results <- do.call(rbind, lapply(names(summary_list), function(name) {
  df <- get_mean_df(summary_list[[name]], name)
  if (!is.null(df)) df else NULL
}))

# UPDATED: factor levels reflect new order
all_results$method <- factor(all_results$method, levels = names(summary_list))

# Thesis colour palette
cb_palette <- c(
  "#1B9E77",  # green
  "#D95F02",  # orange
  "#7570B3",  # violet
  "#E7298A"   # magenta
)

# === Plot 1: Mean Objective Value ===
mean_plot <- ggplot(all_results, aes(x = row, y = mean_y, color = method)) +
  geom_line(size = 0.7, alpha = 0.9, lineend = "round") +
  scale_color_manual(values = cb_palette) +
  labs(x = "Iteration", y = "Mean Objective Value",
       title = "Mean Objective Value Across Methods") +
  theme_minimal(base_size = 14) +
  theme(
    legend.title     = element_blank(),
    legend.position  = "right",
    legend.text      = element_text(size = 11),
    plot.title       = element_text(size = 16, hjust = 0.5),
    axis.text        = element_text(size = 12),
    panel.grid.minor = element_blank()
  )

# === Plot 2: Best Objective Value — zoomed to first 50 iterations ===
best_results <- all_results %>%
  group_by(method) %>%
  arrange(row) %>%
  mutate(best_y = cummin(mean_y))

best_plot <- ggplot(best_results, aes(x = row, y = best_y, color = method)) +
  geom_line(size = 0.7, alpha = 0.9, lineend = "round") +
  scale_color_manual(values = cb_palette) +
  coord_cartesian(xlim = c(1, 50)) +
  labs(x = "Iteration", y = "Best Objective Value",
       title = "Optimisation traces of methods") +
  theme_minimal(base_size = 14) +
  theme(
    legend.title     = element_blank(),
    legend.position  = "right",
    legend.text      = element_text(size = 11),
    plot.title       = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.text        = element_text(size = 12),
    panel.grid.minor = element_blank()
  )
ggsave(
  filename = "/Users/niyati/Projects:Codes/VaR-CoBO//TestPrblms/MaxSAT/maxsat60_mean_objective_methods.pdf",
  plot     = mean_plot,
  device   = cairo_pdf,
  width    = 11, height = 6, units = "in",
  dpi      = 1200, bg = "white"
)

ggsave(
  filename = "/Users/niyati/Projects:Codes/PhD-Read:Write/Chapter 1/Plots/best_objective_maxsat60.pdf",
  plot     = best_plot,
  device   = cairo_pdf,
  width    = 11, height = 6, units = "in",
  dpi      = 1200, bg = "white"
)

# --- Summary table ---
summary_table <- all_results %>%
  group_by(method) %>%
  summarise(
    Mean  = mean(mean_y),
    SD    = sd(mean_y),
    Best  = max(mean_y),
    Worst = min(mean_y)
  ) %>%
  mutate(across(where(is.numeric), ~ round(., 2)))

xt <- xtable(summary_table,
             caption = "Summary of objective values across optimization methods.",
             label   = "tab:summary_methods",
             align   = c("l", "l", "r", "r", "r", "r"))

print(xt,
      include.rownames       = FALSE,
      sanitize.text.function = identity,
      comment                = FALSE,
      booktabs               = TRUE,
      caption.placement      = "top")

# ============================================================
# --- CI Grid Plot — zoomed to first 50 iterations
# ============================================================
methods <- names(summary_list)

ci_plots <- lapply(seq_along(methods), function(i) {
  method_name <- methods[i]
  method_data <- summary_list[[i]]
  df <- get_mean_df(method_data, method_name)
  
  ggplot(df, aes(x = row, y = mean_y)) +
    geom_line(color = cb_palette[i], size = 0.6) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
                fill = cb_palette[i], alpha = 0.2) +
    coord_cartesian(xlim = c(1, 50)) +            # zoom to first 50 iterations
    labs(title = method_name, x = "Iteration", y = "Mean Objective Value") +
    theme_minimal(base_size = 9) +
    theme(
      plot.title       = element_text(size = 9, face = "bold", hjust = 0.5),
      axis.title       = element_text(size = 8),
      axis.text        = element_text(size = 7),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.2, colour = "grey85")
    )
})

ci_grid_plot <- wrap_plots(ci_plots, ncol = 2)   # 4 methods → 2x2 grid

ggsave(
  filename = "/Users/niyati/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/n_vars=60/ci_plots_grid_maxsat60.pdf",
  plot     = ci_grid_plot,
  device   = cairo_pdf,
  width    = 11, height = 6, units = "in",
  dpi      = 1200, bg = "white"
)

# ============================================================
# --- Cummin CI Grid Plot — zoomed to first 50 iterations
# ============================================================
get_cummin_df <- function(method_list, method_name) {
  valid_list <- Filter(function(x) {
    !is.null(x) && is.data.frame(x) && "y" %in% names(x)
  }, method_list)
  
  if (length(valid_list) == 0) return(NULL)
  
  num_rows   <- nrow(prbocs_data[[1]])
  cummin_mat <- sapply(valid_list, function(df) cummin(df$y))
  
  y_means      <- rowMeans(cummin_mat)
  y_stderr     <- apply(cummin_mat, 1, sd) / sqrt(ncol(cummin_mat))
  error_margin <- qt(0.975, df = ncol(cummin_mat) - 1) * y_stderr
  
  data.frame(
    row      = 1:num_rows,
    mean_y   = y_means,
    ci_lower = y_means - error_margin,
    ci_upper = y_means + error_margin,
    method   = method_name
  )
}

cummin_plots <- lapply(seq_along(methods), function(i) {
  method_name <- methods[i]
  method_data <- summary_list[[i]]
  df <- get_cummin_df(method_data, method_name)
  
  ggplot(df, aes(x = row, y = mean_y)) +
    geom_line(color = cb_palette[i], size = 0.6) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
                fill = cb_palette[i], alpha = 0.2) +
    coord_cartesian(xlim = c(1, 50)) +            # zoom to first 50 iterations
    labs(title = method_name, x = "Iteration", y = "Best Objective Value") +
    theme_minimal(base_size = 9) +
    theme(
      plot.title       = element_text(size = 9, face = "bold", hjust = 0.5),
      axis.title       = element_text(size = 8),
      axis.text        = element_text(size = 7),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.2, colour = "grey85")
    )
})

cummin_grid_plot <- wrap_plots(cummin_plots, ncol = 2)   # 4 methods → 2x2 grid

# Base R title workaround
pdf(
  file   = "/Users/niyati/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/n_vars=60/cummin_ci_plots_grid_maxsat60.pdf",
  width  = 11,
  height = 6
)
print(cummin_grid_plot)
title("Optimisation traces of different methods",
      cex.main = 1.2, font.main = 2, line = -1)
dev.off()

# --- Combined cummin plot — zoomed to first 50 iterations ---
cummin_all <- do.call(rbind, lapply(seq_along(methods), function(i) {
  get_cummin_df(summary_list[[i]], methods[i])
}))

cummin_all$method <- factor(cummin_all$method, levels = methods)

cummin_combined_plot <- ggplot(cummin_all, aes(x = row, y = mean_y, color = method, fill = method)) +
  geom_line(size = 0.7, alpha = 0.9, lineend = "round") +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.12, color = NA) +
  scale_color_manual(values = cb_palette) +
  scale_fill_manual(values = cb_palette) +
  coord_cartesian(xlim = c(1, 50)) +              # zoom to first 50 iterations
  labs(x = "Iteration", y = "Best Objective Value") +
  theme_minimal(base_size = 14) +
  theme(
    legend.title     = element_blank(),
    legend.position  = "right",
    legend.text      = element_text(size = 11),
    axis.text        = element_text(size = 12),
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = "/Users/niyati/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/n_vars=60/cummin_ci_combined_maxsat60.pdf",
  plot     = cummin_combined_plot,
  device   = cairo_pdf,
  width    = 9, height = 5, units = "in",
  dpi      = 1200, bg = "white"
)

#-----------------------------
# Computational efficiency
#-----------------------------

prbocs_time    <- list()
prbocs_vb_time <- list()
prbocs_vb_ga_time <- list()
bocs_sa_time   <- list()

for(i in 1:num_instances){
  prbocs_time[[i]]       <- vd[[i]]$PRBOCS$time_taken
  prbocs_vb_time[[i]]    <- vd[[i]]$PRBOCS_VB$time_taken
  prbocs_vb_ga_time[[i]] <- vd[[i]]$PRBOCS_VB_GA$time_taken
  bocs_sa_time[[i]]      <- vd[[i]]$BOCS_SA$time_taken
}

# UPDATED: correct names and order to match paper
time_list <- list(
  `BOCS-SA`      = bocs_sa_time,
  `PRCBO-BFGS`   = prbocs_time,
  `VaR-CBO-BFGS` = prbocs_vb_time,
  `VaR-CBO-GA`   = prbocs_vb_ga_time
)

time_to_hours   <- function(x) as.numeric(x, units = "hours")
time_to_minutes <- function(x) as.numeric(x, units = "mins")

time_df <- do.call(rbind, lapply(names(time_list), function(name) {
  times     <- sapply(time_list[[name]], time_to_hours)
  mean_time <- mean(times, na.rm = TRUE)
  data.frame(
    Method = name,
    `Time (hrs)` = sprintf("%.2f", mean_time),
    check.names = FALSE
  )
}))

print(
  xtable(time_df,
         caption = "Computation Time (in hours) for Each Method",
         label   = "tab:time_results"),
  include.rownames       = FALSE,
  sanitize.text.function = identity
)

# Boxplot
boxplot_data <- do.call(rbind, lapply(names(time_list), function(name) {
  times <- sapply(time_list[[name]], time_to_minutes)
  data.frame(Method = name, Time_minutes = times)
}))

# UPDATED: factor levels reflect new order
boxplot_data$Method <- factor(boxplot_data$Method, levels = names(time_list))

p <- ggplot(boxplot_data, aes(x = Method, y = Time_minutes, fill = Method)) +
  geom_boxplot(outlier.shape = 1, outlier.size = 2, outlier.alpha = 0.7) +
  theme_minimal(base_size = 14) +
  labs(
    title = "Execution Time Comparison (Log Scale)",
    x     = "Method",
    y     = "Time (minutes, log scale)"
  ) +
  theme(
    legend.position = "none",
    plot.title      = element_text(size = 16, face = "bold"),
    axis.title      = element_text(size = 14),
    axis.text.x     = element_text(angle = 45, hjust = 1, size = 12)
  ) +
  scale_fill_manual(values = cb_palette) +
  scale_y_log10() +
  annotation_logticks(sides = "l")

ggsave(
  filename = "/Users/niyati/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/n_vars=60/time_box_maxsat60.pdf",
  plot     = p,
  device   = cairo_pdf,
  width    = 9, height = 5, units = "in",
  dpi      = 1200, bg = "white"
)