# Test file to debug and run SVB (Sparse Variational Bayes) on Ising sparsification problem

# Load required libraries and source code
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/NAS/sample_models.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/Ising/ising.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/NAS/thompson_svb.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/NAS/ordertheta_interaction.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/Ising/prbocs_vb.R")


library(GA)
library(psych)
library(lpSolve)
library(ompr)
library(ompr.roi)
library(ggplot2)
library(dplyr)
library(ROI)
library(GPfit)
library(rstanarm)
library(rstan)
library(sparsevb)
library(bayesplot)

# source("~/Projects:Codes/PhD-Compute/R code/SparseVB/selectiveInference/R/funs.common.R")
# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

evalBudget <- 100
n_init <- 10
order <- 2
instances <- 1

# Fix Theta_P once - same Ising interaction matrix for all instances
# for 4x4 grid
theta_P <- rand_ising_grid(16)
moments <- ising_model_moments(Theta_P)
edges <- get_edge_list(Theta_P)  # 24 edges for 4x4
n_vars <- nrow(edges)

# Placeholder: Initialize theta_current (used inside prbocs_vb_optim)
# This should be initialized properly based on your model dimensions
# For example, zero vector or random initialization
theta_current <- rep(0, n_vars)  # Adjust size if needed

# ---------------------------------------------------------
# Define True Objective Model
# ---------------------------------------------------------
model <- function(x_vals, Theta_P) {
  if (is.vector(x_vals)) {
    x_vals <- matrix(x_vals, nrow = 1)
  }
  apply(x_vals, 1, function(x) KL_divergence_ising(Theta_P, moments, edges, matrix(x, nrow=1)))
}

# ---------------------------------------------------------
# Run experiments for each instance
# ---------------------------------------------------------
results <- list()

for(inst in seq_len(instances)) {
  cat("\nRunning instance:", inst, "\n")
  set.seed(1234 + inst)  # Different seed per instance for reproducibility
  
  starttime <- Sys.time()
  
  # Initial samples for training
  x_vals <- sample_models(n_init, n_vars)
  y_vals <- model(x_vals, theta_P)
  
  cat("Initial sample x values:\n")
  print(x_vals)
  cat("Initial sample y values:\n")
  print(y_vals)
  
  # Calculate iterations left after initial samples
  n_init_actual <- nrow(x_vals)
  n_iter <- evalBudget - n_init_actual
  
  # Prepare training data
  xTrain <- x_vals
  yTrain <- y_vals
  
  # Generate interaction terms of specified order
  xTrain_in_comb <- order_effects(xTrain, order)
  xTrain_in <- xTrain_in_comb$xTrain_in
  inter_combos <- xTrain_in_comb$combos
  
  cat("Training inputs with interaction terms:\n")
  print(xTrain_in)
  
  # Dimensions
  nSamps <- nrow(xTrain_in)
  nCoeffs <- ncol(xTrain_in)
  cat("Samples:", nSamps, "Coefficients:", nCoeffs, "\n")
  
  # Setup dataframe for stan_glm or VB training
  data <- data.frame(y = yTrain, xTrain_in)
  
  # Run Sparse VB optimizer
  prbocs_vb_result <- prbocs_vb_optim(
    data = data,
    evalBudget = evalBudget,
    n_iter = n_iter,
    n_vars = n_vars,
    xTrain = xTrain,
    xTrain_in = xTrain_in,
    theta_current = theta_current,
    order = order
  )
  
  endtime <- Sys.time()
  elapsed_time <- endtime - starttime
  cat("Instance", inst, "completed in", elapsed_time, "seconds.\n")
  
  print(prbocs_vb_result)
  
  results[[inst]] <- list(
    instance_id = inst,
    prbocs_vb_result = prbocs_vb_result,
    time_taken = elapsed_time
  )
}

# Optional: Save results
# folder_name <- paste0("n_vars=", n_vars)
# if (!dir.exists(folder_name)) dir.create(folder_name)
# saveRDS(results, file = file.path(folder_name, "simulation_prbocsvb_results.RData"))
