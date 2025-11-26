# Test problem for a combinatorial optimisation
# PRBOCS method is used to solve the problem here.
# Using sparve variational bayes to estimate the coefficients
# of the linear regression
# Using optim here to solve PRThomp 
# Fixed the theta interaction calculation to theta_i.theta_j from theta_ij
# This code is for the CONTAMINATION test problem in BOCS

# Author: Niyati Seth
# Date:  July 2025
# ---------------------------------------------------------
# LOAD Functions and Libraries
# ---------------------------------------------------------
seed <- 20
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/NAS/sample_models.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/NAS/NAS_mnist_fn.R")
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

# ---------------------------------------------------------
# ORDER EFFECTS/THETA INTERACTION - to account interaction
# ---------------------------------------------------------
# # Function to generate interaction terms up to a given order
order_effects <- function(xTrain, order) {
  # Find dimensions of the matrix
  n_samp <- nrow(xTrain)
  n_vars <- ncol(xTrain)
  
  # Initialize result matrix
  xTrain_in <- xTrain
  
  # Generate interaction terms for each order up to ord_t
  for (ord_i in 2:order) {
    # Generate all combinations of indices (without diagonals)
    # cat("ord_i", ord_i, "\n")
    # cat("n_vars", n_vars, "\n")
    offdProd <- combn(n_vars, ord_i)
    
    # Generate products of input variables
    x_comb <- array(dim = c(n_samp, ncol(offdProd)))
    for (j in 1:ncol(offdProd)) {
      x_comb[, j] <- apply(xTrain[, offdProd[, j], drop = FALSE], 1, prod)
    }
    
    # Append interaction terms to the result matrix
    xTrain_in <- cbind(xTrain_in, x_comb)
  }
  
  return(list(xTrain_in = xTrain_in, combos = offdProd))
}

theta_interaction <- function(theta, order, n_vars){
  
  # Initialize result matrix
  theta <- matrix(theta, nrow = 1, ncol = n_vars)
  theta_in <- theta
  
  for (ord_i in 2:order) {
    # Generate all combinations of indices (without diagonals)
    # cat("ord_i", ord_i, "\n")
    # cat("n_vars", n_vars, "\n")
    offdProd <- combn(n_vars, ord_i)
    # Generate products of input variables
    theta_comb <- array(dim = c(1, ncol(offdProd)))
    for (j in 1:ncol(offdProd)) {
      theta_comb[, j] <- apply(theta[, offdProd[, j], drop = FALSE], 1, prod)
    }
    
    # Append interaction terms to the result matrix
    theta_in <- cbind(theta_in, theta_comb)
  }
  
  return(theta_in)
}

# ---------------------------------------------------------
# ACQUISITION FUNCTION
# ---------------------------------------------------------

# Function to compute thompson sampling
thompson_sam <- function(theta_current, vb_model, duplicate_cols, vb_data, order){
  
  # browser()
  coeffs <- list()
  
  #Create a full mu vector of correct length
  full_mu <- numeric(ncol(vb_data))  # Initialize with zeros

  # Identify non-duplicate (i.e., retained) columns
  kept_cols <- setdiff(seq_along(full_mu), duplicate_cols)
  
  #Fill in mu values from reduced model
  full_mu[kept_cols] <- vb_model$mu
  
  # Copy values for removed (duplicate) columns
  for (col in duplicate_cols) {
    dup_column <- vb_data[, col]
    
    # Find the original column that matches the binary values of the duplicate column
    original_col <- which(apply(vb_data, 2, function(x) all(x == dup_column)) & !(seq_along(vb_data) %in% duplicate_cols))
    
    if (length(original_col) == 1) {
      full_mu[col] <- full_mu[original_col]
    } else {
      warning(paste("Could not uniquely match duplicate column:", names(vb_data)[col]))
    }
  }
  # Add a column of 1s to 'theta_current_in' to account for the intercept
  theta_current_in <- theta_interaction(theta_current, order, n_vars)
  theta_current_in <- c(1, theta_current_in)
  
  coeffs <- full_mu 
  coeffs <- c(vb_model$intercept, coeffs)
  # cat("Estimated coeffs", coeffs, "\n")
  y_pred <-  sum(theta_current_in * coeffs)
  
  return(y_pred = y_pred)
}

# ---------------------------------------------------------
# SET INPUTS
# ---------------------------------------------------------
evalBudget <-100
n_init <- 10
lambda <- .2
minSpend <- 30
order <- 2

node_sizes <- c(16, 32, 64)
L <- 3
d <- L * (length(node_sizes) + 1)
n_vars <- d
set.seed(seed)
  
print("-----------------")
print("SEED")
print(seed)
print("-----------------")
  
# ---------------------------------------------------------
# INITIAL SAMPLES FOR STATISTICAL MODELS
# ---------------------------------------------------------
  X <- sample_models(n_init, d)
  # Evaluate loss
  NN <- evaluate_binary_architectures(X, mode = "combined", alpha = 1, beta = 1)
  
  x_vals <- X
  y_vals <- NN$score
  
  x_vals
  y_vals
  # ---------------------------------------------------------
  # DEFINE TRUE MODEL
  # ---------------------------------------------------------
  model <- function(x_vals, seed){
    para <- evaluate_binary_architectures(x_vals, mode = "combined", alpha = 1, beta = 1) 
    return(para$score)
  }
  
  # ---------------------------------------------------------
  # RUN PROCS
  # ---------------------------------------------------------
  
  # Find number of iterations based on total budget
  n_init <- nrow(x_vals)
  n_iter <- evalBudget - n_init
  
  # Train initial statistical model
  # setup data for training
  xTrain <- x_vals
  yTrain <- y_vals
  
  # Generate interaction terms for the input data
  xTrain_in_comb <- order_effects(xTrain, order)
  xTrain_in <- xTrain_in_comb$xTrain_in
  inter_combos <- xTrain_in_comb$combos
  # cat("xTrain with interaction terms","\n")
  # print(xTrain_in)
  
  cDims <- dim(xTrain_in)
  nSamps <- cDims[1]
  nCoeffs <- cDims[2]
  # cat("nSamps, nCoeffs", c(nSamps, nCoeffs), "\n")
  
  hs_ss_sd <- sd(xTrain)
  n <- nrow(xTrain_in)
  p <- ncol(xTrain_in)
  
  #Setup dataframe for stan_glam training 
  data <- data.frame(y=y_vals , xTrain_in)
  vb_data <- data[,-1]
  
  # Initialize a data frame to store iteration results
  optim_result <- matrix(0, evalBudget, n_vars)
  
  # browser()
  theta_current <-  rep(0.5, ncol(xTrain))
  
  # Find duplicate columns
  duplicate_cols <- which(duplicated(as.list(vb_data)))
  
  # Save the removed columns in a separate data frame
  removed_columns <- vb_data[, duplicate_cols, drop = FALSE]
  
  # Keep only unique columns
  data_reduced <- vb_data[, !duplicated(as.list(vb_data))]
  
  # Prepare data
  X <- as.matrix(data_reduced)  # Assuming the first column is the response variable
  Y <- data[, 1]              # Response variable
  
  # Fit the variational Bayesian model
  vb_model <- svb.fit(
    X = X,
    Y = Y,
    family = "linear",     # For linear regression
    slab = "laplace",      # Default slab prior
    intercept = TRUE       # Include intercept in the model
  )
  
  for (t in 1:evalBudget) {
    x_current <- rbinom(n = n_vars, size = 1, prob = theta_current)
    
    stat_model <- function(theta) {
      thompson_sam(theta, vb_model = vb_model, duplicate_cols, vb_data, order)
    }
    
    
    min_acq <- optim(theta_current, stat_model, method='L-BFGS-B', lower=1e-8, upper=0.99999)
    
    expected_val <- min_acq$par
    cat("expected_val", expected_val, "\n")
    x_new <- rbinom(length(expected_val), 1, expected_val)
    cat("New evaluation point", x_new, "\n")
    
    # browser()
    #Append new point to existing x_vals
    x_vals_updated <- rbind(xTrain, x_new)
    # Evaluate model objective at the new evaluation point
    x_new <- matrix(x_new, nrow = 1, ncol = n_vars)
    y_new <- model(x_new, seed)
    
    x_new_in_comb <- order_effects(x_new, order)
    x_new_in <- x_new_in_comb$xTrain_in
    
    data_new <- data.frame(y = y_new, x_new_in)
    data <- rbind(data, data_new)
    
    theta_current <- expected_val
    
    optim_result[t,] <- expected_val
    
    vb_data <- data[,-1]
    
    # Find duplicate columns
    duplicate_cols <- which(duplicated(as.list(vb_data)))
    
    # Save the removed columns in a separate data frame
    removed_columns <- vb_data[, duplicate_cols, drop = FALSE]
    
    # Keep only unique columns
    data_reduced <- vb_data[, !duplicated(as.list(vb_data))]
    
    # Prepare data
    X <- as.matrix(data_reduced)  # Assuming the first column is the response variable
    Y <- data[, 1]              # Response variable
    
    # Fit the variational Bayesian model
    vb_model <- svb.fit(
      X = X,
      Y = Y,
      family = "linear",     # For linear regression
      slab = "laplace",      # Default slab prior
      intercept = TRUE       # Include intercept in the model
    )
    
  }
  
}
# ---------------------------------------------------------
# PLOTS
# ---------------------------------------------------------
# 
# # Plot objective function values versus iterations for normal optimization
# plot(1:nrow(data), data$y , type = "l", xlab = "Iterations", 
#      ylab = "Objective Function Value", main = "Objective Function vs Iterations", 
#      col = "red", xlim = c(1, nrow(data)))

