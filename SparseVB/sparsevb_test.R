# Test problem for a combinatorial optimisation
# PRBOCS method is used to solve the problem here.
# Approaching the constraints by PRing them before sampling
# Including the code for lpSolve/ompr (Linear Programming) and 
# GA (genetic algorithm) for comparison and true answer
# Edit to restart the gibbs sampler from the averages of the previous iteration
# Modification of the BOCS code 
# The alpha values are sampled from a bayesian regression with a normal distribution prior
# Using PR_Thompson Sampling Here
# The posterior predictive is derived from STAN heirarchal shrinkage
# Addition of constraints in the AcqFun itself is done
# Tested here by simply using a deterministic constraints here (over budget)
# Using optim here to solve PRThomp - doesn't work
# Adding the method my Piironen and Vehtari on hyper prior chaice in heirarchal shrinkage
# Fixed the theta interaction calculation to theta_i.theta_j from theta_ij
# Added a forced restriction so that optim doesn't give zero results
# Removing the main function of PRBOCS and running it sequentially
# Restructuring the code to constrain before running the stan_glm
# This code is for the CONTAMINATION test problem in BOCS

# Author: Niyati Seth
# Date:  March 2025
# ---------------------------------------------------------
# LOAD Functions and Libraries
# ---------------------------------------------------------
devtools::load_all("/Users/niyati/Projects:Codes/PhD-Compute/R code/SparseVB/selectiveInference")

source("~/Projects:Codes/PhD-Compute/R code/SparseVB/sample_models.R")
source("~/Projects:Codes/PhD-Compute/R code/SparseVB/Contamination.R")
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
# library(selectiveInference)
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
thompson_sam <- function(theta_current, mu, intercept, order){
  # browser()
  coeffs <- list()

  # Add a column of 1s to 'theta_current_in' to account for the intercept
  theta_current_in <- theta_interaction(theta_current, order, n_vars)
  theta_current_in <- c(1, theta_current_in)
  
  coeffs <- mu
  coeffs <- c(intercept, coeffs)
  # cat("Estimated coeffs", coeffs, "\n")
  y_pred <-  sum(theta_current_in * coeffs)
  return(y_pred = y_pred)
}

# ---------------------------------------------------------
# SET INPUTS
# ---------------------------------------------------------
n_vars <-50
evalBudget <-100
n_init <- 30
lambda <- .2
minSpend <- 30
order <- 2
seed <- 1


# ---------------------------------------------------------
# INITIAL SAMPLES FOR STATISTICAL MODELS
# ---------------------------------------------------------
x_vals <- sample_models(n_init, n_vars)
y_vals <- contamination_prob(x_vals, 100, seed)

x_vals
y_vals

# ---------------------------------------------------------
# DEFINE TRUE MODEL
# ---------------------------------------------------------
model <- function(x_vals, seed){
  contamination_prob(x_vals, 100, seed) 
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
X <- xTrain_in
y <- y_vals

#Create a dataframe
data <- data.frame(y=y , X)

# Initialize a data frame to store iteration results
optim_result <- matrix(0, evalBudget, n_vars)

# browser()
theta_ini <- rep(0.5, ncol(xTrain))
theta_current <- theta_ini
theta_current_in <- theta_interaction(theta_current, order, n_vars)

# #Initializing the bayesian model with the given data first
# # Check for constant columns
# # Remove constant columns
# is_constant <- apply(data, 2, function(x) length(unique(x)) == 1)
# data_reduced <- data[, !is_constant]
# # Save the names of the columns that were removed
# removed_columns <- names(data)[is_constant]

dataX <- data[,-1]

# Find duplicate columns
dup_cols <- which(duplicated(as.list(dataX)))

# Save the removed columns in a separate data frame
dup_cols_data <- dataX[, dup_cols, drop = FALSE]

# Keep only unique columns
data_unique <- dataX[, !duplicated(as.list(dataX))]

# Prepare data
X <- as.matrix(data_unique)  # Assuming the first column is the response variable
Y <- y              # Response variable

# Fit the variational Bayesian model
vb_model <- svb.fit(
  X = X,
  Y = Y,
  family = "linear",     # For linear regression
  slab = "laplace",      # Default slab prior
  intercept = TRUE       # Include intercept in the model
)
# Extract the estimated coefficients
mu <- rep(NA, ncol(dataX))
  
for (i in 1:ncol(data_unique)) {
  col_name <- colnames(data_unique)[i]
  
  # Find columns in dataX that match the current column of data_unique
  matching_columns <- grep(col_name, colnames(dataX), value = TRUE)
  
  # If there are matching columns in dataX
  if (length(matching_columns) > 0) {
    # Loop through each matching column in dataX
    for (matching_col in matching_columns) {
      # Find the index of the matching column in dataX
      match_index <- which(colnames(dataX) == matching_col)
      
      # Assign the corresponding mu value to this column in mu1
      mu[match_index] <- vb_model$mu[i]
      # gamma[match_index] <- vb_model$gamma[i]
      }
    }
  }
  
intercept <- vb_model$intercept
cat("intercept", intercept, "\n")

for (t in 1:evalBudget) {
  x_current <- rbinom(n = n_vars, size = 1, prob = theta_current)
  stat_model <- function(theta) {
    thompson_sam(theta_current = theta, mu = mu, intercept = intercept, order)
  }
  
  min_acq <- optim(theta_current, stat_model, method='L-BFGS-B', lower=1e-8, upper=0.99999999)
  
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
  
  dataX <- data[,-1]
  
  # Find duplicate columns
  dup_cols <- which(duplicated(as.list(dataX)))
  
  # Save the removed columns in a separate data frame
  dup_cols_data <- dataX[, dup_cols, drop = FALSE]
  
  # Keep only unique columns
  data_unique <- dataX[, !duplicated(as.list(dataX))]
  
  # Prepare data
  X <- as.matrix(data_unique)  # Assuming the first column is the response variable
  Y <- data[,1]            # Response variable
  
  # Fit the variational Bayesian model
  vb_model <- svb.fit(
    X = X,
    Y = Y,
    family = "linear",     # For linear regression
    slab = "laplace",      # Default slab prior
    intercept = TRUE       # Include intercept in the model
  )
  # Extract the estimated coefficients
  mu <- rep(NA, ncol(dataX))
  
  for (i in 1:ncol(data_unique)) {
    col_name <- colnames(data_unique)[i]
    
    # Find columns in dataX that match the current column of data_unique
    matching_columns <- grep(col_name, colnames(dataX), value = TRUE)
    
    # If there are matching columns in dataX
    if (length(matching_columns) > 0) {
      # Loop through each matching column in dataX
      for (matching_col in matching_columns) {
        # Find the index of the matching column in dataX
        match_index <- which(colnames(dataX) == matching_col)
        
        # Assign the corresponding mu value to this column in mu1
        mu[match_index] <- vb_model$mu[i]
        # gamma[match_index] <- vb_model$gamma[i]
      }
    }
  }
  
  intercept <- vb_model$intercept
  cat("intercept", intercept, "\n")
}

# ---------------------------------------------------------
# PLOTS
# ---------------------------------------------------------

# Plot objective function values versus iterations for normal optimization
plot(1:nrow(data), data$y , type = "l", xlab = "Iterations", 
     ylab = "Objective Function Value", main = "Objective Function vs Iterations", 
     col = "red", xlim = c(1, nrow(data)))
