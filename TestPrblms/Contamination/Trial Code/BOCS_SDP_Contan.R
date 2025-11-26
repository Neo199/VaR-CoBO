# Test problem for a combinatorial optimisation
# BOCS method is used to solve the problem here.
# Including the code for lpSolve/ompr (Linear Programming) and 
# Edit to restart the gibbs sampler from the averages of the previous iteration
# Modification of the BOCS code 
# The alpha values are sampled from a bayesian regression with a normal distribution prior
# Using Thompson Sampling solved using GA
# The posterior predictive is derived from STAN hierarchical shrinkage
# Addition of constraints in the AcqFun itself is done
# Tested here by simply using a deterministic constraints here (over budget)
# Adding the method my Piironen and Vehtari on hyper prior chaice in hierarchical shrinkage
# Fixed the theta interaction calculation to theta_i.theta_j from theta_ij
# Added a forced restriction so that optim doesn't give zero results
# Restructuring the code to constrain before running the stan_glm
# No constraints are actually included in the solver(only in the orignal contamination model)

# Author: Niyati Seth
# Date:  January 2025
# ---------------------------------------------------------
# LOAD Functions and Libraries
# ---------------------------------------------------------

source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/Contamination/sample_models.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/Contamination/Contamination.R")
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
library(bayesplot)
library(CVXR)

set.seed(1)

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
thompson_sam <- function(x_current, bayesian_model, removed_columns, data, order){
  
  # browser()
  coeffs <- list()
  
  # Add the removed columns back to the reduced data, filling with the constant values
  for (col in removed_columns) {
    unique_value <- unique(data[[col]])  # Using the unique value from the original data
    
    # Check if there's exactly one unique value to use as a constant
    if (length(unique_value) == 1) {
      # Add the column back to reduced_data, filling with the constant value
      bayesian_model$coefficients[[col]] <- unique_value
    } 
    else {
      stop(paste("Column", col, "does not have a single unique value. Cannot fill with a constant."))
    }
    
  }
  
  # Add a column of 1s to 'theta_current_in' to account for the intercept
  x_current_in <- theta_interaction(x_current, order, n_vars)
  x_current_in <- c(1, x_current_in)
  
  coeffs <- bayesian_model$coefficients
  y_pred <-  sum(x_current_in * coeffs)
  
  return(y_pred = y_pred)
}

# ---------------------------------------------------------
# SEMIDEFINITE PROGRAMMING
# ---------------------------------------------------------
sdp_relaxation <- function(alpha, n_vars, lambda, removed_columns) {
  
  # Append missing variables back with zero coefficients
  alpha_vect <- c(alpha, setNames(rep(0, length(removed_columns)), removed_columns))
  
  # Reorder so that variables are in a consistent order
  alpha_vect <- alpha_vect[order(names(alpha_vect))]
  
  # Extract coefficients
  b <- alpha_vect[2:(n_vars + 1)] + lambda
  a <- alpha_vect[(n_vars + 2):length(alpha_vect)]
  
  # Create index for upper triangle (i < j)
  idx_prod <- t(combn(n_vars, 2))
  n_idx <- nrow(idx_prod)
  
  if (length(a) != n_idx) {
    stop("Number of Coefficients does not match the number of off-diagonal terms!")
  }
  
  # Convert a to symmetric matrix A
  A <- matrix(0, n_vars, n_vars)
  for (i in 1:n_idx) {
    A[idx_prod[i, 1], idx_prod[i, 2]] <- a[i] / 2
    A[idx_prod[i, 2], idx_prod[i, 1]] <- a[i] / 2
  }
  
  # Construct the lifted matrix At (like in SDP)
  bt <- b / 2 + A %*% rep(1, n_vars) / 2
  At <- rbind(
    cbind(A / 4, bt / 2),
    c(t(bt) / 2, 0)
  )
  
  # Approximate PSD matrix X by projecting At onto PSD cone
  eig <- eigen(At, symmetric = TRUE)
  eig$values[eig$values < 0] <- 1e-12  # clip negative eigenvalues
  X_val <- eig$vectors %*% diag(eig$values) %*% t(eig$vectors)
  
  # Cholesky-like factorization
  eig_X <- eigen(X_val, symmetric = TRUE)
  eig_X$values[eig_X$values < 0] <- 0
  L <- eig_X$vectors %*% diag(sqrt(eig_X$values))
  
  # Randomized rounding
  n_rand_vector <- 100
  model_vect <- matrix(0, n_vars, n_rand_vector)
  obj_vect <- numeric(n_rand_vector)
  
  for (kk in 1:n_rand_vector) {
    r <- rnorm(n_vars + 1)
    r <- r / sqrt(sum(r^2))
    y_proj <- t(L) %*% r
    y_soln <- ifelse(y_proj >= 0, 1, -1)
    x_bin <- (y_soln[1:n_vars] + 1) / 2
    model_vect[, kk] <- x_bin
    obj_vect[kk] <- t(x_bin) %*% A %*% x_bin + sum(b * x_bin)
  }
  
  opt_idx <- which.min(obj_vect)
  model <- as.numeric(model_vect[, opt_idx])
  obj <- obj_vect[opt_idx]
  
  return(list(model = model, obj = obj))
}




# sdp_relaxation <- function(alpha, n_vars, lambda, removed_columns){
#   browser()
#   # Append missing variables back with zero coefficients
#   alpha_full <- c(alpha, setNames(rep(0, length(removed_columns)), removed_columns))
# 
#   # Reorder so that variables are in a consistent order
#   alpha_full <- alpha_full[order(names(alpha_full))]
# 
#   # Extract coefficients
#   intercept <- alpha_full[1]
#   b <- alpha_full[2:(n_vars + 1)] + lambda  # Linear terms + lambda
# 
#   # Determine number of quadratic terms
#   n_off_diag <- choose(n_vars, 2)
#   n_diag <- n_vars
#   n_quad <- n_off_diag + n_diag
# 
#   # Extract all quadratic coefficients (off-diagonal + diagonal)
#   a <- alpha_full[(n_vars + 2):(n_vars + 1 + n_quad)]
# 
#   # Build index matrix for quadratic terms
#   off_diag <- t(combn(n_vars, 2))
#   diag_idx <- cbind(1:n_vars, 1:n_vars)
#   idx_prod <- rbind(off_diag, diag_idx)
#   n_idx <- nrow(idx_prod)
# 
#   # Check number of coefficients
#   if (length(a) != n_idx) {
#     stop("Number of coefficients does not match indices!")
#   }
# 
#   # Convert a to matrix form
#   A <- matrix(0, n_vars, n_vars)
#   for (i in 1:n_idx) {
#     A[idx_prod[i, 1], idx_prod[i, 2]] <- a[i] / 2
#     A[idx_prod[i, 2], idx_prod[i, 1]] <- a[i] / 2
#   }
# 
#   # Convert to standard form
#   bt <- b / 2 + A %*% rep(1, n_vars) / 2
#   At <- rbind(cbind(A / 4, bt / 2), c(bt / 2, 0))
# 
#   # SDP approximation using eigendecomposition
#   eig <- eigen(At, symmetric = TRUE)
#   eig$values[eig$values < 0] <- 1e-15
#   X <- eig$vectors %*% diag(eig$values) %*% t(eig$vectors)
# 
#   # Cholesky decomposition
#   L <- chol(X)
# 
#   # Rounding
#   n_rand_vector <- 100
#   model_vect <- matrix(0, nrow = n_vars, ncol = n_rand_vector)
#   obj_vect <- numeric(n_rand_vector)
# 
#   for (kk in 1:n_rand_vector) {
#     r <- rnorm(n_vars + 1)
#     r <- r / sqrt(sum(r^2))
#     y_soln <- sign(t(L) %*% r)
#     model_vect[, kk] <- (y_soln[1:n_vars] + 1) / 2
#     obj_vect[kk] <- t(model_vect[, kk]) %*% A %*% model_vect[, kk] + sum(b * model_vect[, kk]) + intercept
#   }
# 
#   # Best solution
#   opt_idx <- which.min(obj_vect)
#   model <- model_vect[, opt_idx]
#   obj <- obj_vect[opt_idx]
# 
#   return(list(model = model, obj = obj))
# }
# 

# ---------------------------------------------------------
# SET INPUTS
# ---------------------------------------------------------
n_vars <-3
evalBudget <- 10
n_init <- 5
lambda <- 1e-2
costBudget <- 100
minSpend <- 30
order <- 2
seed <- 1

# ---------------------------------------------------------
# DEFINE TRUE MODEL
# ---------------------------------------------------------
model <- function(x_vals, seed){
  contamination_prob(x_vals, 100, seed) 
}

# ---------------------------------------------------------
# INITIAL SAMPLES FOR STATISTICAL MODELS
# ---------------------------------------------------------
x_vals <- sample_models(n_init, n_vars)
y_vals <- contamination_prob(x_vals, 100, seed)

x_vals
y_vals


# ---------------------------------------------------------
# RUN BOCS
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


#Initializing the bayesian model with the given data first
# Check for constant columns
# Remove constant columns
is_constant <- apply(data, 2, function(x) length(unique(x)) == 1)
data_reduced <- data[, !is_constant]
# Save the names of the columns that were removed
removed_columns <- names(data)[is_constant]

#Horseshoe scaling using piironen and vehtari 2017
p0 <- n_vars/2
n <- nrow(xTrain_in)
p <- ncol(xTrain_in)
slab_scale<-sqrt(0.3/p0)*hs_ss_sd 
#global scale without sigma, as the scaling by sigma
#is done inside stan_glm 
global_scale<-(p0/(p-p0))/sqrt(n)
bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(), 
                           prior=hs(global_scale=global_scale, slab_scale=slab_scale), 
                           prior_intercept = normal(0,1), 
                           prior_aux =exponential(0.0000005, autoscale = FALSE), iter = 1000)
# Generate a random binary vector
x_current <-sample(c(0, 1), size = n_vars, replace = TRUE)

for (t in 1:evalBudget) {
  
  coeffs <- bayesian_model$coefficients
  result <- sdp_relaxation(coeffs, n_vars, lambda, removed_columns)
  x_new <- result$model
  
  
  x_new <- matrix(x_new, nrow=1, ncol=n_vars)
  cat("New evaluation point", x_new, "\n")
  
  # browser()
  #Append new point to existing x_vals
  x_vals_updated <- rbind(xTrain, x_new)
  # Evaluate model objective at the new evaluation point
  y_new <- model(x_new, seed)
  
  x_new <- matrix(x_new, nrow = 1, ncol = n_vars)
  x_new_in_comb <- order_effects(x_new, order)
  x_new_in <- x_new_in_comb$xTrain_in
  
  data_new <- data.frame(y = y_new, x_new_in)
  data <- rbind(data, data_new)
  
  
  hs_ss_sd <-sd(x_vals_updated)
  
  x_current <- x_new
  
  optim_result[t,] <- x_new
  
  #Running the Ling reg to train on constrained data
  is_constant <- apply(data, 2, function(x) length(unique(x)) == 1)
  data_reduced <- data[, !is_constant]
  # Save the names of the columns that were removed
  removed_columns <- names(data)[is_constant]
  
  #Horseshoe scaling using piironen and vehtari 2017
  p0 <- n_vars/2
  n <- nrow(xTrain_in)
  p <- ncol(xTrain_in)
  slab_scale<-sqrt(0.3/p0)*hs_ss_sd 
  #global scale without sigma, as the scaling by sigma
  #is done inside stan_glm 
  global_scale<-(p0/(p-p0))/sqrt(n)
  
  bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(), 
                             prior=hs(global_scale=global_scale, slab_scale=slab_scale), 
                             prior_intercept = normal(0,1), 
                             prior_aux =exponential(0.0000005, autoscale = FALSE), iter = 1000)
  
  # bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(), 
  #                            prior=hs(global_scale=global_scale, slab_scale=slab_scale), 
  #                            prior_intercept = normal(0,0.00001, autoscale = FALSE),
  #                            prior_aux =exponential(0.0000005, autoscale = FALSE) ,iter = 1000)
  
}

print(data)
# ---------------------------------------------------------
# PLOTS
# ---------------------------------------------------------

# Plot objective function values versus iterations for normal optimization
plot(1:nrow(data), data$y , type = "l", xlab = "Iterations", 
     ylab = "Objective Function Value", main = "Objective Function vs Iterations", 
     col = "red", xlim = c(1, nrow(data)))


