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
# 
# This code is for the CONTAMINATION test problem in BOCS

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
thompson_sam <- function(theta_current, bayesian_model, removed_columns, data){
  
  # browser()
  coeffs <- list()
  order <- 2
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
  theta_current_in <- theta_interaction(theta_current, order, n_vars)
  theta_current_in <- c(1, theta_current_in)
  
  coeffs <- bayesian_model$coefficients
  y_pred <-  sum(theta_current_in * coeffs)
  
  return(y_pred = y_pred)
}

# ---------------------------------------------------------
# SET INPUTS
# ---------------------------------------------------------
n_vars <-3
evalBudget <-10
n_init <- 5
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
theta_ini <- rep(0.5, ncol(xTrain))
theta_current <- theta_ini
theta_current_in <- theta_interaction(theta_current, order, n_vars)

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
                           prior_intercept = normal(0,1), iter = 1000)


for (t in 1:evalBudget) {
  
  x_current <- rbinom(n = n_vars, size = 1, prob = theta_current)
  
  stat_model <- function(theta) {
    con_y <- thompson_sam(theta, bayesian_model = bayesian_model, removed_columns, data)
  }
  
  ga_model <- function(theta) {
    f <- -(stat_model(theta))
    # pen <- sqrt(.Machine$double.xmax)  # penalty term
    # penalty1 <- max(c1(x_current),0)*pen
    # penalty2 <- max(c2(x_current),0)*pen
    (f) # - penalty1 - penalty2)            # fitness function value
  }
  
  
  GA <- ga(type = "real-valued", fitness = ga_model, lower = rep(0, n_vars), upper = rep(1, n_vars),
           popSize = 100, maxiter = 1000, run = 100)
  plot(GA)
  summary(GA)

  expected_val <- GA@solution
  cat("expected_val", expected_val, "\n")
  x_new <- rbinom(length(expected_val), 1, expected_val)
  # Evaluate model objective at the new evaluation point
  # Truncate x_new to the first n_vars elements
  x_new <- x_new[1:n_vars]
  
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
  
  theta_current <- expected_val
  
  optim_result[t,] <- expected_val
  
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
                             prior_intercept = normal(0,1), iter = 1000)
  
}

# ---------------------------------------------------------
# PLOTS
# ---------------------------------------------------------

# Plot objective function values versus iterations for normal optimization
plot(1:nrow(data), data$y , type = "l", xlab = "Iterations", 
     ylab = "Objective Function Value", main = "Objective Function vs Iterations", 
     col = "red", xlim = c(1, nrow(data)))
