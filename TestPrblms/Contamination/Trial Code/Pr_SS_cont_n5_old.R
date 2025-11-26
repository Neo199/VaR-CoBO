# SIMULATION STUDY
# CONTAMINATION PROBLEM FROM BOCS PAPER
# Comparing all the methods with ours here
# BOCS method is used to solve the problem here.
# Edit to restart the gibbs sampler from the averages of the previous iteration
# Modification of the BOCS code 
# The alpha values are sampled from a bayesian regression with a normal distribution prior
# The posterior predictive is derived from STAN hierarchical shrinkage
# Adding the method my Piironen and Vehtari on hyper prior chaice in hierarchical shrinkage
# 
# The methods differ in acquisition function and are: 
# Thompson Sampling solved using GA
# Thompson Sampling solved using SA
# Acquisition function using SDP
# Thompson Sampling solved using PRBOCS
# Addition of constraints in the AcqFun itself is done in BOCS-GA
# No constraints are actually included in the solver(only in the orignal contamination model)

# Author: Niyati Seth
# Date:  January 2025
# ---------------------------------------------------------
# LOAD Functions and Libraries
# ---------------------------------------------------------
# install.packages("GA")
# install.packages("psych")
# install.packages("lpSolve")
# install.packages("ompr")
# install.packages("ompr.roi")
# install.packages("ggplot2")
# install.packages("ROI")
# install.packages("GPfit")
# install.packages("rstanarm")
# install.packages("rstan")
# install.packages("parallel")

# source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/Contamination/sample_models.R")
# source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/Contamination/Contamination.R")
source("~/Contamination/sample_models.R")
source("~/Contamination/Contamination.R")
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
library(parallel)

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
    # print(paste0(("ord_i", ord_i, "\n")
    # print(paste0(("n_vars", n_vars, "\n")
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
    # print(paste0(("ord_i", ord_i, "\n"))
    # print(paste0(("n_vars", n_vars, "\n"))
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
# SIMULATED ANNEALING
# ---------------------------------------------------------

simulated_annealing <- function(objective, n_vars, evalBudget, x_current) {
  
  # Initialize matrices to store solutions
  model_iter <- matrix(0, nrow = evalBudget, ncol = n_vars)
  obj_iter <- numeric(evalBudget)
  
  # Initial temperature and cooling schedule
  T <- 1.0
  cool <- function(T) 0.8 * T
  
  # Generate initial solution and evaluate objective
  old_x <- x_current
  old_obj <- objective(old_x)
  
  # Initialize best solution
  best_x <- old_x
  best_obj <- old_obj
  
  # Run simulated annealing
  for (t in 1:evalBudget) {
    # Decrease temperature
    T <- cool(T)
    
    # Generate new candidate by flipping a random bit
    flip_bit <- sample(1:n_vars, 1)
    new_x <- old_x
    new_x[flip_bit] <- 1 - new_x[flip_bit]
    
    # Evaluate objective function
    new_obj <- objective(new_x)
    
    # Accept new solution if it improves or probabilistically accept it
    if ((new_obj < old_obj) || (runif(1) < exp((old_obj - new_obj) / T))) {
      old_x <- new_x
      old_obj <- new_obj
    }
    
    # Update best solution
    if (new_obj < best_obj) {
      best_x <- new_x
      best_obj <- new_obj
    }
    
    # Save solution
    model_iter[t, ] <- best_x
    obj_iter[t] <- best_obj
  }
  
  return(list(model_iter = model_iter, obj_iter = obj_iter))
}

# ---------------------------------------------------------
# SEMIDEFINITE PROGRAMMING
# ---------------------------------------------------------

sdp_relaxation <- function(alpha, n_vars, lambda, removed_columns) {
  
  # Append missing variables back with zero coefficients
  alpha_full <- c(alpha, setNames(rep(0, length(removed_columns)), removed_columns))
  
  # Reorder so that variables are in a consistent order
  alpha_full <- alpha_full[order(names(alpha_full))]
  
  # Check results
  print(alpha_full)
  
  # Extract vector of coefficients
  b <- alpha_full[2:(n_vars + 1)] + lambda  # Skip intercept
  a <- alpha_full[(n_vars + 2):(n_vars + 1 + choose(n_vars, 2))]  # Extract quadratic terms
  
  # Get indices for quadratic terms
  idx_prod <- combn(n_vars, 2)
  n_idx <- ncol(idx_prod)
  
  # Check number of coefficients
  if (length(a) != n_idx) {
    stop("Number of coefficients does not match indices!")
  }
  
  # Convert a to matrix form
  A <- matrix(0, n_vars, n_vars)
  for (i in 1:n_idx) {
    A[idx_prod[1, i], idx_prod[2, i]] <- a[i] / 2
    A[idx_prod[2, i], idx_prod[1, i]] <- a[i] / 2
  }
  
  # Convert to standard form
  bt <- (b / 2) + (A %*% rep(1, n_vars)) / 2
  At <- rbind(cbind(A / 4, bt / 2), c(bt / 2, 2))
  
  # Run SDP relaxation (approximate using eigenvalue decomposition)
  X <- diag(n_vars + 1)  # Initial identity matrix
  
  # Enforce positive semi-definiteness using eigen decomposition
  eig <- eigen(At)
  eig$values[eig$values < 0] <- 1e-15  # Adjust negative eigenvalues
  X <- eig$vectors %*% diag(eig$values) %*% t(eig$vectors)
  
  # Compute Cholesky decomposition
  L <- chol(X)
  
  # Repeat rounding for different vectors
  n_rand_vector <- 100
  model_vect <- matrix(0, nrow = n_vars, ncol = n_rand_vector)
  obj_vect <- numeric(n_rand_vector)
  
  for (kk in 1:n_rand_vector) {
    # Generate a random cutting plane vector
    r <- rnorm(n_vars + 1)
    r <- r / sqrt(sum(r^2))  # Normalize vector
    y_soln <- sign(t(L) %*% r)
    
    # Convert solution to original domain
    model_vect[, kk] <- (y_soln[1:n_vars] + 1) / 2
    obj_vect[kk] <- t(model_vect[, kk]) %*% A %*% model_vect[, kk] + sum(b * model_vect[, kk])
  }
  
  # Find optimal rounded solution
  opt_idx <- which.min(obj_vect)
  model <- model_vect[, opt_idx]
  obj <- obj_vect[opt_idx]
  
  return(list(model = model, obj = obj))
}

# ---------------------------------------------------------
# DEFINE TRUE MODEL
# ---------------------------------------------------------
model <- function(x_vals, seed){
  contamination_prob(x_vals, 100, seed) 
}

# ---------------------------------------------------------
# FUNCTION TO RUN INSTANCE
# ---------------------------------------------------------
run_simulation_instance <- function(instance_id, n_vars, evalBudget, n_init, order, seed, folder_name) {
  # tryCatch({
  # Set the seed for reproducibility
  inst_seed <- seed + instance_id
  set.seed(inst_seed)
  
  # ---------------------------------------------------------
  # INITIAL SAMPLES FOR STATISTICAL MODELS
  # ---------------------------------------------------------
  x_vals <- sample_models(n_init, n_vars)
  y_vals <- contamination_prob(x_vals, 100, seed)

  print(x_vals)
  print(y_vals)

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
  # print(paste0(("nSamps, nCoeffs", c(nSamps, nCoeffs), "\n")
  
  hs_ss_sd <- sd(xTrain)
  n <- nrow(xTrain_in)
  p <- ncol(xTrain_in)
  
  #Setup dataframe for stan_glm training 
  X <- xTrain_in
  y <- y_vals
  
  #Create a initial dataframe
  data <- data.frame(y=y , X)
  
  # ---------------------------------------------------------
  # RUN PRBOCS-GA
  # ---------------------------------------------------------
  # Start timing for BOCS-GA
  start_time <- Sys.time()
  
  # Initialize a data frame to store iteration results
  prbocsga_result <- matrix(0, evalBudget, n_vars)
  
  #Initialise BOCS-GA specific dataframe
  prbocsga_data <- data
  
  #Initialise theta values
  theta_ini <- rep(0.5, ncol(xTrain))
  theta_current <- theta_ini
  theta_current_in <- theta_interaction(theta_current, order, n_vars)
  
  #Initializing the bayesian model with the given data first
  # Check for constant columns
  # Remove constant columns
  is_constant <- apply(prbocsga_data, 2, function(x) length(unique(x)) == 1)
  data_reduced <- prbocsga_data[, !is_constant]
  # Save the names of the columns that were removed
  removed_columns <- names(prbocsga_data)[is_constant]
  
  #Horseshoe scaling using piironen and vehtari 2017
  p0 <- n_vars/2
  n <- nrow(xTrain_in)
  p <- ncol(xTrain_in)
  slab_scale<-sqrt(0.3/p0)*hs_ss_sd 
  #global scale without sigma, as the scaling by sigma
  #is done inside stan_glm 
  global_scale<-(p0/(p-p0))/sqrt(n)
  prbocsga_bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(), 
                             prior=hs(global_scale=global_scale, slab_scale=slab_scale), 
                             prior_intercept = normal(0,1), iter = 1000)
  
  for (t in 1:n_iter) {
    print(paste("prbocsga_iteration_",t))
    
    x_current <- rbinom(n = n_vars, size = 1, prob = theta_current)
    
    stat_model <- function(theta) {
      con_y <- thompson_sam(theta, bayesian_model = prbocsga_bayesian_model, removed_columns, prbocsga_data, order = 2)
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
    # plot(GA)
    # summary(GA)
    
    expected_val <- GA@solution
    cat("New expectation of evaluation point", expected_val, "\n")
    
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
    prbocsga_data <- rbind(prbocsga_data, data_new)
    
    
    hs_ss_sd <-sd(x_vals_updated)
    
    theta_current <- expected_val
    
    # optim_result[t,] <- expected_val
    
    #Running the Ling reg to train on constrained data
    is_constant <- apply(prbocsga_data, 2, function(x) length(unique(x)) == 1)
    data_reduced <- prbocsga_data[, !is_constant]
    # Save the names of the columns that were removed
    removed_columns <- names(prbocsga_data)[is_constant]
    
    #Horseshoe scaling using piironen and vehtari 2017
    p0 <- n_vars/2
    n <- nrow(xTrain_in)
    p <- ncol(xTrain_in)
    slab_scale<-sqrt(0.3/p0)*hs_ss_sd 
    #global scale without sigma, as the scaling by sigma
    #is done inside stan_glm 
    global_scale<-(p0/(p-p0))/sqrt(n)
    prbocsga_bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(), 
                               prior=hs(global_scale=global_scale, slab_scale=slab_scale), 
                               prior_intercept = normal(0,1), iter = 1000)
    
  }
  
  
  # End timing
  end_time <- Sys.time()
  prbocsga_elapsed_time <- end_time - start_time
  
  # ---------------------------------------------------------
  # RUN BOCS-GA
  # ---------------------------------------------------------
  
  # Start timing for BOCS-GA
  start_time <- Sys.time()
  
  # Initialize a data frame to store iteration results
  bocsga_result <- matrix(0, evalBudget, n_vars)
  
  #Initialise BOCS-GA specific dataframe
  bocsga_data <- data
  
  # browser()
  
  #Initializing the bayesian model with the given data first
  # Check for constant columns
  # Remove constant columns
  is_constant <- apply(bocsga_data, 2, function(x) length(unique(x)) == 1)
  data_reduced <- bocsga_data[, !is_constant]
  # Save the names of the columns that were removed
  removed_columns <- names(bocsga_data)[is_constant]
  
  #Horseshoe scaling using piironen and vehtari 2017
  p0 <- n_vars/2
  n <- nrow(xTrain_in)
  p <- ncol(xTrain_in)
  slab_scale<-sqrt(0.3/p0)*hs_ss_sd 
  #global scale without sigma, as the scaling by sigma
  #is done inside stan_glm 
  global_scale<-(p0/(p-p0))/sqrt(n)
  bocsga_bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(), 
                             prior=hs(global_scale=global_scale, slab_scale=slab_scale), 
                             prior_intercept = normal(0,1, autoscale = TRUE),
                             prior_aux = exponential(0.0000005, autoscale = FALSE), iter = 2000)
  
  # Generate a random binary vector
  x_current <-sample(c(0, 1), size = n_vars, replace = TRUE)
  
  
  for (t in 1:n_iter) {
    print(paste("bocsga_iteration_",t))
    stat_model <- function(x_current) {
      thompson_sam(x_current, bayesian_model = bocsga_bayesian_model, removed_columns, data, order =2)
    }
    
    ga_model <- function(x_current) {
      f <- -(stat_model(x_current))
      (f) # - penalty1 - penalty2)            # fitness function value
    }
    
    GA <- ga(type = "binary", fitness = ga_model, nBits = n_vars, 
             popSize = 100, maxiter = 1000, run = 100)
    # summary(GA)
    # plot (GA)
    
    
    # Evaluate model objective at the new evaluation point
    # Truncate x_new to the first n_vars elements
    x_new <- GA@solution
    
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
    bocsga_data <- rbind(bocsga_data, data_new)
    
    
    hs_ss_sd <-sd(x_vals_updated)
    
    x_current <- x_new
    bocsga_result[t,] <- x_new
    
    #Running the Ling reg to train on constrained data
    is_constant <- apply(bocsga_data, 2, function(x) length(unique(x)) == 1)
    data_reduced <- bocsga_data[, !is_constant]
    # Save the names of the columns that were removed
    removed_columns <- names(bocsga_data)[is_constant]
    
    #Horseshoe scaling using piironen and vehtari 2017
    p0 <- n_vars/2
    n <- nrow(xTrain_in)
    p <- ncol(xTrain_in)
    slab_scale<-sqrt(0.3/p0)*hs_ss_sd 
    #global scale without sigma, as the scaling by sigma
    #is done inside stan_glm 
    global_scale<-(p0/(p-p0))/sqrt(n)
    
    bocsga_bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(), 
                               prior=hs(global_scale=global_scale, slab_scale=slab_scale), 
                               prior_intercept = normal(0,1, autoscale = TRUE),
                               prior_aux = exponential(0.0000005, autoscale = FALSE), iter = 2000)
    
    # bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(), 
    #                            prior=hs(global_scale=global_scale, slab_scale=slab_scale), 
    #                            prior_intercept = normal(0,0.00001, autoscale = FALSE),
    #                            prior_aux =exponential(0.0000005, autoscale = FALSE) ,iter = 1000)
    
  }
  
  # End timing
  end_time <- Sys.time()
  bocsga_elapsed_time <- end_time - start_time
  
  
  # ---------------------------------------------------------
  # RUN BOCS-SA
  # ---------------------------------------------------------
  
  # Start timing for BOCS-SA
  start_time <- Sys.time()
  
  # Initialize a data frame to store iteration results
  bocssa_result <- matrix(0, evalBudget, n_vars)
  
  #Initialise BOCS-SA specific dataframe
  bocssa_data <- data
  
  #Initializing the bayesian model with the given data first
  # Check for constant columns
  # Remove constant columns
  is_constant <- apply(bocssa_data, 2, function(x) length(unique(x)) == 1)
  data_reduced <- bocssa_data[, !is_constant]
  # Save the names of the columns that were removed
  removed_columns <- names(bocssa_data)[is_constant]
  
  #Horseshoe scaling using piironen and vehtari 2017
  p0 <- n_vars/2
  n <- nrow(xTrain_in)
  p <- ncol(xTrain_in)
  slab_scale<-sqrt(0.3/p0)*hs_ss_sd 
  #global scale without sigma, as the scaling by sigma
  #is done inside stan_glm 
  global_scale<-(p0/(p-p0))/sqrt(n)
  bocssa_bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(), 
                             prior=hs(global_scale=global_scale, slab_scale=slab_scale), 
                             prior_intercept = normal(0,1, autoscale = TRUE),
                             prior_aux = exponential(0.0000005, autoscale = FALSE), iter = 1000)
  
  # Generate a random binary vector
  x_current <-sample(c(0, 1), size = n_vars, replace = TRUE)
  
  for (t in 1:n_iter) {
    print(paste("bocssa_iteration_",t))
    
    stat_model <- function(x_current) {
      thompson_sam(x_current, bayesian_model = bocssa_bayesian_model, removed_columns, data, order =2)
    }
    
    # c1 <- function(x_current) 
    # { unlist(x_current)%*%cost_mat - costBudget }
    # 
    # c2 <- function(x_current) 
    # { minSpend - unlist(x_current)%*%cost_mat}
    # 
    
    # Set the number of SA reruns
    SA_reruns = 5
    
    SA_model <- matrix(0, nrow = SA_reruns, ncol = n_vars)
    SA_obj <- numeric(SA_reruns)
    
    for (j in 1:SA_reruns) {
      SA <- simulated_annealing(stat_model, n_vars, evalBudget, x_current)
      SA_model[j, ] <- SA$model[nrow(SA$model), ]  # Last row is the best model
      SA_obj[j] <- SA$obj[length(SA$obj)]  # Last objective value
    }
    
    # Evaluate model objective at the new evaluation point
    # Find the index of the minimum objective value
    min_idx <- which.min(SA_obj)
    
    # Select the corresponding best model
    x_new <- SA_model[min_idx, ]
    
    
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
    bocssa_data <- rbind(bocssa_data, data_new)
    
    
    hs_ss_sd <-sd(x_vals_updated)
    
    x_current <- x_new
    
    bocssa_result[t,] <- x_new
    
    #Running the Ling reg to train on constrained data
    is_constant <- apply(bocssa_data, 2, function(x) length(unique(x)) == 1)
    data_reduced <- bocssa_data[, !is_constant]
    # Save the names of the columns that were removed
    removed_columns <- names(bocssa_data)[is_constant]
    
    #Horseshoe scaling using piironen and vehtari 2017
    p0 <- n_vars/2
    n <- nrow(xTrain_in)
    p <- ncol(xTrain_in)
    slab_scale<-sqrt(0.3/p0)*hs_ss_sd 
    #global scale without sigma, as the scaling by sigma
    #is done inside stan_glm 
    global_scale<-(p0/(p-p0))/sqrt(n)
    
    bocssa_bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(), 
                               prior=hs(global_scale=global_scale, slab_scale=slab_scale), 
                               prior_intercept = normal(0,1, autoscale = TRUE),
                               prior_aux = exponential(0.0000005, autoscale = FALSE), iter = 1000)
  }
  
  end_time <- Sys.time()
  bocssa_elapsed_time <- end_time - start_time
  
  # ---------------------------------------------------------
  # RUN BOCS-SDP
  # ---------------------------------------------------------
   
  # Start timing for BOCS-SDP
  start_time <- Sys.time()
  lambda <- .2 #For SDP
  
  # Initialize a data frame to store iteration results
  bocssdp_result <- matrix(0, evalBudget, n_vars)
  
  #Initialise BOCS-SA specific dataframe
  bocssdp_data <- data
  
  #Initializing the bayesian model with the given data first
  # Check for constant columns
  # Remove constant columns
  is_constant <- apply(bocssdp_data, 2, function(x) length(unique(x)) == 1)
  data_reduced <- bocssdp_data[, !is_constant]
  # Save the names of the columns that were removed
  removed_columns <- names(bocssdp_data)[is_constant]
  
  #Horseshoe scaling using piironen and vehtari 2017
  p0 <- n_vars/2
  n <- nrow(xTrain_in)
  p <- ncol(xTrain_in)
  slab_scale<-sqrt(0.3/p0)*hs_ss_sd 
  #global scale without sigma, as the scaling by sigma
  #is done inside stan_glm 
  global_scale<-(p0/(p-p0))/sqrt(n)
  bocssdp_bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(), 
                             prior=hs(global_scale=global_scale, slab_scale=slab_scale), 
                             prior_intercept = normal(0,1, autoscale = TRUE),
                             prior_aux = exponential(0.0000005, autoscale = FALSE), iter = 1000)
  
  # Generate a random binary vector
  x_current <-sample(c(0, 1), size = n_vars, replace = TRUE)
  
  for (t in 1:n_iter) {
    print(paste("bocssdp_iteration_",t))
    coeffs <- bocssdp_bayesian_model$coefficients
    sdp_result <- sdp_relaxation(coeffs, n_vars, lambda, removed_columns)
    x_new <- sdp_result$model
    
    
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
    bocssdp_data <- rbind(bocssdp_data, data_new)
    
    
    hs_ss_sd <-sd(x_vals_updated)
    
    x_current <- x_new
    
    bocssdp_result[t,] <- x_new
    
    #Running the Ling reg to train on constrained data
    is_constant <- apply(bocssdp_data, 2, function(x) length(unique(x)) == 1)
    data_reduced <- bocssdp_data[, !is_constant]
    # Save the names of the columns that were removed
    removed_columns <- names(bocssdp_data)[is_constant]
    
    #Horseshoe scaling using piironen and vehtari 2017
    p0 <- n_vars/2
    n <- nrow(xTrain_in)
    p <- ncol(xTrain_in)
    slab_scale<-sqrt(0.3/p0)*hs_ss_sd 
    #global scale without sigma, as the scaling by sigma
    #is done inside stan_glm 
    global_scale<-(p0/(p-p0))/sqrt(n)
    
    bocssdp_bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(), 
                               prior=hs(global_scale=global_scale, slab_scale=slab_scale), 
                               prior_intercept = normal(0,1, autoscale = TRUE),
                               prior_aux = exponential(0.0000005, autoscale = FALSE), iter = 1000)
    
    # bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(), 
    #                            prior=hs(global_scale=global_scale, slab_scale=slab_scale), 
    #                            prior_intercept = normal(0,0.00001, autoscale = FALSE),
    #                            prior_aux =exponential(0.0000005, autoscale = FALSE) ,iter = 1000)
    
  }
  
  end_time <- Sys.time()
  bocssdp_elapsed_time <- end_time - start_time
  
  # ---------------------------------------------------------
  # PLOTs
  # ---------------------------------------------------------
  # library(ggplot2)
  # 
  # # Create a data frame for ggplot
  # bocsga_data$Method <- "BOCS-GA"
  # bocssa_data$Method <- "BOCS-SA"
  # bocssdp_data$Method <- "BOCS-SDP"
  # 
  # # Combine all data into one data frame
  # plot_data <- rbind(bocsga_data, bocssa_data, bocssdp_data)
  # plot_data$Iteration <- rep(1:nrow(plot_data), times = c(nrow(bocsga_data), nrow(bocssa_data), nrow(bocssdp_data)))
  # 
  # # Generate the plot
  # p <- ggplot(plot_data, aes(x = Iteration, y = y, color = Method)) +
  #   geom_line() +
  #   labs(title = "Objective Function vs Iterations",
  #        x = "Iterations",
  #        y = "Objective Function Value") +
  #   theme_minimal()
  # 
  # # Save the plot
  # plot_file <- file.path(folder_name, paste0("instance_", instance_id, "_objective_plot.pdf"))
  # ggsave(plot_file, plot = p, device = "pdf", width = 8, height = 6)
  # 
  # Generate and save the plot in the folder
  # plot_file <- file.path(folder_name, paste0("instance_", instance_id, "_objective_plot.pdf"))
  # pdf(file = plot_file)
  # # Plot BOCS-GA data
  # plot(1:nrow(bocsga_data), bocsga_data$y, type = "l", xlab = "Iterations",
  #      ylab = "Objective Function Value", main = "Objective Function vs Iterations",
  #      col = "red", xlim = c(1, max(nrow(bocsga_data), nrow(bocssa_data), nrow(bocssdp_data), nrow(prbocsga_data))),
  #      ylim = c(1, max(bocsga_data$y, bocssa_data$y, bocssdp_data$y, prbocsga_data$y)))
  # 
  # # Add BOCS-SA data as a new line
  # lines(1:nrow(bocssa_data), bocssa_data$y, col = "blue")
  # 
  # # Add BOCS-SDP data as a new line
  # lines(1:nrow(bocssdp_data), bocssdp_data$y, col = "green")
  # 
  # # Add PRBOCS-GA data as a new line
  # lines(1:nrow(prbocsga_data), prbocsga_data$y, col = "orange")
  # 
  # # Add a legend to differentiate the models
  # legend("topright", legend = c("BOCS-GA", "BOCS-SA", "BOCS-SDP", "PRBOCS_GA"), col = c("red", "blue","green", "orange"), lty = 1)
  # 
  # dev.off()
  
  # Capture and return the result for the instance
  prbocsga <- list(instance_id = instance_id, solution = tail(prbocsga_data, n =1), data = prbocsga_data, 
                 model = prbocsga_bayesian_model, time_taken = prbocsga_elapsed_time)
  bocsga <- list(instance_id = instance_id, solution = tail(bocsga_data, n =1), data = bocsga_data, 
                 model = bocsga_bayesian_model, time_taken = bocsga_elapsed_time)
  bocssa <- list(instance_id = instance_id, solution = tail(bocssa_data, n =1), data = bocssa_data, 
                 model = bocssa_bayesian_model, time_taken = bocssa_elapsed_time)
  bocssdp <- list(instance_id = instance_id, solution = tail(bocssdp_data, n =1), data = bocssdp_data, 
                 model = bocssdp_bayesian_model, time_taken = bocssdp_elapsed_time)
  
  result <- list(PRBOCS_GA = prbocsga, BOCS_GA = bocsga, BOCS_SA = bocssa, BOCS_SDP = bocssdp)
  
  # return(result)
  # },error = function(e) {
  #   message(paste("Error in instance:", instance_id, ":", e$message))
  #   return(NULL)  # Return NULL if an error occurs
  # })
}

# ---------------------------------------------------------
# FUNCTION TO RUN SIMULATION
# ---------------------------------------------------------

run_simulation_study <- function(n_vars, evalBudget, n_init, order, seed, num_instances) {
  # Create a folder named "n_vars=n_vars" if it doesn't exist
  folder_name <- paste0("n_vars=", n_vars)
  if (!dir.exists(folder_name)) {
    dir.create(folder_name)
  }
  
  # browser()
  # Number of available cores (adjust as needed)
  num_cores <- detectCores() - 1  # Use all cores minus one to avoid overloading your system
  
  run_instance <- function(i){
    print(paste("Running instance:", i))
    browser()
    res <- run_simulation_instance(i, n_vars, evalBudget, n_init, order, seed, folder_name)
    print(paste("Finished instance:", i))
  }
  
  # Run simulations in parallel
  results <- mclapply(1:num_instances, run_instance, 
                      mc.cores = num_cores, 
                      mc.preschedule = FALSE)
  
  
  # Remove NULL results (if any instances failed)
  results <- Filter(Negate(is.null), results)
  
  # Get model names dynamically from results
  model_names <- unique(unlist(lapply(results, names)))
  
  # Initialize an empty list for storing results
  model_results_list <- setNames(vector("list", length(model_names)), model_names)
  
  # Organize results by model
  for (instance_result in results) {
    if (is.null(instance_result)) next
    
    print("Processing an instance:")
    print(names(instance_result))  # Debugging step
    
    for (model_name in names(instance_result)) {  
      model_data <- instance_result[[model_name]]
      
      if (!is.null(model_data)) {
        model_results_list[[model_name]] <- append(model_results_list[[model_name]], list(model_data))
      }
    }
  }
  
  save(results, file = file.path(folder_name, "simulation_results.RData"))
  
  # Debugging: Print structured results before writing
  print("Model results list:")
  print(str(model_results_list))
  
  # Write each model's results to its own file
  for (model_name in names(model_results_list)) {
    model_results <- model_results_list[[model_name]]
    
    # Skip if no results exist for this model
    if (length(model_results) == 0) {
      print(paste("Skipping", model_name, "because no results were found"))
      next
    }
    
    # Create a separate file for each model
    model_file <- file.path(folder_name, paste0(model_name, "_results.txt"))
    print(paste("Writing to file:", model_file))  # Debugging step
    
    fileConn <- file(model_file, "w")  # Open file for writing
    
    writeLines(c(
      paste0("========== Model: ", model_name, " ==========\n")
    ), fileConn)
    
    for (instance in model_results) {
      writeLines(c(
        paste0("Instance: ", instance$instance_id),
        "-----------------------------------",
        paste0("Solution: ", paste(instance$solution, collapse = ", ")),
        "-----------------------------------",
        "Data:",
        paste(capture.output(print(instance$data)), collapse = "\n"),
        "-----------------------------------",
        "Model Summary:",
        paste(capture.output(print(instance$model)), collapse = "\n"),
        "-----------------------------------",
        paste0("Time Taken: ", instance$time_taken),
        "===================================",
        ""
      ), fileConn)
    }
    
    close(fileConn)
  }
  
  return(results)
}





# ---------------------------------------------------------
# SET INPUTS
# ---------------------------------------------------------

# # Main function to run the study for multiple values of n_vars
# run_simulation_for_multiple_n_vars <- function(n_vars_list, evalBudget, n_init, order, seed, num_instances) {
#   for (n_vars in n_vars_list) {
#     print(paste0("\nRunning simulation for n_vars =", n_vars, "...\n"))
#     run_simulation_study(n_vars, evalBudget, n_init, order, seed, num_instances)
#     }
# }
# 
# # Define parameters
# n_vars_list <- c(3, 5, 2)# 10, 15, 20, 50, 100, 150, 200)
# evalBudget <- 100
# order <- 2
# seed <- 1
# num_instances <- 5
# 
# # Run the simulations
# results <- run_simulation_for_multiple_n_vars(n_vars_list, evalBudget, n_init, order, seed, num_instances)


n_vars <- 5
evalBudget <- 100
n_init <- 3
order <- 2
seed <- 1
num_instances <- 100

# Run the simulation study
results <- run_simulation_study(n_vars, evalBudget, n_init, order, seed, num_instances)

#Inspect the results after the simulations are done
print(results)
