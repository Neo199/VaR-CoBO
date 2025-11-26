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
# SET INPUTS
# ---------------------------------------------------------
n_vars <-3
evalBudget <- 10
n_init <- 5
lambda <- .2
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
# INITIAL SAMPL
# 
# S FOR STATISTICAL MODELS
# ---------------------------------------------------------
x_vals <- sample_models(n_init, n_vars)
y_vals <- contamination_prob(x_vals, 100, seed)

x_vals
y_vals


# ---------------------------------------------------------
# RUN LPSOLVE
# ---------------------------------------------------------
# gamma <- 1
# con_res <- Contamination(x_vals, 100, seed)
# cost <- con_res$fn
# constraint <- con_res$constraint
# 
# # Define the MIP model
# lpmodel <- MIPModel() %>%
#   # Add binary decision variables x[i] for each item
#   add_variable(x[i], i = 1:n_vars, type = "binary") %>%
#   set_objective(sum_expr(out[i] <- cost - sum(gamma * constraint), i = 1:n_vars), "mi n")  # Set the objective function: minimize the total cost
# 
# # Solve the model
# lpresult <- solve_model(lpmodel, with_ROI(solver = "lpsolve"))
# 
# # Extract the solution
# solution <- get_solution(lpresult, x[i])
# 
# # Display the solution
# print(solution)

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
                           prior_intercept = normal(0,1, autoscale = TRUE),
                           prior_aux = exponential(0.0000005, autoscale = FALSE), iter = 1000)

# Generate a random binary vector
x_current <-sample(c(0, 1), size = n_vars, replace = TRUE)

for (t in 1:evalBudget) {
  stat_model <- function(x_current) {
    thompson_sam(x_current, bayesian_model = bayesian_model, removed_columns, data, order =2)
  }

  # c1 <- function(x_current) 
  # { unlist(x_current)%*%cost_mat - costBudget }
  # 
  # c2 <- function(x_current) 
  # { minSpend - unlist(x_current)%*%cost_mat}
  # 
  ga_model <- function(x_current) {
    f <- -(stat_model(x_current))
    # pen <- sqrt(.Machine$double.xmax)  # penalty term
    # penalty1 <- max(c1(x_current),0)*pen
    # penalty2 <- max(c2(x_current),0)*pen
    (f) # - penalty1 - penalty2)            # fitness function value
  }
  
  GA <- ga(type = "binary", fitness = ga_model, nBits = n_vars, 
           popSize = 100, maxiter = 1000, run = 100, monitor = FALSE)
  summary(GA)
  plot (GA)
  

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
  
  # bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(), 
  #                            prior=hs(global_scale=global_scale, slab_scale=slab_scale), 
  #                            prior_intercept = normal(0,1), iter = 1000)

  bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(),
                             prior=hs(global_scale=global_scale, slab_scale=slab_scale),
                             prior_intercept = normal(0,1, autoscale = TRUE),
                             prior_aux = exponential(0.0000005, autoscale = FALSE), iter = 1000)

  # bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(), 
  #                            prior=hs(global_scale=global_scale, slab_scale=slab_scale), 
  #                            prior_intercept = normal(0,0.00001, autoscale = FALSE),
  #                            prior_aux =exponential(0.0000005, autoscale = FALSE) ,iter = 1000)
  
}

# ---------------------------------------------------------
# GA MODEL
# ---------------------------------------------------------

fitness_function <- function(x) {
  out <- -(contamination_prob(x, 100, seed=1))
  (out)
}

# 
# # Run the genetic algorithm
# ga_result <- ga(type = "binary", fitness = fitness_function, nBits = n_vars, 
#                 popSize = 100, maxiter = 1000, run = 100)
# 
# # Best solution found
# optimal_solution <-  ga_result@solution
# 
# # Fitness value of the best solution
# best_fitness <- ga_result@fitnessValue
# 
# # Print the results
# cat("Optimal Solution:", optimal_solution, "\n")
# cat("Best Fitness Value:", best_fitness, "\n")
# 
# # Plot the GA process
# plot(ga_result)

# ---------------------------------------------------------
# SIMULATED ANNEALING
# ---------------------------------------------------------

simulated_annealing <- function(objective, inputs) {
  # Extract inputs
  n_vars <- inputs$n_vars
  n_iter <- inputs$evalBudget
  
  # Initialize matrices to store solutions
  model_iter <- matrix(0, nrow = n_iter, ncol = n_vars)
  obj_iter <- numeric(n_iter)
  
  # Initial temperature and cooling schedule
  T <- 1.0
  cool <- function(T) 0.8 * T
  
  # Generate initial solution and evaluate objective
  old_x <- matrix(sample(c(0,1), n_vars, replace = TRUE), nrow = 1)
  old_obj <- objective(old_x)
  
  # Initialize best solution
  best_x <- old_x
  best_obj <- old_obj
  
  # Run simulated annealing
  for (t in 1:n_iter) {
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
# PLOTS
# ---------------------------------------------------------

# Plot objective function values versus iterations for normal optimization
plot(1:nrow(data), data$y , type = "l", xlab = "Iterations", 
     ylab = "Objective Function Value", main = "Objective Function vs Iterations", 
     col = "red", xlim = c(1, nrow(data)))

pp_check(bayesian_model)


