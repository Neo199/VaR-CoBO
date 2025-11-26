
# ---------------------------------------------------------
# LOAD Functions and Libraries
# ---------------------------------------------------------
source("~/Projects:Codes/PhD-Compute/R code/SparseVB/sample_models.R")
source("~/Projects:Codes/PhD-Compute/R code/SparseVB/Contamination.R")
source("~/Projects:Codes/PhD-Compute/R code/SparseVB/sparsevb/R/RcppExports.R")
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

# ---------------------------------------------------------
# LOAD Svb Functions
# ---------------------------------------------------------
checkargs.xy <- function(x, y) {
  if (missing(x)) stop("x is missing")
  if (is.null(x) || !is.matrix(x)) stop("x must be a matrix")
  if (missing(y)) stop("y is missing")
  if (is.null(y) || !is.numeric(y)) stop("y must be numeric")
  if (ncol(x) == 0) stop("There must be at least one predictor [must have ncol(x) > 0]")
  if (checkcols(x)) stop("x cannot have duplicate columns")
  if (length(y) == 0) stop("There must be at least one data point [must have length(y) > 0]")
  if (length(y)!=nrow(x)) stop("Dimensions don't match [length(y) != nrow(x)]")
}

estimateSigma <- function (x, y, intercept = TRUE, standardize = TRUE) {
  checkargs.xy(x, rep(0, nrow(x)))
  # browser()
  # if (nrow(x) < 10) 
  #   stop("Number of observations must be at least 10 to run estimateSigma")
  cvfit = cv.glmnet(x, y, intercept = intercept, standardize = standardize)
  lamhat = cvfit$lambda.min
  fit = glmnet(x, y, standardize = standardize)
  yhat = predict(fit, x, s = lamhat)
  nz = sum(predict(fit, s = lamhat, type = "coef") != 0)
  if ((nz + 1) >= length(y)){
    den = 2
  } else {
    den = length(y) - nz - 1}
  sigma = sqrt(sum((y - yhat)^2)/(den))
  return(list(sigmahat = sigma, df = nz))
}

svb.fit <- function(X,
                    Y,
                    family = c("linear", "logistic"),
                    slab = c("laplace", "gaussian"),
                    mu,
                    sigma = rep(1, ncol(X)),
                    gamma,
                    alpha,
                    beta,
                    prior_scale = 1,
                    update_order,
                    intercept = FALSE,
                    noise_sd,
                    max_iter = 1000,
                    tol = 1e-5) {
  
  #extract problem dimensions
  n = nrow(X)
  p = ncol(X)
  
  #rescale data if necessary
  if(match.arg(family) == "linear" && missing(noise_sd)) {
    noise_sd = estimateSigma(X, Y)$sigmahat
  } else if(match.arg(family) == "logistic") {
    noise_sd = 1 
  }
  X = X/noise_sd
  Y = Y/noise_sd
  
  #compute initial estimator for mu
  if(missing(mu)) {
    # browser()
    cvfit = cv.glmnet(X, Y, family = ifelse(match.arg(family) == "linear", "gaussian", "binomial"), intercept = intercept, alpha = 0)
    mu = as.numeric(coef(cvfit, s = "lambda.min"))
    
    if(intercept) {
      mu = c(mu[2:(p+1)], mu[1])
    } else {
      mu = mu[2:(p+1)]
    }
  } else if (intercept) {
    mu = c(mu, 0)
  }
  
  #generate prioritized updating order
  if(missing(update_order)) {
    update_order = order(abs(mu[1:p]), decreasing = TRUE)
    update_order = update_order - 1
  }
  
  #compute initial estimators for alpha, beta, and gamma
  if(missing(gamma) || missing(alpha) || missing(beta)) {
    cvfit = cv.glmnet(X, Y, family = ifelse(match.arg(family) == "linear", "gaussian", "binomial"), intercept = intercept, alpha = 1)
    
    s_hat = length(which(coef(cvfit, s = "lambda.1se")[-1] != 0))
    
    if(missing(alpha)) {
      alpha = s_hat
    }
    if(missing(beta)) {
      beta = p - s_hat
    }
    if(missing(gamma)){
      gamma = rep(s_hat/p, p)
      gamma[which(coef(cvfit, s = "lambda.1se")[-1] != 0)] = 1
    }
  }
  
  #add intercept
  if(intercept){
    sigma = c(sigma, 1)
    gamma = c(gamma, 1)
    update_order = c(p, update_order)
    X = cbind(X, rep(1/noise_sd, n))
  }
  
  #match internal function call and generate list of arguments
  fn = paste("fit", match.arg(family), sep = '_')
  arg = list(X, Y, mu, sigma, gamma, alpha, beta, prior_scale, update_order, match.arg(slab), max_iter, tol)
  
  #perform chosen computation
  approximate_posterior = lapply(do.call(fn, arg), as.numeric)
  
  #convert results to R-style vectors since RcppArmadillo returns in matrix form
  return(list(mu = approximate_posterior$mu[1:p], sigma = approximate_posterior$sigma[1:p], gamma = approximate_posterior$gamma[1:p], intercept = ifelse(intercept, mu[p+1], NA)))
}

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
n_vars <-5
evalBudget <-15
n_init <- 5
lambda <- .2
minSpend <- 30
order <- 2
seed <- 11


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
