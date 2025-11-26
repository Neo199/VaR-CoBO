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
# SET INPUTS
# ---------------------------------------------------------
n_vars <-100
evalBudget <- 20
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

ga_model <- function(x) {
  x_mat <- matrix(x, nrow = 1)  # convert vector to 1-row matrix
  -model(x_mat, seed)
}

GA <- ga(type = "binary", fitness = ga_model, nBits = n_vars, 
         popSize = 100, maxiter = 1000, run = 100, monitor = FALSE)
summary(GA)
plot (GA)

# ---------------------------------------------------------
# RUN GA
# ---------------------------------------------------------

# Start timing for GA
start_time <- Sys.time()

ga_result <- ga(data, evalBudget, n_iter, n_vars, xTrain, xTrain_in, order)

end_time <- Sys.time()
bocssdp_elapsed_time <- end_time - start_time
