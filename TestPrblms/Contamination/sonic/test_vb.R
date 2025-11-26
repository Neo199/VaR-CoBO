#Test file to figure out issues in svb 
#
#install.packages("/home/people/20204013/r450", lib = "/home/people/20204013/r450/lib64/R/library", repos = NULL, type = "source")

source("/home/people/20204013/sonic/sample_models.R")
source("/home/people/20204013/sonic/Contamination.R")
source("/home/people/20204013/sonic/thompson_svb.R")
source("/home/people/20204013/sonic/ordertheta_interaction.R")
source("/home/people/20204013/sonic/prbocs_vb.R")
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
library(selectiveInference)
library(bayesplot)


# ---------------------------------------------------------
# SET INPUTS
# ---------------------------------------------------------
n_vars <- 25
evalBudget <- 100
n_init <- 10
order <- 2
seed <- 1
instances <- 20

# ---------------------------------------------------------
# DEFINE TRUE MODEL
# ---------------------------------------------------------
model <- function(x_vals, seed){
  contamination_prob(x_vals, 100, seed) 
}

for(inst in 1:instances){
  
  set.seed(inst)
  
  # ---------------------------------------------------------
  # INITIAL SAMPLES FOR STATISTICAL MODELS
  # ---------------------------------------------------------
  x_vals <- sample_models(n_init, n_vars)
  y_vals <- model(x_vals, seed)
  
  x_vals
  y_vals
  
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
  
  prbocs_vb_result <- prbocs_vb_optim(data, evalBudget, n_iter,
                                      n_vars, xTrain, xTrain_in, 
                                      theta_current, order)
  print("Prbocs vb Instance", inst)
  print(prbocs_vb_result)
}

