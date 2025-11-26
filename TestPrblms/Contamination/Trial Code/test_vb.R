#Test file to figure out issues in svb 
#
#install.packages("/home/people/20204013/r450", lib = "/home/people/20204013/r450/lib64/R/library", repos = NULL, type = "source")

source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/Contamination/sonic/sample_models.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/Contamination/sonic/Contamination.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/Contamination/sonic/thompson_svb.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/Contamination/sonic/ordertheta_interaction.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/Contamination/sonic/prbocs_vb.R")

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

source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/Contamination/sonic/funs.common.R")
# ---------------------------------------------------------
# SET INPUTS
# ---------------------------------------------------------
n_vars <- 100
evalBudget <- 100
n_init <- 10
order <- 2
seed <- 1
instances <- 2

# ---------------------------------------------------------
# DEFINE TRUE MODEL
# ---------------------------------------------------------
model <- function(x_vals, seed){
  contamination_prob(x_vals, 100, seed) 
}

for(inst in 1:instances){
  starttime <- Sys.time()
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
  
  endtime <- Sys.time()
  print("Prbocs vb Instance", inst)
  print(prbocs_vb_result)
  prbocs_vb_elapsed_time <- starttime - endtime
  
  prbocs_vb <- list(instance_id = inst, prbocs_vb_result = prbocs_vb_result, 
                    time_taken = prbocs_vb_elapsed_time)
}

folder_name <- paste0("n_vars=", n_vars)
if (!dir.exists(folder_name)) {
  dir.create(folder_name)
}
saveRDS(prbocs_vb, file = file.path(folder_name, "simulation_prbocsvb_results.RData"))