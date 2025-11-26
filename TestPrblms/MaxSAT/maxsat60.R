# SIMULATION STUDY
# MaxSAT PROBLEM FROM BOCS PAPER
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
# No constraints are actually included in the solver(only in the orignal MaxSAT model)

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

source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/MaxSAT.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/sample_models.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/thompson.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/thompson_svb.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/ordertheta_interaction.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/sim_anneal.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/semi_def_progm.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/prbocs.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/prbocs_vb.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/prbocs_ga.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/prbocs_vb_ga.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/bocs_ga.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/bocs_sa.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/bocs_sdp.R")
library(GA)
library(psych)
library(lpSolve)
library(ompr)
library(ompr.roi)
library(ggplot2)
library(sparsevb)
library(selectiveInference)
library(dplyr)
library(ROI)
library(GPfit)
library(rstanarm)
library(bayesplot)
library(parallel)
library(CVXR)

# ---------------------------------------------------------
# DEFINE TRUE MODEL
# ---------------------------------------------------------

maxsat_obj <- read_maxsat("~/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/maxsat.wcnf")

model <- function(x_vals){
  maxsat(x_vals, maxsat_obj, seed = 1) 
}

# ---------------------------------------------------------
# FUNCTION TO RUN INSTANCE
# ---------------------------------------------------------
run_simulation_instance <- function(instance_id, n_vars, evalBudget, n_init, order,
                                    seed, folder_name) {
  tryCatch({
    # Set the seed for reproducibility
    inst_seed <- seed + instance_id
    set.seed(inst_seed)
    
    # ---------------------------------------------------------
    # INITIAL SAMPLES FOR STATISTICAL MODELS
    # ---------------------------------------------------------
    x_vals <- sample_models(n_init, n_vars)
    y_vals <- model(x_vals)
    
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
    
    n <- nrow(xTrain_in)
    p <- ncol(xTrain_in)
    
    #Setup dataframe for stan_glm training 
    X <- xTrain_in
    y <- y_vals
    
    #Create a initial dataframe
    data <- data.frame(y=y , X)
    
    #Initialise theta values
    theta_ini <- rep(0.5, ncol(xTrain))
    theta_current <- theta_ini
    
    # browser()
    # ---------------------------------------------------------
    # RUN PRBOCS
    # ---------------------------------------------------------
    # Start timing for BOCS-GA
    start_time <- Sys.time()
    
    prbocs_result <- prbocs_optim(data, evalBudget, n_iter, n_vars, xTrain, xTrain_in, theta_current, order)
    
    # End timing
    end_time <- Sys.time()
    prbocs_elapsed_time <- end_time - start_time
    
    
    # ---------------------------------------------------------
    # RUN PRBOCS-GA
    # ---------------------------------------------------------
    # Start timing for BOCS-GA
    # start_time <- Sys.time()
    # 
    # prbocsga_result <- prbocs_ga(data, evalBudget, n_iter, n_vars, xTrain, xTrain_in, theta_current, order)    
    # 
    # # End timing
    # end_time <- Sys.time()
    # prbocsga_elapsed_time <- end_time - start_time
    
    # ---------------------------------------------------------
    # RUN PRBOCS-VB
    # ---------------------------------------------------------
    # Start timing for BOCS-GA
    start_time <- Sys.time()
    
    prbocs_vb_result <- prbocs_vb(data, evalBudget, n_iter, n_vars, xTrain, xTrain_in, theta_current, order)
    
    # End timing
    end_time <- Sys.time()
    prbocs_vb_elapsed_time <- end_time - start_time
    
    # ---------------------------------------------------------
    # RUN PRBOCS-VB-GA
    # ---------------------------------------------------------
    # Start timing for BOCS-GA
    start_time <- Sys.time()
    
    prbocs_vb_ga_result <- prbocs_vb_ga(data, evalBudget, n_iter, n_vars, xTrain, xTrain_in, theta_current, order)
    
    # End timing
    end_time <- Sys.time()
    prbocs_vb_ga_elapsed_time <- end_time - start_time
    
    # # ---------------------------------------------------------
    # # RUN BOCS-GA
    # # ---------------------------------------------------------
    # 
    # # Start timing for BOCS-GA
    # start_time <- Sys.time()
    # 
    # bocsga_result <- bocs_ga(data, evalBudget, n_iter, n_vars, xTrain, xTrain_in, order)
    # 
    # # End timing
    # end_time <- Sys.time()
    # bocsga_elapsed_time <- end_time - start_time
    # 
    # 
    # # ---------------------------------------------------------
    # # RUN BOCS-SA
    # # ---------------------------------------------------------
    # 
    # # Start timing for BOCS-SA
    start_time <- Sys.time()

    bocssa_result <- bocs_sa(data, evalBudget, n_iter, n_vars, xTrain, xTrain_in, order)

    end_time <- Sys.time()
    bocssa_elapsed_time <- end_time - start_time
    # 
    # ---------------------------------------------------------
    # RUN BOCS-SDP
    # ---------------------------------------------------------
    
    # Start timing for BOCS-SDP
    start_time <- Sys.time()

    bocssdp_result <- bocs_sdp(data, evalBudget, n_iter, n_vars, xTrain, xTrain_in, order)

    end_time <- Sys.time()
    bocssdp_elapsed_time <- end_time - start_time
    
    # ---------------------------------------------------------
    # RUN GA
    # ---------------------------------------------------------
    
    # Start timing for GA
    start_time <- Sys.time()
    
    ga_model <- function(x) {
      x_mat <- matrix(x, nrow = 1)  # convert vector to 1-row matrix
      -model(x_mat)
    }
    
    GA_run <- ga(type = "binary", fitness = ga_model, nBits = n_vars, 
                 popSize = 100, maxiter = 1000, run = 100, monitor = FALSE)
    ga_result <- list(solution = GA_run@solution, fitness_value = GA_run@fitnessValue)
    
    end_time <- Sys.time()
    ga_elapsed_time <- end_time - start_time
    
    #-----------------------------------------------------------
    # Capture and return the result for the instance
    prbocs <- list(instance_id = instance_id, prbocs_result = prbocs_result, 
                   time_taken = prbocs_elapsed_time)
    # prbocsga <- list(instance_id = instance_id, prbocsga_result = prbocsga_result,
    #                  time_taken = prbocsga_elapsed_time)
    prbocs_vb <- list(instance_id = instance_id, prbocs_vb_result = prbocs_vb_result, 
                      time_taken = prbocs_vb_elapsed_time)
    prbocs_vb_ga <- list(instance_id = instance_id, prbocs_vb_ga_result = prbocs_vb_ga_result, 
                         time_taken = prbocs_vb_ga_elapsed_time)
    # bocsga <- list(instance_id = instance_id, bocsga_result = bocsga_result,
    #                time_taken = bocsga_elapsed_time)
    bocssa <- list(instance_id = instance_id, bocssa_result = bocssa_result,
                   time_taken = bocssa_elapsed_time)
    bocssdp <- list(instance_id = instance_id, bocssdp_result = bocssdp_result,
                   time_taken = bocssdp_elapsed_time)
    ga <- list(instance_id = instance_id, ga_result = ga_result,
               time_taken = ga_elapsed_time)
    # result <- list(PRBOCS = prbocs, PRBOCS_GA = prbocsga, BOCS_GA = bocsga, 
    #                PRBOCS_VB = prbocs_vb, PRBOCS_VB_GA = prbocs_vb_ga, 
    #                BOCS_SA = bocssa, BOCS_SDP = bocssdp, GA = ga)
    result <- list(PRBOCS = prbocs, PRBOCS_VB = prbocs_vb,
                   PRBOCS_VB_GA = prbocs_vb_ga, BOCS_SA = bocssa, BOCS_SDP = bocssdp, GA = ga)
    
    return(result)
  },error = function(e) {
    message(paste("Error in instance:", instance_id, ":", e$message))
    return(NULL)  # Return NULL if an error occurs
  })
}

# ---------------------------------------------------------
# SET INPUTS
# ---------------------------------------------------------
n_vars <- 60
evalBudget <- 270
n_init <- 20
order <- 2
seed <- 1
num_instances <- 1

# Create results folder if it doesn't exist
folder_name <- paste0("maxsat_n_vars=", n_vars)
if (!dir.exists(folder_name)) dir.create(folder_name)

# ---------------------------------------------------------
# RUN SIMULATIONS IN PARALLEL
# ---------------------------------------------------------
cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "- Starting parallel simulation...\n")

# Detect available cores (Mac uses forked processes → no Windows issues)
n_cores <- parallel::detectCores() - 1  # leave one core free
cat("Using", n_cores, "cores\n")

# Run instances in parallel
parallel::mclapply(
  X = 1:num_instances,
  FUN = function(instance_id) {
    cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
        "- Starting instance", instance_id, "\n")
    
    res <- run_simulation_instance(
      instance_id = instance_id,
      n_vars = n_vars,
      evalBudget = evalBudget,
      n_init = n_init,
      order = order,
      seed = seed,
      folder_name = folder_name
    )
    
    # Save each instance result
    saveRDS(res, file = file.path(folder_name, paste0("instance_", instance_id, ".RData")))
    
    cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
        "- Finished instance", instance_id, "\n")
    return(TRUE)
  },
  mc.cores = n_cores
)

cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "- All instances completed.\n")


