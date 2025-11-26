# Parallizing vcdoe form IT test

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

source("~/sonic/sample_models.R")
source("~/sonic/Contamination.R")
source("~/sonic/thompson.R")
source("~/sonic/thompson_svb.R")
source("~/sonic/ordertheta_interaction.R")
source("~/sonic/sim_anneal.R")
source("~/sonic/semi_def_progm.R")
source("~/sonic/prbocs.R")
source("~/sonic/prbocs_vb.R")
source("~/sonic/prbocs_ga.R")
source("~/sonic/prbocs_vb_ga.R")
source("~/sonic/bocs_ga.R")
source("~/sonic/bocs_sa.R")
source("~/sonic/bocs_sdp.R")
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
library(doParallel)
library(foreach)

# ---------------------------------------------------------
# DEFINE TRUE MODEL
# ---------------------------------------------------------
model <- function(x_vals, seed){
  contamination_prob(x_vals, 100, seed) 
}

run_simulation_instance <- function(instance_id, n_vars, evalBudget, n_init, order,
                                    seed, folder_name) {
  tryCatch({
    inst_seed <- seed + instance_id
    set.seed(inst_seed)

    # Initial samples
    x_vals <- sample_models(n_init, n_vars)
    y_vals <- contamination_prob(x_vals, 100, seed)

    # Prepare training data
    xTrain <- x_vals
    yTrain <- y_vals
    xTrain_in_comb <- order_effects(xTrain, order)
    xTrain_in <- xTrain_in_comb$xTrain_in

    data <- data.frame(y = yTrain, xTrain_in)
    theta_ini <- rep(0.5, ncol(xTrain))
    theta_current <- theta_ini

    # -------------------------------
    # Setup parallel backend
    # -------------------------------
    ncores <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", unset="1"))   # will equal SLURM_CPUS_PER_TASK if exported properly
    cl <- makeCluster(ncores)
    registerDoParallel(cl)

    clusterEvalQ(cl, {
	library(GA)
	library(rstan)
	library(rstanarm)
	library(dplyr)
	library(sparsevb)
	source("~/sonic/sample_models.R")
	source("~/sonic/Contamination.R")
	source("~/sonic/thompson.R")
	source("~/sonic/thompson_svb.R")
	source("~/sonic/ordertheta_interaction.R")
	source("~/sonic/sim_anneal.R")
	source("~/sonic/semi_def_progm.R")
	source("~/sonic/prbocs.R")
	source("~/sonic/prbocs_vb.R")
	source("~/sonic/prbocs_ga.R")
	source("~/sonic/prbocs_vb_ga.R")
	source("~/sonic/bocs_ga.R")
	source("~/sonic/bocs_sa.R")
	source("~/sonic/bocs_sdp.R")
	NULL
	})

    # List of methods to run
    method_list <- list(
      PRBOCS      = function() prbocs_optim(data, evalBudget, evalBudget - n_init, n_vars, xTrain, xTrain_in, theta_current, order),
      PRBOCS_GA   = function() prbocs_ga(data, evalBudget, evalBudget - n_init, n_vars, xTrain, xTrain_in, theta_current, order),
      PRBOCS_VB   = function() prbocs_vb_optim(data, evalBudget, evalBudget - n_init, n_vars, xTrain, xTrain_in, theta_current, order),
      PRBOCS_VB_GA= function() prbocs_vb_ga(data, evalBudget, evalBudget - n_init, n_vars, xTrain, xTrain_in, theta_current, order),
      BOCS_GA     = function() bocs_ga(data, evalBudget, evalBudget - n_init, n_vars, xTrain, xTrain_in, order),
      BOCS_SA     = function() bocs_sa(data, evalBudget, evalBudget - n_init, n_vars, xTrain, xTrain_in, order),
      BOCS_SDP    = function() bocs_sdp(data, evalBudget, evalBudget - n_init, n_vars, xTrain, xTrain_in, order),
      GA          = function() {
        ga_model <- function(x) {
          x_mat <- matrix(x, nrow = 1)
          -model(x_mat, seed)
        }
        GA_run <- ga(type = "binary", fitness = ga_model, nBits = n_vars,
                     popSize = 100, maxiter = 1000, run = 100, monitor = FALSE)
        list(solution = GA_run@solution, fitness_value = GA_run@fitnessValue)
      }
    )

    # Export needed objects from this function to workers

    clusterExport(cl, varlist = c("data", "evalBudget", "n_init", "n_vars",
                                  "xTrain", "xTrain_in", "theta_current",
                                  "order", "seed", "method_list", "model",
                                  "contamination_prob", "sample_models", "order_effects"),
                envir = environment())

    # -------------------------------
    # Run methods in parallel
    # -------------------------------
    results <- foreach(method = names(method_list), .combine = "list",
                       .multicombine = TRUE, .inorder = TRUE,
                       .packages = c("GA","rstan","rstanarm","dplyr", "sparsevb")) %dopar% {
      start_time <- Sys.time()
      res <- method_list[[method]]()
      end_time <- Sys.time()
      list(
        instance_id = instance_id,
        method = method,
        result = res,
        time_taken = end_time - start_time
      )
    }

    stopCluster(cl)

    # Convert results into a named list (like your original)
    names(results) <- names(method_list)
    return(results)

  }, error = function(e) {
    message(paste("Error in instance:", instance_id, ":", e$message))
    return(NULL)
  })
}


# ---------------------------------------------------------
# SET INPUTS
# ---------------------------------------------------------
n_vars <- 25
evalBudget <- 250
n_init <- 10
order <- 2
seed <- 1
num_instances <- 20

# --------------------------
# Run the simulation study
#---------------------------

folder_name <- paste0("n_vars=", n_vars)
if (!dir.exists(folder_name)) {
  dir.create(folder_name)
}

args <- commandArgs(trailingOnly = TRUE)
if(length(args) == 0 || is.na(as.integer(args[1]))) {
  # fallback default for interactive runs
  instance_id <- 1
  message("No instance ID argument supplied, defaulting to instance_id = 1")
} else {
  instance_id <- as.integer(args[1])
}

cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "- Starting instance", instance_id, "\n")

res <- run_simulation_instance(instance_id, n_vars, evalBudget, n_init, order, seed, folder_name)

saveRDS(res, file = file.path(folder_name, paste0("instance_", instance_id, ".RData")))

cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "- Finished instance", instance_id, "\n")

