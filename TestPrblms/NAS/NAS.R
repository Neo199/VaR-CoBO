seed <- 9

source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/NAS/Oh_NAS.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/NAS/sample_models.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/NAS/thompson.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/NAS/thompson_svb.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/NAS/ordertheta_interaction.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/NAS/sim_anneal.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/NAS/semi_def_progm.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/NAS/prbocs.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/NAS/prbocs_vb.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/NAS/prbocs_ga.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/NAS/prbocs_vb_ga.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/NAS/bocs_ga.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/NAS/bocs_sa.R")
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/NAS/bocs_sdp.R")
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
library(bayesplot)
library(parallel)
library(sparsevb)

# ---------------------------------------------------------
# DEFINE TRUE MODEL
# ---------------------------------------------------------
model <- function(x_vals, seed){
  para <- evaluate_binary_architectures(x_vals, mode = "combined", alpha = 1, beta = 1) 
  return(para$score)
}

node_sizes <- c(16, 32, 64)
L <- 3
d <- L * (length(node_sizes) + 1)
n_vars <- d

evalBudget <- 200
n_init <- 10
order <- 2


set.seed(seed)
X <- sample_models(n_init, d)
# Evaluate loss
NN <- evaluate_binary_architectures(X, mode = "combined", alpha = 1, beta = 1)

x_vals <- X
y_vals <- NN$score


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
# RUN PRBOCS-VB
# ---------------------------------------------------------
# Start timing for BOCS-GA
start_time <- Sys.time()

prbocs_vb_result_acc <- prbocs_vb_optim_max(data, evalBudget, n_iter, n_vars, xTrain, xTrain_in, theta_current, order)
prbocs_vb_result <- prbocs_vb_optim(data, evalBudget, n_iter, n_vars, xTrain, xTrain_in, theta_current, order)


# End timing
end_time <- Sys.time()
prbocs_vb_elapsed_time <- end_time - start_time

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
start_time <- Sys.time()

prbocsga_result <- prbocs_ga(data, evalBudget, n_iter, n_vars, xTrain, xTrain_in, theta_current, order)    

# End timing
end_time <- Sys.time()
prbocsga_elapsed_time <- end_time - start_time

# ---------------------------------------------------------
# RUN PRBOCS-VB-GA
# ---------------------------------------------------------
# Start timing for BOCS-GA
start_time <- Sys.time()
browser()
prbocs_vb_ga_result <- prbocs_vb_ga(data, evalBudget, n_iter, n_vars, xTrain, xTrain_in, theta_current, order)

# End timing
end_time <- Sys.time()
prbocs_vb_ga_elapsed_time <- end_time - start_time

# ---------------------------------------------------------
# RUN BOCS-GA
# ---------------------------------------------------------

# Start timing for BOCS-GA
start_time <- Sys.time()

bocsga_result <- bocs_ga(data, evalBudget, n_iter, n_vars, xTrain, xTrain_in, order)

# End timing
end_time <- Sys.time()
bocsga_elapsed_time <- end_time - start_time


# ---------------------------------------------------------
# RUN BOCS-SA
# ---------------------------------------------------------

# Start timing for BOCS-SA
start_time <- Sys.time()

bocssa_result <- bocs_sa(data, evalBudget, n_iter, n_vars, xTrain, xTrain_in, order)

end_time <- Sys.time()
bocssa_elapsed_time <- end_time - start_time

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
  -model(x_mat, seed)
}

GA_run <- ga(type = "binary", fitness = ga_model, nBits = n_vars, 
             popSize = 100, maxiter = 1000, run = 100, monitor = FALSE)
ga_result <- list(solution = GA_run@solution, fitness_value = GA_run@fitnessValue)

end_time <- Sys.time()
ga_elapsed_time <- end_time - start_time

#-----------------------------------------------------------
# Capture and return the result for the instance
#-----------------------------------------------------------

prbocs <- list(instance_id = instance_id, prbocs_result = prbocs_result, 
               time_taken = prbocs_elapsed_time)
prbocsga <- list(instance_id = instance_id, prbocsga_result = prbocs_result,
                 time_taken = prbocsga_elapsed_time)
prbocs_vb <- list(instance_id = instance_id, prbocs_vb_result = prbocs_vb_result, 
                  time_taken = prbocs_vb_elapsed_time)
prbocs_vb_ga <- list(instance_id = instance_id, prbocs_vb_ga_result = prbocs_vb_ga_result, 
                     time_taken = prbocs_vb_ga_elapsed_time)
bocsga <- list(instance_id = instance_id, bocsga_result = bocsga_result,
               time_taken = bocsga_elapsed_time)
bocssa <- list(instance_id = instance_id, bocssa_result = bocssa_result,
               time_taken = bocssa_elapsed_time)
bocssdp <- list(instance_id = instance_id, bocssdp_result = bocssdp_result,
                time_taken = bocssdp_elapsed_time)
ga <- list(instance_id = instance_id, ga_result = ga_result,
           time_taken = ga_elapsed_time)
result <- list(PRBOCS = prbocs, PRBOCS_GA = prbocsga, BOCS_GA = bocsga, 
               PRBOCS_VB = prbocs_vb, PRBOCS_VB_GA = prbocs_vb_ga, 
               BOCS_SA = bocssa, BOCS_SDP = bocssdp, GA = ga)


plot(c(1:evalBudget), prbocs_vb_result_acc$data$y, "l")

library(ggplot2)

# Create data frame
df <- data.frame(
  Iteration = 1:evalBudget,
  Accuracy = prbocs_vb_result_acc$data$y
)

# Generate the plot
ggplot(df, aes(x = Iteration, y = Accuracy)) +
  geom_line(color = "#1f78b4", size = 1.2) +
  theme_minimal(base_size = 14) +
  labs(
    title = "PRBOCS-VB Optimisation Performance",
    x = "Evaluation Budget",
    y = "Test Accuracy"
  ) +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    panel.grid.major = element_line(color = "gray90"),
    panel.grid.minor = element_blank()
  )

