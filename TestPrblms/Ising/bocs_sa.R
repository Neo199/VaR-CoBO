# RUN BOCS-SA
# SIMULATION STUDY
# Function for BOCS-SA

bocs_sa <- function(data, evalBudget, n_iter, n_vars, xTrain, xTrain_in, order){
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
  hs_ss_sd <- sd(xTrain)
  slab_scale<-sqrt(0.3/p0)*hs_ss_sd 
  #global scale without sigma, as the scaling by sigma
  #is done inside stan_glm 
  global_scale<-(p0/(p-p0))/sqrt(n)
  bocssa_bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(), 
                                    prior=hs(global_scale=global_scale, slab_scale=slab_scale), 
                                    prior_intercept = normal(0,1), iter = 1000, chains = 4, cores = 4, refresh = 0)
  
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
    y_new <- model(x_new, theta_P)
    
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
                                      prior_intercept = normal(0,1), iter = 1000, chains = 4, cores = 4, refresh = 0)
  }
  
  result <- list(solution = tail(bocssa_data, n =1), data = bocssa_data, 
                 model = bocssa_bayesian_model) 
  return(result)
}