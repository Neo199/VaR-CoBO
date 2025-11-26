# RUN BOCS-SDP
# SIMULATION STUDY
# Function for BOCS-SDP


bocs_sdp <- function(data, evalBudget, n_iter, n_vars, xTrain, xTrain_in, order){
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
  hs_ss_sd <- sd(xTrain)
  slab_scale<-sqrt(0.3/p0)*hs_ss_sd 
  #global scale without sigma, as the scaling by sigma
  #is done inside stan_glm 
  global_scale<-(p0/(p-p0))/sqrt(n)
  bocssdp_bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(), 
                                     prior=hs(global_scale=global_scale, slab_scale=slab_scale), 
                                     prior_intercept = normal(0,1), iter = 1000, refresh = 0,
                                     cores = 4)
  
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
    y_new <- model(x_new)
    
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
                                       prior_intercept = normal(0,1), iter = 1000, refresh = 0,
                                       cores = 4)
  }
  
  result <- list(solution = tail(bocssdp_data, n =1), data = bocssdp_data, 
                 model = bocssdp_bayesian_model) 
  return(result)
}