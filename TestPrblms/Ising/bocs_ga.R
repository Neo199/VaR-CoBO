# RUN BOCS-GA
# SIMULATION STUDY
# Function for BOCS-GA
 
bocs_ga <- function(data, evalBudget, n_iter, n_vars, xTrain, xTrain_in, order){
  # Initialize a data frame to store iteration results
  bocsga_result <- matrix(0, evalBudget, n_vars)
  
  #Initialise BOCS-GA specific dataframe
  bocsga_data <- data
  
  #Initializing the bayesian model with the given data first
  # Check for constant columns
  # Remove constant columns
  is_constant <- apply(bocsga_data, 2, function(x) length(unique(x)) == 1)
  data_reduced <- bocsga_data[, !is_constant]
  # Save the names of the columns that were removed
  removed_columns <- names(bocsga_data)[is_constant]
  
  #Horseshoe scaling using piironen and vehtari 2017
  p0 <- n_vars/2
  n <- nrow(xTrain_in)
  p <- ncol(xTrain_in)
  hs_ss_sd <- sd(xTrain)
  slab_scale<-sqrt(0.3/p0)*hs_ss_sd 
  #global scale without sigma, as the scaling by sigma
  #is done inside stan_glm 
  global_scale<-(p0/(p-p0))/sqrt(n)
  bocsga_bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(), 
                                        prior=hs(global_scale=global_scale, slab_scale=slab_scale), 
                                        prior_intercept = normal(0,1), iter = 1000, chains = 4, cores = 4, refresh = 0)
  # Generate a random binary vector
  x_current <-sample(c(0, 1), size = n_vars, replace = TRUE)
  
  for (t in 1:n_iter) {
    print(paste("bocsga_iteration_",t))
    stat_model <- function(x_current) {
      thompson_sam(x_current, bayesian_model = bocsga_bayesian_model, removed_columns, data, order =2)
    }
    
    ga_model <- function(x_current) {
      f <- -(stat_model(x_current))
      (f) # - penalty1 - penalty2)            # fitness function value
    }
    
    GA <- ga(type = "binary", fitness = ga_model, nBits = n_vars, 
             popSize = 100, maxiter = 1000, run = 100, monitor = FALSE)
    # summary(GA)
    # plot (GA)
    
    
    # Evaluate model objective at the new evaluation point
    # Truncate x_new to the first n_vars elements
    x_new <- GA@solution
    
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
    bocsga_data <- rbind(bocsga_data, data_new)
    
    
    hs_ss_sd <-sd(x_vals_updated)
    
    x_current <- x_new
    bocsga_result[t,] <- x_new
    
    #Running the Ling reg to train on constrained data
    is_constant <- apply(bocsga_data, 2, function(x) length(unique(x)) == 1)
    data_reduced <- bocsga_data[, !is_constant]
    # Save the names of the columns that were removed
    removed_columns <- names(bocsga_data)[is_constant]
    
    #Horseshoe scaling using piironen and vehtari 2017
    p0 <- n_vars/2
    n <- nrow(xTrain_in)
    p <- ncol(xTrain_in)
    slab_scale<-sqrt(0.3/p0)*hs_ss_sd 
    #global scale without sigma, as the scaling by sigma
    #is done inside stan_glm 
    global_scale<-(p0/(p-p0))/sqrt(n)
    
    bocsga_bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(), 
                                      prior=hs(global_scale=global_scale, slab_scale=slab_scale), 
                                      prior_intercept = normal(0,1), iter = 1000, chains = 4, cores = 4, refresh = 0)
  }
  
  result <- list(solution = tail(bocsga_data, n =1), data = bocsga_data, 
                 model = bocsga_bayesian_model) 
  return(result)
} 