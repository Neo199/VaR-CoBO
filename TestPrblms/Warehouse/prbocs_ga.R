# RUN PRBOCS-GA
# SIMULATION STUDY
# Function for PRBOCS-GA


prbocs_ga <- function(data, evalBudget, n_iter, n_vars, xTrain, xTrain_in, theta_current, order){
  # Initialize a data frame to store iteration results
  prbocsga_result <- matrix(0, evalBudget, n_vars)
  
  #Initialise BOCS-GA specific dataframe
  prbocsga_data <- data
  
  #Initializing the bayesian model with the given data first
  # Check for constant columns
  # Remove constant columns
  is_constant <- apply(prbocsga_data, 2, function(x) length(unique(x)) == 1)
  data_reduced <- prbocsga_data[, !is_constant]
  # Save the names of the columns that were removed
  removed_columns <- names(prbocsga_data)[is_constant]
  
  #Horseshoe scaling using piironen and vehtari 2017
  p0 <- n_vars/2
  n <- nrow(xTrain_in)
  p <- ncol(xTrain_in)
  hs_ss_sd <- sd(xTrain)
  slab_scale<-sqrt(0.3/p0)*hs_ss_sd 
  #global scale without sigma, as the scaling by sigma
  #is done inside stan_glm 
  global_scale<-(p0/(p-p0))/sqrt(n)
  prbocsga_bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(), 
                                      prior=hs(global_scale=global_scale, slab_scale=slab_scale), 
                                      prior_intercept = normal(0,1), iter = 1000, refresh = 0,
                                      cores = 4)
  for (t in 1:n_iter) {
    print(paste("prbocsga_iteration_",t))
    
    x_current <- rbinom(n = n_vars, size = 1, prob = theta_current)
    
    stat_model <- function(theta) {
      con_y <- thompson_sam(theta, bayesian_model = prbocsga_bayesian_model, removed_columns, prbocsga_data, order = 2)
    }
    
    ga_model <- function(theta) {
      f <- -(stat_model(theta))
      (f)                 # fitness function value
    }
    
    
    GA <- ga(type = "real-valued", fitness = ga_model, lower = rep(0, n_vars), upper = rep(1, n_vars),
             popSize = 100, maxiter = 1000, run = 100, monitor = FALSE)
    # plot(GA)
    # summary(GA)
    
    expected_val <- GA@solution
    cat("New expectation of evaluation point", expected_val, "\n")
    
    x_new <- rbinom(length(expected_val), 1, expected_val)
    # Evaluate model objective at the new evaluation point
    # Truncate x_new to the first n_vars elements
    x_new <- x_new[1:n_vars]
    
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
    prbocsga_data <- rbind(prbocsga_data, data_new)
    
    
    hs_ss_sd <-sd(x_vals_updated)
    
    theta_current <- expected_val
    
    # optim_result[t,] <- expected_val
    
    #Running the Ling reg to train on constrained data
    is_constant <- apply(prbocsga_data, 2, function(x) length(unique(x)) == 1)
    data_reduced <- prbocsga_data[, !is_constant]
    # Save the names of the columns that were removed
    removed_columns <- names(prbocsga_data)[is_constant]
    
    #Horseshoe scaling using piironen and vehtari 2017
    p0 <- n_vars/2
    n <- nrow(xTrain_in)
    p <- ncol(xTrain_in)
    slab_scale<-sqrt(0.3/p0)*hs_ss_sd 
    #global scale without sigma, as the scaling by sigma
    #is done inside stan_glm 
    global_scale<-(p0/(p-p0))/sqrt(n)
    prbocsga_bayesian_model <- stan_glm(y ~ ., data = data_reduced, family = gaussian(), 
                                        prior=hs(global_scale=global_scale, slab_scale=slab_scale), 
                                        prior_intercept = normal(0,1), iter = 1000, refresh = 0,
                                        cores = 4)
    }
  
  result <- list(solution = tail(prbocsga_data, n =1), data = prbocsga_data, 
                 model = prbocsga_bayesian_model) 
  return(result)
  
}