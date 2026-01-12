# Function to compute thompson sampling
# this function is designed to be used with STAN mcmc
thompson_sam <- function(x_current, bayesian_model, removed_columns, data, order){
  
  # browser()
  coeffs <- list()
  
  # Add the removed columns back to the reduced data, filling with the constant values
  for (col in removed_columns) {
    unique_value <- unique(data[[col]])  # Using the unique value from the original data
    
    # Check if there's exactly one unique value to use as a constant
    if (length(unique_value) == 1) {
      # Add the column back to reduced_data, filling with the constant value
      bayesian_model$coefficients[[col]] <- unique_value
    } 
    else {
      stop(paste("Column", col, "does not have a single unique value. Cannot fill with a constant."))
    }
    
  }
  
  # Add a column of 1s to 'theta_current_in' to account for the intercept
  x_current_in <- theta_interaction(x_current, order, n_vars)
  x_current_in <- c(1, x_current_in)
  
  coeffs <- bayesian_model$coefficients
  y_pred <-  sum(x_current_in * coeffs)
  
  return(y_pred = y_pred)
}
