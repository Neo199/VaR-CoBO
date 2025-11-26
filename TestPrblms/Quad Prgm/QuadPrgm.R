# This code generate the Q matrix for the quadratic programming 
# objective function.
# K value is dependent on the correlation length (L_c)
# The lambda parameter from the regualrization parameter can set 
# from the main.

# Author: Niyati Seth
# Date:  January 2025

quad_mat <- function(n_vars, corr_len) {
  # Decay function
  K <- function(s, t) exp(-((s - t)^2) / corr_len)
  
  # Compute the decay matrix
  decay <- matrix(0, n_vars, n_vars)
  for (i in 1:n_vars) {
    for (j in 1:n_vars) {
      decay[i, j] <- K(i, j)
    }
  }
  
  # Generate random quadratic model
  # and apply exponential decay to Q
  Q <- matrix(rnorm(n_vars * n_vars), n_vars, n_vars)
  Qa <- Q * decay
  
  return(Qa)
}
