
# Function to compute expected improvement wrt theta
# with monte carlo estimation
expected_improvement <- function(theta, model, f_best, N=100, order, n_vars) {
  # Sample N binary vectors from Bernoulli(theta)
  # Each row is a sample x^s
  # browser()
  theta_in <- theta_interaction(theta, order, n_vars)
  x_samples <- matrix(rbinom(N * length(theta_in), 1, theta_in), nrow = N)
  
  # allocate the array to store each EI value
  EI_vals <- numeric(N)
  
  # Monte Carlo loop: evaluate the EI for each sample
  for (i in 1:N) {
    # Get the sample x_i
    x_new <- x_samples[i, ]
    x_new <- data.frame(matrix(x_new, nrow = 1, ncol = ncol(theta_in)))
    
    # Generate posterior predictive samples
    posterior_samples <- posterior_predict(model, newdata = x_new)
    
    # Estimate the mean and standard deviation from posterior samples
    mu <- mean(posterior_samples)
    sigma <- sd(posterior_samples)
    
    # Prevent numerical issues by setting a lower bound on sigma
    sigma[sigma < 1e-9] <- 1e-9
    
    # Compute Z = (f_best - mu_x) / sigma_x
    Z <- (f_best - mu) / sigma
    
    # EI formula for minimization
    EI_vals[i] <- (f_best - mu) * pnorm(Z) + sigma * dnorm(Z)
  }
  
  EI <- mean(EI_vals)
  return(-EI) #since we want to maximise
}