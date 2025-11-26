# ---------------------------------------------------------
# Warehouse data generator
# ---------------------------------------------------------
Generate_WLP_data <- function(n_customers, n_warehouses, seed = 1) {
  set.seed(seed)
  
  # Random fixed opening costs for warehouses
  f <- round(runif(n_warehouses, 50, 200), 2)
  
  # Random supply cost matrix (customers x warehouses)
  C <- matrix(round(runif(n_customers * n_warehouses, 5, 50), 2),
              nrow = n_customers, ncol = n_warehouses)
  
  list(f = f, C = C)
}


# ---------------------------------------------------------
# Binary-friendly Warehouse Location Objective
# ---------------------------------------------------------
WarehouseRelax_binary <- function(z, data, lambda = 1, rho = 50) {
  f <- data$f
  C <- data$C
  I <- nrow(C)
  J <- ncol(C)
  
  # Extract y (first J vars) and x (remaining I*J vars)
  y <- z[1:J]
  x <- matrix(z[(J + 1):(J + I * J)], nrow = I, ncol = J, byrow = TRUE)
  
  # Penalize violation of x_ij <= y_j (elementwise)
  penalty_violation <- sum(pmax(x - matrix(rep(y, each = I), nrow = I), 0))
  
  # Lagrangian-relaxed objective
  obj <- sum(f * y) + sum(C * x) + lambda * sum(y) +
    rho * sum((rowSums(x) - 1)^2) + 1e4 * penalty_violation
  
  return(obj)
}


# ---------------------------------------------------------
# Wrapper for Bayesian Optimization (like contamination_prob)
# ---------------------------------------------------------
warehouse_prob <- function(X, data, lambda = 1, rho = 50) {
  n_samples <- nrow(X)
  out <- numeric(n_samples)
  
  for (i in 1:n_samples) {
    out[i] <- WarehouseRelax_binary(X[i, ], data, lambda, rho)
  }
  return(out)
}

# 
# data <- Generate_WLP_data(n_customers = 5, n_warehouses = 3)
# 
# # Number of binary variables = J (for y) + I*J (for x)
# n_vars <- length(data$f) + nrow(data$C) * ncol(data$C)
# 
# # Random binary candidates
# set.seed(123)
# X <- matrix(sample(c(0,1), 10 * n_vars, replace = TRUE), nrow = 10)
# 
# # Evaluate
# out <- warehouse_prob(X, data)
# print(out)

