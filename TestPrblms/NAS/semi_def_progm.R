# ---------------------------------------------------------
# SEMIDEFINITE PROGRAMMING
# ---------------------------------------------------------

sdp_relaxation <- function(alpha, n_vars, lambda, removed_columns) {
  
  # Append missing variables back with zero coefficients
  alpha_vect <- c(alpha, setNames(rep(0, length(removed_columns)), removed_columns))
  
  # Reorder so that variables are in a consistent order
  alpha_vect <- alpha_vect[order(names(alpha_vect))]
  
  b <- alpha_vect[2:(n_vars + 1)] + lambda
  a <- alpha_vect[(n_vars + 2):length(alpha_vect)]
  
  # Create index for upper triangle (i < j)
  idx_prod <- t(combn(n_vars, 2))
  n_idx <- nrow(idx_prod)
  
  if (length(a) != n_idx) {
    stop("Number of Coefficients does not match the number of off-diagonal terms!")
  }
  
  # Convert a to symmetric matrix A (only off-diagonal terms)
  A <- matrix(0, n_vars, n_vars)
  for (i in 1:n_idx) {
    A[idx_prod[i, 1], idx_prod[i, 2]] <- a[i] / 2
    A[idx_prod[i, 2], idx_prod[i, 1]] <- a[i] / 2
  }
  
  bt <- b / 2 + A %*% rep(1, n_vars) / 2
  At <- rbind(
    cbind(A / 4, bt / 2),
    c(t(bt) / 2, 0)
  )
  
  # SDP with CVXR
  library(CVXR)
  X <- Variable(n_vars + 1, n_vars + 1, PSD = TRUE)
  objective <- Minimize(sum(At * X))  # trace(At %*% X)
  constraints <- list(
    diag(X) == 1
  )
  result <- solve(Problem(objective, constraints))
  X_val <- result$getValue(X)
  
  # Cholesky-like decomposition via eigen
  eig <- eigen(X_val, symmetric = TRUE)
  eig$values[eig$values < 0] <- 0  # safety for numerical issues
  L <- eig$vectors %*% diag(sqrt(eig$values))
  
  # Rounding
  n_rand_vector <- 100
  model_vect <- matrix(0, n_vars, n_rand_vector)
  obj_vect <- numeric(n_rand_vector)
  
  for (kk in 1:n_rand_vector) {
    r <- rnorm(n_vars + 1)
    r <- r / sqrt(sum(r^2))
    y_proj <- t(L) %*% r
    y_soln <- ifelse(y_proj >= 0, 1, -1)
    x_bin <- (y_soln[1:n_vars] + 1) / 2
    model_vect[, kk] <- x_bin
    obj_vect[kk] <- t(x_bin) %*% A %*% x_bin + sum(b * x_bin)
  }
  
  
  opt_idx <- which.min(obj_vect)
  model <- as.numeric(model_vect[, opt_idx])
  obj <- obj_vect[opt_idx]
  
  return(list(model = model, obj = obj))
}


# Old sdp code - it works but for smaller egs it doesn't optimise
# sdp_relaxation <- function(alpha, n_vars, lambda, removed_columns) {
#   
#   # Append missing variables back with zero coefficients
#   alpha_full <- c(alpha, setNames(rep(0, length(removed_columns)), removed_columns))
#   
#   # Reorder so that variables are in a consistent order
#   alpha_full <- alpha_full[order(names(alpha_full))]
#   
#   # Check results
#   print(alpha_full)
#   
#   # Extract vector of coefficients
#   b <- alpha_full[2:(n_vars + 1)] + lambda  # Skip intercept
#   a <- alpha_full[(n_vars + 2):(n_vars + 1 + choose(n_vars, 2))]  # Extract quadratic terms
#   
#   # Get indices for quadratic terms
#   idx_prod <- combn(n_vars, 2)
#   n_idx <- ncol(idx_prod)
#   
#   # Check number of coefficients
#   if (length(a) != n_idx) {
#     stop("Number of coefficients does not match indices!")
#   }
#   
#   # Convert a to matrix form
#   A <- matrix(0, n_vars, n_vars)
#   for (i in 1:n_idx) {
#     A[idx_prod[1, i], idx_prod[2, i]] <- a[i] / 2
#     A[idx_prod[2, i], idx_prod[1, i]] <- a[i] / 2
#   }
#   
#   # Convert to standard form
#   bt <- (b / 2) + (A %*% rep(1, n_vars)) / 2
#   At <- rbind(cbind(A / 4, bt / 2), c(bt / 2, 2))
#   
#   # Run SDP relaxation (approximate using eigenvalue decomposition)
#   X <- diag(n_vars + 1)  # Initial identity matrix
#   
#   # Enforce positive semi-definiteness using eigen decomposition
#   eig <- eigen(At)
#   eig$values[eig$values < 0] <- 1e-15  # Adjust negative eigenvalues
#   X <- eig$vectors %*% diag(eig$values) %*% t(eig$vectors)
#   
#   # Compute Cholesky decomposition
#   L <- chol(X)
#   
#   # Repeat rounding for different vectors
#   n_rand_vector <- 100
#   model_vect <- matrix(0, nrow = n_vars, ncol = n_rand_vector)
#   obj_vect <- numeric(n_rand_vector)
#   
#   for (kk in 1:n_rand_vector) {
#     # Generate a random cutting plane vector
#     r <- rnorm(n_vars + 1)
#     r <- r / sqrt(sum(r^2))  # Normalize vector
#     y_soln <- sign(t(L) %*% r)
#     
#     # Convert solution to original domain
#     model_vect[, kk] <- (y_soln[1:n_vars] + 1) / 2
#     obj_vect[kk] <- t(model_vect[, kk]) %*% A %*% model_vect[, kk] + sum(b * model_vect[, kk])
#   }
#   
#   # Find optimal rounded solution
#   opt_idx <- which.min(obj_vect)
#   model <- model_vect[, opt_idx]
#   obj <- obj_vect[opt_idx]
#   
#   return(list(model = model, obj = obj))
# }