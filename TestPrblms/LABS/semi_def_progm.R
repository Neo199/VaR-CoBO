# ---------------------------------------------------------
# SEMIDEFINITE PROGRAMMING
# ---------------------------------------------------------

sdp_relaxation <- function(alpha, n_vars, lambda, removed_columns) {
  
  # Append missing variables back with zero coefficients
  alpha_vect <- c(alpha, setNames(rep(0, length(removed_columns)), removed_columns))
  
  # Reorder so that variables are in a consistent order
  alpha_vect <- alpha_vect[order(names(alpha_vect))]
  
  # Extract coefficients
  b <- alpha_vect[2:(n_vars + 1)] + lambda
  a <- alpha_vect[(n_vars + 2):length(alpha_vect)]
  
  # Create index for upper triangle (i < j)
  idx_prod <- t(combn(n_vars, 2))
  n_idx <- nrow(idx_prod)
  
  if (length(a) != n_idx) {
    stop("Number of Coefficients does not match the number of off-diagonal terms!")
  }
  
  # Convert a to symmetric matrix A
  A <- matrix(0, n_vars, n_vars)
  for (i in 1:n_idx) {
    A[idx_prod[i, 1], idx_prod[i, 2]] <- a[i] / 2
    A[idx_prod[i, 2], idx_prod[i, 1]] <- a[i] / 2
  }
  
  # Construct the lifted matrix At (like in SDP)
  bt <- b / 2 + A %*% rep(1, n_vars) / 2
  At <- rbind(
    cbind(A / 4, bt / 2),
    c(t(bt) / 2, 0)
  )
  
  # Approximate PSD matrix X by projecting At onto PSD cone
  eig <- eigen(At, symmetric = TRUE)
  eig$values[eig$values < 0] <- 1e-12  # clip negative eigenvalues
  X_val <- eig$vectors %*% diag(eig$values) %*% t(eig$vectors)
  
  # Cholesky-like factorization
  eig_X <- eigen(X_val, symmetric = TRUE)
  eig_X$values[eig_X$values < 0] <- 0
  L <- eig_X$vectors %*% diag(sqrt(eig_X$values))
  
  # Randomized rounding
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
  
  opt_idx <- which.max(obj_vect)
  model <- as.numeric(model_vect[, opt_idx])
  obj <- obj_vect[opt_idx]
  
  return(list(model = model, obj = obj))
}
