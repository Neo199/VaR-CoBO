# ===========================================
# Minimal Ising KL divergence in R
# Matches MATLAB's "KL_divergence_ising" logic
# For n = 16 (4x4 grid)
# ===========================================

# 1. Generate a random Ising grid
rand_ising_grid <- function(n_vars) {
  n_side <- sqrt(n_vars)
  if (floor(n_side) != n_side) stop("Number of nodes is not square")
  Q <- matrix(0, n_vars, n_vars)
  
  # Horizontal edges
  for (i in 1:n_side) {
    for (j in 1:(n_side - 1)) {
      node <- (i - 1) * n_side + j
      Q[node, node + 1] <- runif(1, 0.05, 5.0)
      Q[node + 1, node] <- Q[node, node + 1]
    }
  }
  
  # Vertical edges
  for (i in 1:(n_side - 1)) {
    for (j in 1:n_side) {
      node <- (i - 1) * n_side + j
      Q[node, node + n_side] <- runif(1, 0.05, 5.0)
      Q[node + n_side, node] <- Q[node, node + n_side]
    }
  }
  
  # Random sign flips
  rand_sign <- matrix(sample(c(-1, 1), n_vars * n_vars, replace = TRUE), n_vars, n_vars)
  rand_sign[upper.tri(rand_sign, diag = TRUE)] <- 0
  rand_sign <- rand_sign + t(rand_sign)
  Q <- Q * rand_sign
  
  return(Q)
}

# 2. Compute moments E[z_i z_j]
ising_model_moments <- function(Q) {
  n_vars <- nrow(Q)
  configs <- as.matrix(expand.grid(rep(list(c(-1, 1)), n_vars)))
  weights <- apply(configs, 1, function(z) exp(t(z) %*% Q %*% z))
  Z <- sum(weights)
  
  moments <- matrix(0, n_vars, n_vars)
  for (i in 1:n_vars) {
    for (j in 1:n_vars) {
      moments[i, j] <- sum(configs[, i] * configs[, j] * weights) / Z
    }
  }
  return(moments)
}

# 3. Get edge list in MATLAB order (lower-triangular, col-major)
get_edge_list <- function(Q) {
  mask <- lower.tri(Q, diag = FALSE) & (Q != 0)
  idx <- which(mask, arr.ind = TRUE)
  idx <- idx[order(idx[, 2], idx[, 1]), ]  # sort by col, then row
  return(idx)  # each row = (i, j)
}

# 4. KL divergence function
KL_divergence_ising <- function(Theta_P, moments, edges, x) {
  n_vars <- nrow(Theta_P)
  configs <- as.matrix(expand.grid(rep(list(c(-1, 1)), n_vars)))
  
  # Partition function for P
  P_vals <- apply(configs, 1, function(z) exp(t(z) %*% Theta_P %*% z))
  Zp <- sum(P_vals)
  
  # Ensure x is a matrix
  if (is.vector(x)) x <- matrix(x, nrow = 1)
  KL_vals <- numeric(nrow(x))
  
  for (row in 1:nrow(x)) {
    Theta_Q <- matrix(0, n_vars, n_vars)
    for (e in 1:nrow(edges)) {
      i <- edges[e, 1]
      j <- edges[e, 2]
      Theta_Q[i, j] <- x[row, e] * Theta_P[i, j]
      Theta_Q[j, i] <- Theta_Q[i, j]
    }
    
    # Partition function for Q
    Q_vals <- apply(configs, 1, function(z) exp(t(z) %*% Theta_Q %*% z))
    Zq <- sum(Q_vals)
    
    # KL computation
    KL_vals[row] <- sum((Theta_P - Theta_Q) * moments) + log(Zq) - log(Zp)
  }
  
  return(KL_vals)
}

# ===========================================
# Example usage
# ===========================================
# set.seed(1)
# n_vars <- 9
# Theta_P <- rand_ising_grid(n_vars)
# moments <- ising_model_moments(Theta_P)
# edges <- get_edge_list(Theta_P)  # 12 edges for 3x3
# 
# # Random edge vector (0 = removed, 1 = kept)
# x_test <- sample(c(0, 1), nrow(edges), replace = TRUE)
# 
# # Compute KL divergence
# KL_div <- KL_divergence_ising(Theta_P, moments, edges, x_test)
# print(KL_div)
