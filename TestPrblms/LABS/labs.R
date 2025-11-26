labs_function <- function(x, flip = FALSE, seed = 43) {
  # Ensure input is a matrix
  if (is.vector(x)) {
    x <- matrix(x, nrow = 1)
  }
  
  dim <- ncol(x)
  fxs <- numeric(nrow(x))
  
  # Generate random flip vector if required
  random_flip <- NULL
  if (flip) {
    set.seed(seed)
    random_flip <- sample(c(-1, 1), size = dim, replace = TRUE)
  }
  
  for (i in seq_len(nrow(x))) {
    ux <- x[i, ]
    ux[ux == 0] <- -1  # convert 0s to -1s
    
    if (!is.null(random_flip)) {
      ux <- ux * random_flip
    }
    
    e <- 0
    for (k in 1:(dim - 1)) {
      e <- e + (sum(ux[1:(dim - k)] * ux[(k + 1):dim]))^2
    }
    
    fx <- (dim^2 / (2 * e))
    fxs[i] <- fx
  }
  
  return(fxs)
}

# # Example input: a binary vector
# x <- c(1, 0, 1, 1, 0, 0, 1, 0)
# 
# # Evaluate LABS function
# labs_function(x)
# 
# # For a matrix of binary inputs (each row is a different input)
# X <- matrix(sample(c(0,1), 30, replace = TRUE), nrow = 3)
# labs_function(X, flip = TRUE)
