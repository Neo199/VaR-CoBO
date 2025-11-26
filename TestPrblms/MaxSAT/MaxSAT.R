# ---- MaxSAT ----
# Same as the python code from
# Deshwal_Bayesian Optimization over High-Dimensional Combinatorial Spaces via Dictionary-based Embeddings
# https://github.com/aryandeshwal/BODi/blob/main/bodi/test_functions.py

read_maxsat <- function(filepath) {
  # Read all lines
  lines <- readLines(filepath)
  
  # Find "p " line
  p_line <- lines[grepl("^p ", lines)]
  if (length(p_line) == 0) stop("No problem line found in WCNF.")
  
  parts <- strsplit(trimws(p_line), " +")[[1]]
  n_vars <- as.integer(parts[3])
  n_clauses <- as.integer(parts[4])
  
  # Get clause lines (skip comments and p-line)
  clause_lines <- lines[!grepl("^(c|p)", lines)]
  
  # Parse clauses into weight and literals
  clauses <- lapply(clause_lines, function(line) {
    tokens <- strsplit(trimws(line), " +")[[1]]
    weight <- as.numeric(tokens[1])
    literals <- as.integer(tokens[-c(1, length(tokens))])  # remove weight and trailing 0
    list(weight = weight, literals = literals)
  })
  
  # Normalize weights
  weights <- sapply(clauses, `[[`, "weight")
  weight_mean <- mean(weights)
  weight_sd <- sd(weights)
  norm_weights <- (weights - weight_mean) / weight_sd
  
  # Build a data structure
  for (i in seq_along(clauses)) {
    clauses[[i]]$weight <- norm_weights[i]
  }
  
  list(
    n_vars = n_vars,
    n_clauses = n_clauses,
    clauses = clauses
  )
}

# ---- Evaluate a solution ----
evaluate_maxsat <- function(x, maxsat_obj) {
  if (length(x) != maxsat_obj$n_vars) {
    stop("x must have length n_vars.")
  }
  x <- as.logical(x)  # convert to TRUE/FALSE
  
  satisfied <- sapply(maxsat_obj$clauses, function(clause) {
    literals <- clause$literals
    # Check if any literal in the clause is satisfied
    any(sapply(literals, function(l) {
      if (l > 0) x[l] else !x[abs(l)]
    }))
  })
  
  total_score <- sum(sapply(seq_along(maxsat_obj$clauses), function(i) {
    if (satisfied[i]) maxsat_obj$clauses[[i]]$weight else 0
  }))
  
  return(total_score)
}

# # ---- Example usage ----
# # Load your maxsat file (same as frb-frb10-6-4.wcnf)
# maxsat <- read_maxsat("/Users/niyati/Projects:Codes/PhD-Compute/R code/TestPrblms/MaxSAT/maxsat.wcnf")  # replace with your file path
# 
# # Print problem info
# cat("Number of variables:", maxsat$n_vars, "\n")
# cat("Number of clauses:", maxsat$n_clauses, "\n")
# 
# # Evaluate a random binary vector
# set.seed(42)
# x <- sample(c(0, 1), maxsat$n_vars, replace = TRUE)
# score <- evaluate_maxsat(x, maxsat)
# cat("Score for random x:", score, "\n")

# ---- MaxSAT wrapper with seed ----
# wrapper to match model(x_vals, seed)
maxsat <- function(x_vals, maxsat_obj, seed) {
  set.seed(seed)  # For reproducibility if random behavior is introduced
  
  if (is.vector(x_vals)) {
    return(evaluate_maxsat(round(x_vals), maxsat_obj))
  } else if (is.matrix(x_vals)) {
    return(apply(x_vals, 1, function(row) evaluate_maxsat(round(row), maxsat_obj)))
  } else {
    stop("x_vals must be a vector or a matrix.")
  }
}