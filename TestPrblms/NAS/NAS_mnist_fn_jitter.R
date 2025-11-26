#Function for neural architecture search
#I add a jitter penalty here for invalid architectures

library(keras)
library(tensorflow)
library(digest)  # For hash function

# Set seed for reproducibility
set.seed(seed)                             # R base random
tensorflow::tf$random$set_seed(seed)     # TensorFlow internal seed

# Load and preprocess MNIST data
mnist <- dataset_mnist()
x_train <- mnist$train$x / 255
y_train <- to_categorical(mnist$train$y, 10)
x_train <- array_reshape(x_train, c(nrow(x_train), 784))

# ------------ CONFIG ------------

node_sizes <- c(16, 32, 64)  # Hidden layer size options
L <- 3                       # Max number of hidden layers
k <- length(node_sizes)      # Number of node size options
block_size <- k + 1          # Each layer: one-hot vector (k sizes + OFF)
total_bits <- L * block_size # Total bits per architecture

# ------------ JITTER PENALTY ------------

jitter_penalty <- function(base = 1e8, epsilon = 100000) {
  base + runif(1, 10000, epsilon)
}

# ------------ ENCODING HELPERS ------------

is_onehot_block <- function(block) {
  sum(block) == 1 && all(block %in% c(0, 1))
}

is_valid_architecture <- function(x_bin, node_sizes) {
  k <- length(node_sizes)
  if (length(x_bin) %% (k + 1) != 0) return(FALSE)
  L <- length(x_bin) / (k + 1)
  
  for (i in 1:L) {
    block <- x_bin[((i - 1)*(k + 1) + 1):(i*(k + 1))]
    if (!is_onehot_block(block)) return(FALSE)
  }
  return(TRUE)
}

decode_architecture <- function(x_bin, node_sizes) {
  k <- length(node_sizes)
  L <- length(x_bin) / (k + 1)
  config <- c()
  
  for (i in 1:L) {
    block <- x_bin[((i - 1)*(k + 1) + 1):(i*(k + 1))]
    idx <- which(block == 1)
    if (length(idx) != 1 || idx > (k + 1)) stop("Invalid one-hot block")
    if (idx <= k) {
      config <- c(config, node_sizes[idx])
    }
  }
  
  return(config)
}

# Generate a valid random architecture
generate_valid_architecture_bin <- function(L = 3, node_sizes = c(16, 32, 64)) {
  k <- length(node_sizes)
  arch_bin <- c()
  for (i in 1:L) {
    option <- sample(1:(k + 1), 1)  # Choose one: size or OFF
    block <- rep(0, k + 1)
    block[option] <- 1
    arch_bin <- c(arch_bin, block)
  }
  return(arch_bin)
}

# Generate multiple valid architectures
generate_arch_matrix <- function(n, L = 3, node_sizes = c(16, 32, 64)) {
  do.call(rbind, replicate(n, generate_valid_architecture_bin(L, node_sizes), simplify = FALSE))
}

# ------------ EVALUATION FUNCTION ------------

evaluate_architecture <- function(x_bin, node_sizes, mode = c("combined", "loss", "accuracy"), 
                                  alpha = 1.0, beta = 1.0) {
  mode <- match.arg(mode)
  
  valid <- is_valid_architecture(x_bin, node_sizes)
  if (!valid) {
    penalty <- jitter_penalty()
    return(list(score = penalty, loss = 1e8, acc = 0, valid = FALSE))
  }
  
  layer_config <- decode_architecture(x_bin, node_sizes)
  
  if (length(layer_config) == 0) {
    penalty <- jitter_penalty()
    return(list(score = penalty, loss = 1e8, acc = 0, valid = FALSE))
  }
  
  model <- keras_model_sequential()
  model %>% layer_dense(units = layer_config[1], activation = "relu", input_shape = 784)
  
  if (length(layer_config) > 1) {
    for (units in layer_config[-1]) {
      model %>% layer_dense(units = units, activation = "relu")
    }
  }
  
  model %>%
    layer_dense(units = 10, activation = "softmax") %>%
    compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = "accuracy")
  
  history <- model %>% fit(
    x_train, y_train,
    epochs = 3, batch_size = 128,
    verbose = 0
  )
  
  train_loss <- history$metrics$loss[[3]]
  train_acc <- history$metrics$accuracy[[3]]
  
  score <- switch(mode,
                  "loss" = train_loss,
                  "accuracy" = -train_acc,
                  "combined" = alpha * train_loss - beta * train_acc
  )
  
  return(list(score = score, loss = train_loss, acc = train_acc, valid = TRUE))
}

# ------------ BATCH EVALUATION FUNCTION ------------

evaluate_binary_architectures <- function(x_bin, node_sizes = c(16, 32, 64), 
                                          mode = "combined", alpha = 1.0, beta = 1.0) {
  x_bin <- as.matrix(x_bin)
  n <- nrow(x_bin)
  results <- data.frame(score = numeric(n), loss = numeric(n), acc = numeric(n), valid = logical(n))
  
  for (i in 1:n) {
    res <- evaluate_architecture(x_bin[i, ], node_sizes, mode, alpha, beta)
    results[i, ] <- c(res$score, res$loss, res$acc, res$valid)
  }
  
  return(results)
}

# # Create some valid architectures
# x_bin_mat <- generate_arch_matrix(5, L = 3, node_sizes = c(16, 32, 64))
# 
# # Add one intentionally invalid row
# bad_row <- sample(0:1, ncol(x_bin_mat), replace = TRUE)
# x_bin_mat <- rbind(x_bin_mat, bad_row)
# 
# # Evaluate
# results <- evaluate_binary_architectures(x_bin_mat, mode = "combined", alpha = 1, beta = 1)
# print(results)
