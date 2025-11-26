
# binary_nn_objective.R

# Load necessary libraries
source("~/Projects:Codes/PhD-Compute/R code/TestPrblms/NAS/sample_models.R")
library(keras)
library(tensorflow)

# Prepare MNIST data
mnist <- dataset_mnist()
x_train <- mnist$train$x / 255
y_train <- to_categorical(mnist$train$y, 10)
x_test <- mnist$test$x / 255
y_test <- to_categorical(mnist$test$y, 10)

x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# Decode a binary vector into architecture
decode_architecture <- function(x_bin, node_sizes = c(16, 32, 64)) {
  L <- length(x_bin) / (length(node_sizes) + 1)
  config <- c()
  for (i in 1:L) {
    on_bit <- x_bin[(i - 1) * 4 + 1]
    size_bits <- x_bin[(i - 1) * 4 + 2:(length(node_sizes) + 1)]
    if (on_bit == 1 && sum(size_bits) == 1) {
      size_index <- which(size_bits == 1)
      config <- c(config, node_sizes[size_index])
    }
  }
  return(config)
}

# Evaluate architecture: return both validation loss and test accuracy
evaluate_architecture <- function(layer_config, x_train, y_train, x_test, y_test) {
  if (length(layer_config) == 0) return(list(loss = Inf, acc = 0))
  
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
  
  # Train briefly
  history <- model %>% fit(
    x_train, y_train,
    epochs = 2, batch_size = 128,
    validation_split = 0.2,
    verbose = 0
  )
  
  scores <- model %>% evaluate(x_test, y_test, verbose = 0)
  val_loss <- history$metrics$val_loss[[2]]
  test_acc <- scores[[2]]
  
  return(list(loss = val_loss, acc = test_acc))
}

# Objective function: returns list(loss, acc) for each architecture in X
architecture_objective <- function(X) {
  n <- nrow(X)
  results <- data.frame(loss = numeric(n), acc = numeric(n))
  for (i in 1:n) {
    config <- decode_architecture(X[i, ])
    eval <- evaluate_architecture(config, x_train, y_train, x_test, y_test)
    results$loss[i] <- eval$loss
    results$acc[i] <- eval$acc
  }
  return(results)
}

# Example usage: evaluate random architectures
set.seed(42)
L <- 7
K <- 3
d <- L * (K + 1)
n <- 5
# X <- matrix(sample(c(0,1), n * d, replace = TRUE), nrow = n)
X <- sample_models(n, d)


# Enforce layer structure
for (i in 1:n) {
  for (l in 0:(L - 1)) {
    if (X[i, l*4 + 1] == 0) {
      X[i, (l*4 + 2):(l*4 + 4)] <- 0
    } else {
      selected <- sample(1:3, 1)
      X[i, (l*4 + 2):(l*4 + 4)] <- 0
      X[i, l*4 + 1 + selected] <- 1
    }
  }
}

# Evaluate and print results
results <- architecture_objective(X)
print(results)
