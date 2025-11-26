library(keras)

# Load MNIST data globally once
mnist <- dataset_mnist()
x_train <- mnist$train$x / 255
y_train <- to_categorical(mnist$train$y, 10)
x_test <- mnist$test$x / 255
y_test <- to_categorical(mnist$test$y, 10)
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# Define the evaluation function
evaluate_architecture <- function(x, seed, layer_sizes = c(10, 20, 30)) {
  set.seed(seed)
  num_layers <- length(x) / length(layer_sizes)
  if (num_layers != floor(num_layers)) stop("Invalid input length for x")
  
  # Decode binary architecture vector
  model <- keras_model_sequential()
  input_added <- FALSE
  
  for (i in seq_len(num_layers)) {
    start <- (i - 1) * length(layer_sizes) + 1
    end <- i * length(layer_sizes)
    group <- x[start:end]
    
    if (sum(group) > 1) {
      # Infeasible: multiple sizes selected → penalize
      return(list(loss = 1e6, accuracy = 0))
    }
    if (sum(group) == 0) {
      # Skip layer if no size is selected
      next
    }
    
    size <- layer_sizes[which(group == 1)]
    if (!input_added) {
      model %>% layer_dense(units = size, activation = "relu", input_shape = 784)
      input_added <- TRUE
    } else {
      model %>% layer_dense(units = size, activation = "relu")
    }
  }
  
  # Add output layer
  model %>% layer_dense(units = 10, activation = "softmax")
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )
  
  # Train with small epochs for speed
  history <- model %>% fit(
    x_train, y_train,
    epochs = 2, batch_size = 128,
    verbose = 0,
    validation_data = list(x_test, y_test)
  )
  
  val_loss <- history$metrics$val_loss[[2]]
  val_acc <- history$metrics$val_accuracy[[2]]
  
  return(list(loss = val_loss, accuracy = val_acc))
}

# # A valid binary vector for 3 layers and 3 size options
# x <- c(0, 1, 0, 1, 0, 0, 0, 1, 0)  # means: layer1=20, layer2=10, layer3=20
# print(evaluate_architecture(x))
