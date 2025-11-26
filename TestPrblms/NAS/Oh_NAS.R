# =========================
# NAS (Binary) — R torch
# =========================

library(torch)
library(torchvision)
library(magrittr)

# ---------- Transforms ----------
transform_factory <- function(mean, std) {
  function(img) {
    img %>%
      transform_to_tensor() %>%
      transform_normalize(mean = mean, std = std)
  }
}

# ---------- Dataset loaders ----------
load_dataset <- function(data_type, batch_size = 64) {
  if (data_type == "MNIST") {
    transform <- transform_factory(mean = 0.5, std = 0.5)
    train_ds <- mnist_dataset(root = "./data", train = TRUE,  download = TRUE, transform = transform)
    test_ds  <- mnist_dataset(root = "./data", train = FALSE, download = TRUE, transform = transform)
  } else if (data_type == "FashionMNIST") {
    transform <- transform_factory(mean = 0.5, std = 0.5)
    train_ds <- fashion_mnist_dataset(root = "./data", train = TRUE,  download = TRUE, transform = transform)
    test_ds  <- fashion_mnist_dataset(root = "./data", train = FALSE, download = TRUE, transform = transform)
  } else if (data_type == "CIFAR10") {
    transform <- transform_factory(mean = c(0.4914, 0.4822, 0.4465),
                                   std  = c(0.2023, 0.1994, 0.2010))
    train_ds <- cifar10_dataset(root = "./data", train = TRUE,  download = TRUE, transform = transform)
    test_ds  <- cifar10_dataset(root = "./data", train = FALSE, download = TRUE, transform = transform)
  } else {
    stop("Unsupported dataset: ", data_type)
  }
  
  # fixed validation split
  n_valid <- min(10000L, length(train_ds) %/% 5L)
  indices <- torch_randperm(length(train_ds))$to(dtype = torch_int64())
  train_idx <- as.array(indices[1:(length(train_ds) - n_valid)])
  valid_idx <- as.array(indices[(length(train_ds) - n_valid + 1):length(train_ds)])
  
  train_dl <- dataloader(dataset_subset(train_ds, indices = train_idx),
                         batch_size = batch_size, shuffle = TRUE)
  valid_dl <- dataloader(dataset_subset(train_ds, indices = valid_idx),
                         batch_size = batch_size, shuffle = FALSE)
  test_dl  <- dataloader(test_ds, batch_size = batch_size, shuffle = FALSE)
  
  list(train = train_dl, valid = valid_dl, test = test_dl)
}

# ---------- Dataset info ----------
get_dataset_info <- function(data_type) {
  if (data_type %in% c("MNIST", "FashionMNIST")) {
    list(n_ch_in = 1L, h_in = 28L, w_in = 28L, n_classes = 10L, n_ch_base = 8L)
  } else if (data_type == "CIFAR10") {
    list(n_ch_in = 3L, h_in = 32L, w_in = 32L, n_classes = 10L, n_ch_base = 16L)
  } else {
    stop("Unsupported dataset: ", data_type)
  }
}
# ---------- Train loop ----------
# NASBinaryCNN to maker adam friendly
NASBinaryCNN <- nn_module(
  "NASBinaryCNN",
  initialize = function(model_list) {
    # assign all modules from your list
    self$data_type <- model_list$data_type
    self$conv0 <- model_list$conv0
    self$bn0 <- model_list$bn0
    self$relu0 <- model_list$relu0
    self$cell1 <- model_list$cell1
    self$maxpool1 <- model_list$maxpool1
    self$conv1 <- model_list$conv1
    self$cell2 <- model_list$cell2
    self$maxpool2 <- model_list$maxpool2
    self$fc <- model_list$fc
    
    if (!is.null(model_list$conv2)) self$conv2 <- model_list$conv2
    if (!is.null(model_list$cell3)) self$cell3 <- model_list$cell3
    if (!is.null(model_list$maxpool3)) self$maxpool3 <- model_list$maxpool3
  },
  
  forward = function(x) {
    x <- self$relu0(self$bn0(self$conv0(x)))
    x <- self$conv1(self$maxpool1(forward_nas_cell(self$cell1, x)))
    
    if (self$data_type %in% c("MNIST", "FashionMNIST")) {
      x <- self$maxpool2(forward_nas_cell(self$cell2, x))
    } else {
      x <- self$conv2(self$maxpool2(forward_nas_cell(self$cell2, x)))
      x <- self$maxpool3(forward_nas_cell(self$cell3, x))
    }
    
    self$fc(x$view(c(x$size(1), -1)))
  }
)


train_model <- function(model, n_epochs, train_loader, valid_loader, device = 0, verbose = TRUE) {
  criterion <- nn_cross_entropy_loss()
  model <- NASBinaryCNN(model)
  optimizer <- optim_adam(model$parameters, lr = 0.001)
  
  best_acc <- 0
  for (epoch in 1:n_epochs) {
    # browser()
    if (verbose) cat(sprintf("[epoch %02d] train ...\n", epoch))
    model$train()
    train_iter <- dataloader_make_iter(train_loader)
    repeat {
      batch <- tryCatch(dataloader_next(train_iter), error = function(e) NULL)
      if (is.null(batch)) break
      inputs <- batch[[1]]
      labels <- batch[[2]]
      
      optimizer$zero_grad()
      outputs <- model(inputs)
      loss <- criterion(outputs, labels)
      loss$backward()
      optimizer$step()
    }
    
    # eval
    if (verbose) cat(sprintf("[epoch %02d] eval ...\n", epoch))
    model$eval()
    correct <- 0
    total <- 0
    valid_iter <- dataloader_make_iter(valid_loader)
    repeat {
      batch <- tryCatch(dataloader_next(valid_iter), error = function(e) NULL)
      if (is.null(batch)) break
      inputs <- batch[[1]]
      labels <- batch[[2]]
      outputs <- model(inputs)
      preds <- torch_argmax(outputs, dim = 2)
      correct <- correct + (preds == labels)$sum()$item()
      total <- total + labels$size(1)
    }
    acc <- correct / total
    best_acc <- max(best_acc, acc)
    if (verbose) cat(sprintf("[epoch %02d] acc: %.4f (best=%.4f)\n", epoch, acc, best_acc))
  }
  best_acc
}

# ---------- FLOP proxy (parameter count) ----------
count_ops <- function(model) {
  model <- NASBinaryCNN(model)
  sum(sapply(model$parameters, function(p) p$numel()))
}

# ---------- Parse connectivity (7-node DAG) ----------
# x_conn is length 21 (upper triangle, i<j)
parse_connectivity <- function(x_conn) {
  stopifnot(length(x_conn) == 21)
  adj <- matrix(0L, nrow = 7L, ncol = 7L)
  idx <- 1L
  for (i in 1:6) {
    for (j in (i+1):7) {
      adj[i, j] <- as.integer(x_conn[idx])
      idx <- idx + 1L
    }
  }
  adj
}

# ---------- Graph utilities ----------
# Ensure there is a path 1->7; zero out isolated vertices; return NULL if unreachable
valid_net_topo <- function(adj_mat) {
  n <- nrow(adj_mat)
  stopifnot(n == ncol(adj_mat))
  # must be upper triangular (DAG by construction)
  if (sum((adj_mat * lower.tri(adj_mat)) != 0) > 0) stop("Adjacency must be upper triangular")
  
  # reachability from 1
  reachable <- rep(FALSE, n); reachable[1] <- TRUE
  for (s in 1:(n-1)) {
    new_r <- which((adj_mat[reachable, , drop = FALSE] > 0), arr.ind = TRUE)[,2]
    if (length(new_r)) reachable[new_r] <- TRUE
  }
  # reachability to n (reverse graph)
  rev_adj <- t(adj_mat)
  can_reach_end <- rep(FALSE, n); can_reach_end[n] <- TRUE
  for (s in 1:(n-1)) {
    new_r <- which((rev_adj[can_reach_end, , drop = FALSE] > 0), arr.ind = TRUE)[,2]
    if (length(new_r)) can_reach_end[new_r] <- TRUE
  }
  
  if (!(reachable[n] && can_reach_end[1])) return(NULL)
  
  keep <- which(reachable & can_reach_end)
  drop <- setdiff(seq_len(n), keep)
  if (length(drop)) {
    adj_mat[drop, ] <- 0L
    adj_mat[, drop] <- 0L
  }
  adj_mat
}

# Layered topological order (Kahn)
# Returns list of integer vectors; first layer should contain node 1
topo_sort <- function(adj_mat) {
  n <- nrow(adj_mat)
  indeg <- colSums(adj_mat)
  layers <- list()
  seen <- rep(FALSE, n)
  S <- which(indeg == 0)
  while (length(S) > 0) {
    layers <- append(layers, list(S))
    seen[S] <- TRUE
    for (v in S) {
      nbrs <- which(adj_mat[v, ] == 1L)
      if (length(nbrs)) indeg[nbrs] <- indeg[nbrs] - 1L
    }
    S <- which(indeg == 0 & !seen)
  }
  if (!all(seen | colSums(adj_mat) == 0 | rowSums(adj_mat) == 0)) {
    stop("Graph may contain a cycle or disconnected structure not pruned.")
  }
  layers
}

# ---------- NASBinaryCell (functional version) ----------
make_nas_cell <- function(node_type, adj_mat, n_channels) {
  n_nodes <- nrow(adj_mat)
  stopifnot(length(node_type) == (n_nodes - 2L) * 2L)
  topo_order <- topo_sort(adj_mat)
  
  # build nodes (2..n-1)
  nodes <- vector("list", n_nodes)
  for (i in 2:(n_nodes - 1L)) {
    t1 <- node_type[2*i - 3L]
    t2 <- node_type[2*i - 2L]
    if (t1 == 0 && t2 == 0) {
      nodes[[i]] <- nn_identity()
    } else if (t1 == 0 && t2 == 1) {
      nodes[[i]] <- nn_max_pool2d(kernel_size = 3, stride = 1, padding = 1)
    } else if (t1 == 1 && t2 == 0) {
      nodes[[i]] <- nn_conv2d(in_channels = n_channels,
                              out_channels = n_channels,
                              kernel_size = 3, padding = 1)
    } else {
      nodes[[i]] <- nn_conv2d(in_channels = n_channels,
                              out_channels = n_channels,
                              kernel_size = 5, padding = 2)
    }
  }
  nodes[[n_nodes]] <- nn_identity()
  
  list(adj_mat = adj_mat,
       topo_order = topo_order,
       nodes = nodes,
       n_nodes = n_nodes)
}

forward_nas_cell <- function(cell, x) {
  node_out <- vector("list", cell$n_nodes)
  node_out[[1]] <- x
  
  # skip the first layer (which includes 1) when iterating
  for (idx_set in cell$topo_order[-1]) {
    for (j in idx_set) {
      inputs <- lapply(which(cell$adj_mat[, j] == 1L),
                       function(k) node_out[[k]])
      node_in <- if (length(inputs) == 0) {
        torch_zeros_like(node_out[[1]])
      } else {
        Reduce(`+`, inputs)
      }
      node_out[[j]] <- cell$nodes[[j]](node_in)
    }
  }
  node_out[[cell$n_nodes]]
}

# ---------- NASBinaryCNN (functional version) ----------

make_nas_cnn <- function(data_type, node_type, adj_mat, n_ch_in, h_in, w_in, n_ch_base) {
  stopifnot(data_type %in% c("MNIST", "FashionMNIST", "CIFAR10"))
  conv0 <- nn_conv2d(n_ch_in, n_ch_base, kernel_size = 3, padding = 1, bias = TRUE)
  bn0   <- nn_batch_norm2d(num_features = n_ch_base)
  relu0 <- nn_relu()
  cell1 <- make_nas_cell(node_type, adj_mat, n_ch_base)
  maxpool1 <- nn_max_pool2d(kernel_size = 2)
  conv1 <- nn_conv2d(n_ch_base, n_ch_base*2, kernel_size = 1, bias = TRUE)
  cell2 <- make_nas_cell(node_type, adj_mat, n_ch_base*2)
  maxpool2 <- nn_max_pool2d(kernel_size = 2)
  
  if (data_type %in% c("MNIST", "FashionMNIST")) {
    fc <- nn_linear(in_features = as.integer(n_ch_base*2*(h_in/4)*(w_in/4)),
                    out_features = 10)
    extra <- list()
  } else {
    conv2 <- nn_conv2d(n_ch_base*2, n_ch_base*4, kernel_size = 1, bias = TRUE)
    cell3 <- make_nas_cell(node_type, adj_mat, n_ch_base*4)
    maxpool3 <- nn_max_pool2d(kernel_size = 2)
    fc <- nn_linear(in_features = as.integer(n_ch_base*4*(h_in/8)*(w_in/8)),
                    out_features = 10)
    extra <- list(conv2 = conv2, cell3 = cell3, maxpool3 = maxpool3)
  }
  
  c(list(data_type = data_type,
         conv0 = conv0, bn0 = bn0, relu0 = relu0,
         cell1 = cell1, maxpool1 = maxpool1,
         conv1 = conv1, cell2 = cell2, maxpool2 = maxpool2,
         fc = fc),
    extra)
}

forward_nas_cnn <- function(model, x) {
  x <- model$relu0(model$bn0(model$conv0(x)))
  x <- model$conv1(model$maxpool1(forward_nas_cell(model$cell1, x)))
  
  if (model$data_type %in% c("MNIST", "FashionMNIST")) {
    x <- model$maxpool2(forward_nas_cell(model$cell2, x))
  } else {
    x <- model$conv2(model$maxpool2(forward_nas_cell(model$cell2, x)))
    x <- model$maxpool3(forward_nas_cell(model$cell3, x))
  }
  
  model$fc(x$view(c(x$size(1), -1)))
}

# ---------- Build model from binary vector x ----------
# x = c(conn[21], node_type[10]) for a 7-node cell
build_nas_model <- function(x, data_type) {
  info <- get_dataset_info(data_type)
  stopifnot(length(x) == 31)
  x <- as.integer(x)
  adj <- parse_connectivity(x[1:21])

  # ---- Early constraint check ----
  n <- nrow(adj)
  # must have at least one outgoing edge from node 1
  if (sum(adj[1, ]) == 0) return(NULL)
  # must have at least one incoming edge to node 7
  if (sum(adj[, n]) == 0) return(NULL)
  
  # now do full pruning/reachability check
  adj <- valid_net_topo(adj)
  if (is.null(adj)) return(NULL)
  
  node_type <- x[22:31]
  make_nas_cnn(data_type, node_type, adj,
               info$n_ch_in, info$h_in, info$w_in, info$n_ch_base)
}

# ---------- NAS evaluation (objective) ----------
nas_binary_evaluate <- function(
    x, data_type, loaders,
    n_epochs = 3, device = 0, flop_weight = 0.02, seed = NULL, verbose = TRUE
) {
  if (!is.null(seed)) {
    set.seed(seed)
    torch_manual_seed(seed)
  }
  
  if (verbose) cat("-> build dynamic NAS cell\n")
  model <- build_nas_model(x, data_type)
  if (is.null(model)) {
    if (verbose) cat("-> invalid topology (no path 1->7); returning heavy penalty\n")
    return(10)  # penalty
  }
  
  if (verbose) cat("-> train & validate\n")
  eval_acc <- train_model(model, n_epochs, loaders$train, loaders$valid, device, verbose)
  
  params <- count_ops(model)
  max_params <- 1e6
  param_ratio <- min(1, params / max_params)
  
  score <- (1 - eval_acc) + flop_weight * param_ratio
  if (verbose) cat(sprintf("-> acc=%.4f  params=%d  score=%.6f\n", eval_acc, params, score))
  return(score)
}

# ---------- Optimizer-facing factory ----------
# Returns a list with: loaders, objective(x), and a sampler for random x
make_nas_objective <- function(
    data_type = c("MNIST", "FashionMNIST", "CIFAR10"),
    batch_size = 32,
    n_epochs = 20,
    device = 0,
    flop_weight = 0.02,
    seed,
    verbose_each = TRUE
) {
  data_type <- match.arg(data_type)
  loaders <- load_dataset(data_type, batch_size)
  
  objective <- function(x) {
    nas_binary_evaluate(
      x = x, data_type = data_type, loaders = loaders,
      n_epochs = n_epochs, device = device, flop_weight = flop_weight,
      seed = seed, verbose = verbose_each
    )
  }
  
  list(loaders = loaders, objective = objective)
}

# ---------- Minimal example ----------

# sample_x <- function() as.integer(sample(0:1, 31, replace = TRUE))
# 
# obj <- make_nas_objective(data_type = "MNIST", verbose_each = TRUE, seed = seed)
# x <- sample_x()
# score <- obj$objective(x)
# cat("score:", score, "\n")
# 
# 
# x_vals <- sample_models(10,31) 
# score <- apply(x_vals, 1, function(x_row) {
#   obj$objective(as.integer(c(x_row)))
# })


