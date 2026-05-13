# VaR-CBO: Scalable Variational Bayesian Optimisation for Combinatorial Problems

Repository for the simulations and methods presented in:

> **Scalable Variational Bayesian Optimisation for Combinatorial Problems**
> Niyati Seth, Michael Fop — September 2023

VaR-CBO (Variational pRobabilistic Combinatorial Bayesian Optimisation) is a novel framework that integrates variational inference and probabilistic reparameterisation to enable scalable Bayesian optimisation in binary combinatorial settings. It achieves competitive optimisation performance with substantially reduced computational cost compared to MCMC-based approaches such as BOCS.

---

## Overview

Combinatorial optimisation problems arise across operations research, logistics, network design, and the sciences, but their exponentially large search spaces make exact solutions infeasible. VaR-CBO addresses this by:

- Replacing MCMC-based surrogate inference with a **sparse variational Bayes (VB)** approximation via a spike-and-slab prior, enabling fast and repeated surrogate updates within the BO loop.
- Introducing **probabilistic reparameterisation** of binary decision variables, transforming combinatorial acquisition optimisation into a smooth, continuous problem amenable to gradient-based methods (L-BFGS-B) or genetic algorithms (GA).

Together, these components make VaR-CBO particularly suited to settings where objective evaluations are expensive and traditional combinatorial solvers become computationally prohibitive.

---

## Methods Implemented

The repository implements and compares the following methods, each with variants for acquisition function optimisation:

| Method | Surrogate Inference | Acquisition Optimisation |
|---|---|---|
| BOCS-SA | Gibbs sampling (MCMC) | Simulated annealing |
| BOCS-SDP | Gibbs sampling (MCMC) | Semidefinite programming |
| BOCS-GA | Gibbs sampling (MCMC) | Genetic algorithm |
| PRCBO-BFGS | Gibbs sampling (MCMC) | Probabilistic reparameterisation + L-BFGS-B |
| PRCBO-GA | Gibbs sampling (MCMC) | Probabilistic reparameterisation + GA |
| VaR-CBO-BFGS | Variational Bayes | Probabilistic reparameterisation + L-BFGS-B |
| VaR-CBO-GA | Variational Bayes | Probabilistic reparameterisation + GA |

The main contribution — VaR-CBO — is implemented in `prbocs-vb.R`.

---

## Benchmarks

Methods are evaluated on three benchmark combinatorial optimisation problems:

- **Contamination control** (`p = 25`): minimising intervention costs in a food supply chain while keeping bacterial prevalence below a safety threshold.
- **Ising model sparsification** (`p = 24`): finding a sparse approximation to a probabilistic graphical model by minimising KL divergence.
- **Maximum satisfiability (MaxSAT)** (`p = 60`): a weighted MaxSAT benchmark from the MaxSAT Evaluation 2018 competition (`frb-frb10-6-4.wcnf`).

VaR-CBO consistently matches the optimisation performance of MCMC-based baselines while achieving significantly faster runtimes.

---

## Installation

### Required: `sparsevb` (GitHub version)

A modified version of the `sparsevb` R package is required to accommodate VaR-CBO. Install it directly from GitHub:

```r
# install.packages("remotes")
remotes::install_github("<your-github-org>/sparsevb")
```

> **Note:** The CRAN version of `sparsevb` will not work. Several changes were made to integrate the package into the VaR-CBO framework.

### Required: Stan

Stan is required to run all BOCS and PRCBO comparison methods (via `rstanarm`). Follow the installation instructions at [mc-stan.org](https://mc-stan.org/users/interfaces/rstan).

### All other dependencies

All remaining packages can be installed from CRAN:

```r
install.packages(c("GA", "optimx", "rstanarm"))
```

---

## Repository Structure

```
.
├── minimisation/        # VaR-CBO and comparison methods for minimisation problems
├── maximisation/        # Variants adapted for maximisation problems
├── prbocs-vb.R          # Main VaR-CBO implementation (primary contribution)
└── README.md
```

Minimisation and maximisation variants have been separated into distinct folders for ease of use.

---

## Usage

To run the benchmark examples directly, navigate to the relevant folder (`minimisation/` or `maximisation/`) and source the corresponding script. The benchmark problem setups follow the experimental configurations described in the paper.

To use VaR-CBO on a custom problem, the main function in `prbocs-vb.R` accepts:

- A black-box objective function `f` over binary inputs `x ∈ {0, 1}^p`
- An initial dataset of evaluated points
- A sample budget `N_max`

The function returns the best observed solution within the evaluation budget.

---

## Citation

If you use this code in your work, please cite:

```bibtex
@article{seth2023varcbo,
  title     = {Scalable Variational {B}ayesian Optimisation for Combinatorial Problems},
  author    = {Seth, Niyati and Fop, Michael},
  year      = {2023}
}
```

---

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
