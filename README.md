# Experimental Investigation of Bias Delocalization in Unadjusted Langevin Sampling

[![Julia 1.6+](https://img.shields.io/badge/Julia-1.6%2B-blue.svg)](https://julialang.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code for an experimental investigation into the "delocalization of bias" hypothesis for high-dimensional models sampled with Unadjusted Langevin Algorithm (ULA).

### The Hypothesis

> The core hypothesis posits that for models with sparse interactions (i.e., a sparse Hessian of the drift term) and a Gaussian stationary distribution, the sampling bias on any single marginal is nearly independent of the system's total dimension.

## Prerequisites

### 1. Prerequisites

- **Julia:** Ensure you have Julia v1.6 or a later version installed.
- **Julia Packages:** Install the required packages by running the following command in a Julia REPL:

  ```julia
  using Pkg
  Pkg.add([
      "Distributions", "Plots", "ProgressMeter", 
      "StatsBase", "JLD2", "LinearAlgebra", 
      "Random", "Printf"
  ])
  ```

### 2. Running the Experiments

You can run either of the two main experiment scripts from your terminal. Results will be automatically saved to the `results/` directory.

- **To validate the core theory:**
  ```bash
  julia LangevinBiasExperiments_a.jl
  ```

- **To test relaxations of the hypothesis conditions:**
  ```bash
  julia LangevinBiasExperiments_x.jl
  ```

---

## Project Structure

The repository is organized as follows:

```
.
├── LangevinBiasExperiments_a.jl  # Main script for theory validation experiments.
├── LangevinBiasExperiments_x.jl  # Script for exploring relaxed conditions.
├── README.md                     # You are here!
└── results/
    ├── all_experiments_plot.png  # Output: Plot of marginal bias vs. dimension.
    ├── acceptance_rates_plot.png # Output: Plot of MALA acceptance rates vs. dimension.
    └── raw_data.jld2             # Output: Binary file with all raw experiment data.
```

---

## Results

Upon successful execution, the `results/` directory will contain:

- **`all_experiments_plot.png`**: Visualizes the core hypothesis by plotting the marginal sampling bias against the model dimension.
- **`acceptance_rates_plot.png`**: Shows the Metropolis-Adjusted Langevin Algorithm (MALA) acceptance rates, useful for diagnostics.
- **`raw_data.jld2`**: A JLD2 file containing the complete, unprocessed data from all experimental runs for further analysis.