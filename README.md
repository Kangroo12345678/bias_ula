# Experimental Investigation of Bias Delocalization in Unadjusted Langevin Sampling

The experiments are designed to test the "delocalization of bias" hypothesis, which posits that for models with sparse interactions(Hessian of drift in the Langevin equation), Gaussian stationary distribution, the bias on any single marginal(and finite coupling of marginals) is nearly/completely independent of the total dimension.

## Structure

- `LangevinBiasExperiments_a.jl`: Script running 4 scenarios for which, we have developed theory for. Validation Experiment.
- `LangevinBiasExperiments_x.jl`: Experimenting possible relaxation of delocalization conditions.
- `README.md`: You are here!
- `results/`: A directory where the output plots and raw experiment data will be saved.
  - `all_experiments_plot.png`: Plot of marginal bias vs. dimension.
  - `acceptance_rates_plot.png`: Plot of MALA acceptance rates vs. dimension.
  - `raw_data.jld2`: A binary file containing all raw data from the experiments.

## Prerequisites: before you run this project

- **Julia:** Version 1.6 or later.
- **Julia Packages:** The following packages are required:
    ```julia
    import Pkg
    Pkg.add(["Distributions", "Plots", "ProgressMeter", "StatsBase", "JLD2", "LinearAlgebra", "Random", "Printf"])
    ```

## Running the project

Just run the script you wish to run, after a while, you could see the results in the result folder.

```bash
julia LangevinBiasExperiments_a.jl