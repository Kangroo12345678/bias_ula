# Author: Michael Kang

using LinearAlgebra, Statistics, Random, Plots, Distributions

# Set a consistent theme and create directory for results
theme(:dark)
const RESULTS_DIR = "results/concentration"
mkpath(RESULTS_DIR)
println("Plots will be saved to: $(pwd())/$(RESULTS_DIR)")

# ==============================================================================
#  Core Simulation & Analysis Utilities
# ==============================================================================

"""
    run_ula(grad_V, x0, h, n_steps, n_burnin) -> Matrix

Simulates the Unadjusted Langevin Algorithm (ULA).
Returns a (d x n_samples) matrix of samples.
"""
function run_ula(grad_V::Function, x0::Vector{Float64}, h::Float64, n_steps::Int, n_burnin::Int)
    d = length(x0)
    x = copy(x0)
    n_samples = n_steps - n_burnin
    samples = zeros(d, n_samples)
    # The noise term is sqrt(2h) * Z, where Z ~ N(0, I).
    # This is equivalent to drawing from a Normal distribution with std dev sqrt(2h).
    noise_dist = Normal(0, sqrt(2 * h))

    for k in 1:n_steps
        # ULA update step
        x .-= h * grad_V(x) .+ rand(noise_dist, d)
        
        # Store sample after burn-in period
        if k > n_burnin
            samples[:, k - n_burnin] .= x
        end
    end
    return samples
end

"""
    compute_w1(samples1, samples2) -> Float64

Computes the 1-Wasserstein distance between two 1D empirical distributions.
"""
function compute_w1(samples1::AbstractVector, samples2::AbstractVector)
    # This is a standard estimator for 1D W_1 distance from samples.
    return mean(abs.(sort(samples1) .- sort(samples2)))
end

"""
    get_true_marginal_samples(J, num_samples) -> Matrix

Generates samples from the true marginals of a Gaussian distribution
defined by the precision matrix J.
"""
function get_true_marginal_samples(J::AbstractMatrix, num_samples::Int)
    # The target distribution is N(0, J⁻¹)
    if !isposdef(J)
        @warn "Precision matrix J is not positive definite. Using pseudo-inverse."
        cov_matrix = pinv(J)
    else
        cov_matrix = inv(J)
    end
    
    d = size(J, 1)
    # Marginal variances are the diagonal elements of the covariance matrix
    marginal_vars = diag(cov_matrix)
    
    # Handle potential numerical issues where variance might be slightly negative
    marginal_vars[marginal_vars .< 0] .= 0
    
    true_samples = zeros(d, num_samples)
    for i in 1:d
        true_samples[i, :] = randn(num_samples) .* sqrt(marginal_vars[i])
    end
    return true_samples
end

# ==============================================================================
#  Experiment 1: Bias Amplification via Non-Normal Influence
# ==============================================================================
println("\nRunning Experiment 1: Non-Normal Influence Chain...")

function run_experiment_1(dims_to_test)
    max_biases = []
    
    for d in dims_to_test
        println("  Testing d = $d...")
        # --- Model Setup ---
        β = 0.8
        α_vec = [1.15^(d - i) for i in 1:d]
        grad_V(x) = begin
            g = zeros(d);
            g[1] = α_vec[1] * x[1] - β * x[2];
            g[d] = α_vec[d] * x[d] - β * x[d-1];
            for i in 2:d-1; g[i] = α_vec[i] * x[i] - β * (x[i-1] + x[i+1]); end
            return g
        end
        # NOTE: The "not positive definite" warning is expected for this potential.
        # The decaying diagonal terms α_i eventually violate diagonal dominance,
        # making the Hessian singular. The code correctly handles this by using
        # the pseudo-inverse to find the covariance of the associated degenerate Gaussian.
        J = Tridiagonal(-β * ones(d-1), α_vec, -β * ones(d-1))

        # --- Simulation ---
        h = 0.005; n_steps = 2_000_000; n_burnin = 200_000
        ula_samples = run_ula(grad_V, zeros(d), h, n_steps, n_burnin)
        true_samples = get_true_marginal_samples(J, size(ula_samples, 2))
        
        # --- Bias Calculation ---
        biases = [compute_w1(ula_samples[i, :], true_samples[i, :]) for i in 1:d]
        push!(max_biases, maximum(biases))

        # --- Plotting for individual dimension ---
        p = plot(1:d, biases, title="Exp 1: Bias vs. Index (d=$d)", xlabel="Coordinate Index", ylabel="W₁ Bias", legend=false, marker=:o)
        savefig(p, joinpath(RESULTS_DIR, "exp1_d=$(d).png"))
    end
    
    # --- Plotting Scaling Results ---
    p_scaling = plot(dims_to_test, max_biases, title="Exp 1: Bias Scaling", xlabel="Dimension d", ylabel="Max. Marginal W₁ Bias", legend=false, marker=:o, lw=2)
    savefig(p_scaling, joinpath(RESULTS_DIR, "exp1_scaling.png"))
end

# run_experiment_1([20, 40, 60, 80])
# println("Experiment 1 complete.")

# ==============================================================================
#  Experiment 2: Bias Concentration in a Strongly Coupled Subspace
# ==============================================================================
println("\nRunning Experiment 2: Subspace Concentration...")

function run_experiment_2(dims_to_test)
    mean_core_biases = []
    mean_halo_biases = []
    
    for d in dims_to_test
        println("  Testing d = $d...")
        # --- Model Setup ---
        core_size = 8
        # Ensure d is large enough for a halo
        if d <= core_size; continue; end
        core_idx = 1:core_size
        halo_idx = (core_size+1):d
        p = (α_strong=1.0, β_strong=0.45, α_weak=4.0, β_weak=0.2, β_interact=0.05)
        
        J = zeros(d,d)
        # Core block
        J[core_idx, core_idx] = Tridiagonal(-p.β_strong*ones(core_size-1), p.α_strong*ones(core_size), -p.β_strong*ones(core_size-1))
        # Halo block
        J[halo_idx, halo_idx] = Tridiagonal(-p.β_weak*ones(d-core_size-1), p.α_weak*ones(d-core_size), -p.β_weak*ones(d-core_size-1))
        # Interaction
        J[core_size, core_size+1] = J[core_size+1, core_size] = -p.β_interact
        grad_V(x) = J * x
        
        # --- Simulation ---
        h = 0.01; n_steps = 2_000_000; n_burnin = 200_000
        ula_samples = run_ula(grad_V, zeros(d), h, n_steps, n_burnin)
        true_samples = get_true_marginal_samples(J, size(ula_samples, 2))
        
        # --- Bias Calculation ---
        biases = [compute_w1(ula_samples[i, :], true_samples[i, :]) for i in 1:d]
        push!(mean_core_biases, mean(biases[core_idx]))
        push!(mean_halo_biases, isempty(halo_idx) ? 0.0 : mean(biases[halo_idx]))
        
        # --- Plotting for individual dimension ---
        p = plot(1:d, biases, title="Exp 2: Bias vs. Index (d=$d)", xlabel="Coordinate Index", ylabel="W₁ Bias", legend=false, marker=:o)
        vspan!(p, core_idx, color=:red, alpha=0.2, label="Core")
        savefig(p, joinpath(RESULTS_DIR, "exp2_d=$(d).png"))
    end
    
    # --- Plotting Scaling Results ---
    valid_dims = filter(x -> x > 8, dims_to_test)
    p_scaling = plot(valid_dims, mean_core_biases, label="Avg. Core Bias", marker=:o, lw=2)
    plot!(p_scaling, valid_dims, mean_halo_biases, label="Avg. Halo Bias", marker=:s, lw=2)
    title!("Exp 2: Bias Scaling"); xlabel!("Dimension d"); ylabel!("Average W₁ Bias")
    savefig(p_scaling, joinpath(RESULTS_DIR, "exp2_scaling.png"))
end

# run_experiment_2([20, 40, 60, 80])
# println("Experiment 2 complete.")


# ==============================================================================
#  Experiment 3: Global Bias Inflation Near Phase Transition
# ==============================================================================
println("\nRunning Experiment 3: Near-Critical Bias Inflation...")

function run_experiment_3(dims_to_test)
    total_biases_stable = []
    total_biases_critical = []
    
    for d in dims_to_test
        println("  Testing d = $d...")
        # --- Model Setup ---
        α = 1.0
        β_stable = 0.25      # rho(R) = 0.5
        β_critical = 0.48    # rho(R) = 0.96
        
        # --- FIX: Construct J as a dense matrix to allow setting corner elements ---
        function make_ring_J(d, α, β)
            J_dense = diagm(0 => α * ones(d), 1 => -β * ones(d-1), -1 => -β * ones(d-1))
            J_dense[1, d] = -β # Periodic boundary condition
            J_dense[d, 1] = -β # Periodic boundary condition
            return J_dense
        end
        
        J_stable = make_ring_J(d, α, β_stable)
        J_critical = make_ring_J(d, α, β_critical)
        grad_V_stable(x) = J_stable * x
        grad_V_critical(x) = J_critical * x
        
        # --- Simulation ---
        h = 0.001; n_steps = 3000000; n_burnin = 300000
        ula_stable = run_ula(grad_V_stable, zeros(d), h, n_steps, n_burnin)
        true_stable = get_true_marginal_samples(J_stable, size(ula_stable, 2))
        ula_critical = run_ula(grad_V_critical, zeros(d), h, n_steps, n_burnin)
        true_critical = get_true_marginal_samples(J_critical, size(ula_critical, 2))
        
        # --- Bias Calculation ---
        biases_stable = [compute_w1(ula_stable[i,:], true_stable[i,:]) for i in 1:d]
        biases_critical = [compute_w1(ula_critical[i,:], true_critical[i,:]) for i in 1:d]
        push!(total_biases_stable, sum(biases_stable))
        push!(total_biases_critical, sum(biases_critical))

        # --- Plotting for individual dimension ---
        p = plot(1:d, biases_stable, label="Stable (β=$β_stable)", c=2, marker=:o)
        plot!(p, 1:d, biases_critical, label="Near-Critical (β=$β_critical)", c=3, marker=:s)
        title!("Exp 3: Bias vs. Index (d=$d)"); xlabel!("Coordinate Index"); ylabel!("W₁ Bias")
        savefig(p, joinpath(RESULTS_DIR, "exp3_d=$(d).png"))
    end
    
    # --- Plotting Scaling Results ---
    p_scaling = plot(dims_to_test, total_biases_stable, label="Stable (β=0.25)", marker=:o, lw=2, yaxis=:log)
    plot!(p_scaling, dims_to_test, total_biases_critical, label="Near-Critical (β=0.48)", marker=:s, lw=2)
    title!("Exp 3: Total Bias Scaling (Log Scale)"); xlabel!("Dimension d"); ylabel!("Total W₁ Bias (Σᵢ W₁)")
    savefig(p_scaling, joinpath(RESULTS_DIR, "exp3_scaling_log.png"))
end

run_experiment_3([16, 32, 64, 128, 256, 512, 1024])
println("Experiment 3 complete. All experiments finished.")