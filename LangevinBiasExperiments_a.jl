# LangevinBiasExperiments.jl
#
# Verification script for the delocalization of bias phenomenon in ULA samplers.
# We tested 4 situations: 
# - Gaussian Distribution with Known Mean and Variance; Experimental results from ULA Chain and Theoretical Bias from lyapunov solver
# - Product Measure with Known Potential; Results from ULA Chain and MALA Chain
# - Sparse Graphical Model (Tridiagonal) with Known Potential; Results from ULA Chain and MALA Chain
# - Rotated Product Measure (Counter-Example) with Known Potential; Results from ULA Chain and MALA Chain

# Author: Michael Kang


using LinearAlgebra
using Distributions
using Random
using Plots
using ProgressMeter
using StatsBase
using JLD2
using Printf

# After the script, you will find the results directory in the same directory level as this script, including plots and raw data.
if !isdir("results")
    mkdir("results")
end


# Closed-book formula for computing the W₁ distance between two samples; numerical version
"""
Arguments: samples1, samples2 - vectors of samples from two distributions; they do not need to be of the same length.
"""
function compute_w1_dist(samples1::Vector, samples2::Vector)
    # Filter out non-finite values to prevent invalid results
    s1_finite = filter(isfinite, samples1)
    s2_finite = filter(isfinite, samples2)
    if isempty(s1_finite) || isempty(s2_finite) return Inf end

    N = min(length(s1_finite), length(s2_finite))
    s1_sorted = sort(s1_finite[1:N])
    s2_sorted = sort(s2_finite[1:N])
    return sum(abs.(s1_sorted - s2_sorted)) / N
end

function run_ula(grad_V, x0, n_samples, h; burn_in=10000)
    d = length(x0)
    x = copy(x0)
    samples = zeros(d, n_samples)
    for _ in 1:burn_in
        noise = randn(d)
        x .-= h .* grad_V(x) .- sqrt(2*h) .* noise
    end
    for i in 1:n_samples
        noise = randn(d)
        x .-= h .* grad_V(x) .- sqrt(2*h) .* noise
        samples[:, i] = x
    end
    return samples
end

function run_mala(V, grad_V, x0, n_samples, h; burn_in=10000)
    d = length(x0)
    x_current = copy(x0)
    samples = zeros(d, n_samples)
    accepted_count = 0
    p = Progress(burn_in + n_samples, 1, "Running MALA for d=$d...")
    for i in 1:(burn_in + n_samples)
        grad_current = grad_V(x_current)
        proposal_mean = x_current - h * grad_current
        noise = randn(d)
        x_proposal = proposal_mean + sqrt(2*h) * noise
        grad_proposal = grad_V(x_proposal)
        log_pi_proposal = -V(x_proposal)
        log_pi_current = -V(x_current)
        log_q_proposal_to_current = -norm(x_current - (x_proposal - h * grad_proposal))^2 / (4*h)
        log_q_current_to_proposal = -norm(x_proposal - proposal_mean)^2 / (4*h)
        log_alpha = (log_pi_proposal + log_q_proposal_to_current) - (log_pi_current + log_q_current_to_proposal)
        if log(rand()) < log_alpha
            x_current = x_proposal
            if i > burn_in; accepted_count += 1; end
        end
        if i > burn_in; samples[:, i - burn_in] = x_current; end
        next!(p)
    end
    return samples, accepted_count / n_samples
end


# Hyperparameters for the chain
const N_SAMPLES = 100000
const BURN_IN = 20000
const H = 0.005 # Step size for ULA and MALA, under 0.005 for appropriate accecptance rate
const DIMS = [8, 16, 32, 64, 128, 256, 512, 1024] # dimensions of probability distributions
const N_REPEATS = 5 # Repeat times for chains

# --- Experiment 1: Gaussian Case(the stationary distribution is a multi-variate Gaussian) ---
function experiment_gaussian(d)
    println("Running Gaussian experiment for d=$d...")
    A = randn(d, d) / sqrt(d)
    alpha = 0.1
    P_raw = A' * A # Ensure positive semi-definiteness
    precision_matrix = (P_raw + P_raw') / 2 + alpha * I # To ensure alpha-strong convexity and symmetry
    C_raw = inv(precision_matrix)
    covariance_matrix = (C_raw + C_raw') / 2 # To handle numerical nuances, ensure symmetry
    grad_V(x) = precision_matrix * x
    # solving Lyapunov equation, for Covariance Matrix: Sigma_h <-- this part uses the theory derived from the closed form of Langevin dynamics for Gaussian, thanks to the drift linear to x
    A_lyap = I - H * precision_matrix 
    C_lyap = 2 * H * Matrix(I, d, d)
    Sigma_h = copy(C_lyap)
    for _ in 1:500
        Sigma_h_new = A_lyap * Sigma_h * A_lyap' + C_lyap
        if norm(Sigma_h_new - Sigma_h) < 1e-10 * norm(Sigma_h); break; end
        Sigma_h = Sigma_h_new
    end
    Sigma_h = (Sigma_h + Sigma_h') / 2
    theoretical_bias = sqrt(2/pi) * abs(sqrt(covariance_matrix[1, 1]) - sqrt(Sigma_h[1, 1])) # mean is the same, bias only relevant to Sigma_h and Sigma
    
    ula_samples = run_ula(grad_V, zeros(d), N_SAMPLES, H)
    true_samples = rand(MvNormal(covariance_matrix), N_SAMPLES) # Because we know the true dist, no need for MALA
    numerical_bias = compute_w1_dist(ula_samples[1, :], true_samples[1, :])
    return numerical_bias, theoretical_bias
end

# --- Experiment 2: Product Measure Case, marginals independently coupled ---
function experiment_product(d)
    println("Running Product Measure experiment for d=$d...")
    V(x) = sum(0.5 .* x.^2 + 0.25 .* x.^4)
    grad_V(x) = x .+ x.^3
    ula_samples = run_ula(grad_V, zeros(d), N_SAMPLES, H)
    mala_samples, acc_rate = run_mala(V, grad_V, zeros(d), N_SAMPLES, H)
    bias = compute_w1_dist(ula_samples[1, :], mala_samples[1, :])
    return bias, acc_rate
end

# --- Experiment 3: Sparse Graphical Model (Tridiagonal), potential of one marginal only depend on its neighbor dimensions, very strictly local ---
function experiment_sparse(d)
    println("Running Sparse (Tridiagonal) experiment for d=$d...")
    V(x) = 0.5 * sum(x.^2) + (d > 1 ? 0.25 * sum((x[1:d-1] .- x[2:d]).^2) : 0.0)
    function grad_V(x)
        g = copy(x)
        if d > 1; diffs = x[1:d-1] - x[2:d]; g[1:d-1] .+= 0.5 .* diffs; g[2:d] .-= 0.5 .* diffs; end
        return g
    end
    ula_samples = run_ula(grad_V, zeros(d), N_SAMPLES, H)
    mala_samples, acc_rate = run_mala(V, grad_V, zeros(d), N_SAMPLES, H)
    bias = compute_w1_dist(ula_samples[1, :], mala_samples[1, :])
    return bias, acc_rate
end

# --- Experiment 4: Rotated Product Measure (Counter-Example) ---
const K_GAMMA = 2.0 # shape parameter
const THETA_GAMMA = 1.0 # scale parameter
const MEAN_GAMMA = K_GAMMA * THETA_GAMMA
const BOUNDARY = -MEAN_GAMMA
V_1d(z::Real) = z + MEAN_GAMMA > 0 ? (z + MEAN_GAMMA) / THETA_GAMMA - (K_GAMMA - 1) * log(z + MEAN_GAMMA) : Inf
grad_V_1d(z::Real) = 1.0 / THETA_GAMMA - (K_GAMMA - 1) / (z + MEAN_GAMMA)

# ULA for Rotated Potential, after fixing numerical nuances, inf values  
function run_ula_rotated(grad_V_1d_func::Function, Q::AbstractMatrix, Qt::AbstractMatrix, x0::Vector, n_samples::Int, h::Float64)
    d = length(x0)
    y = copy(x0)
    samples = zeros(d, n_samples)
    p = Progress(BURN_IN + n_samples, 1, "Running ULA for Rotated (d=$d)...")
    for i in 1:(BURN_IN + n_samples)
        y_current = copy(y) # Store current position

        # Propose a ULA step
        x = Qt * y_current
        grad_prod = grad_V_1d_func.(x)
        grad_rotated = Q * grad_prod
        noise = randn(d)
        y_proposal = y_current - h * grad_rotated + sqrt(2*h) * noise

        # Stability Check
        x_proposal_check = Qt * y_proposal
        if any(z -> z <= BOUNDARY, x_proposal_check)
            # If step is invalid, reject it (i.e., do nothing, y remains y_current)
            y = y_current
        else
            # If step is valid, accept it
            y = y_proposal
        end
        
        if i > BURN_IN; samples[:, i - BURN_IN] = y; end
        next!(p)
    end
    return samples
end

function run_mala_product(V_1d_func::Function, grad_V_1d_func::Function, x0::Vector, n_samples::Int, h::Float64)
    d = length(x0)
    x_current = copy(x0)
    samples = zeros(d, n_samples)
    accepted_count = 0
    p = Progress(BURN_IN + n_samples, 1, "Running MALA for Rotated (d=$d)...")
    for i in 1:(BURN_IN + n_samples)
        grad_current = grad_V_1d_func.(x_current)
        proposal_mean = x_current - h * grad_current
        noise = randn(d)
        x_proposal = proposal_mean + sqrt(2*h) * noise
        if any(z -> z <= BOUNDARY, x_proposal)
            if i > BURN_IN; samples[:, i - BURN_IN] = x_current; end; next!(p); continue
        end
        log_pi_proposal = -sum(V_1d_func.(x_proposal))
        log_pi_current = -sum(V_1d_func.(x_current))
        log_q_proposal_to_current = -norm(x_current - (x_proposal - h * grad_V_1d_func.(x_proposal)))^2 / (4*h)
        log_q_current_to_proposal = -norm(x_proposal - proposal_mean)^2 / (4*h)
        log_alpha = (log_pi_proposal + log_q_proposal_to_current) - (log_pi_current + log_q_current_to_proposal)
        if log(rand()) < log_alpha
            x_current = x_proposal
            if i > BURN_IN; accepted_count += 1; end
        end
        if i > BURN_IN; samples[:, i - BURN_IN] = x_current; end
        next!(p)
    end
    return samples, accepted_count / n_samples
end

get_rotation_matrix(d) = d > 1 ? vcat(ones(1, d) / sqrt(d), nullspace(ones(1, d))') : Matrix(1.0I, 1, 1)

function run_ula_1d_reflected(grad_V_1d_func, x0, n_samples, h)
    z = Float64(x0)
    samples = zeros(n_samples)
    for i in 1:(20000 + n_samples)
        z -= h * grad_V_1d_func(z) - sqrt(2*h) * randn()
        if z <= BOUNDARY; z = BOUNDARY + (BOUNDARY - z); end
        if i > 20000; samples[i - 20000] = z; end
    end
    return samples
end

function experiment_rotated(d)
    println("Running Rotated experiment for d=$d...")
    Random.seed!(d) # dimensions are different, making them, actually, good random seeds
    x0 = zeros(d)
    Q = get_rotation_matrix(d)
    ula_samples_rot = run_ula_rotated(grad_V_1d, Q, Q', x0, N_SAMPLES, H)
    mala_samples_prod, acc_rate = run_mala_product(V_1d, grad_V_1d, x0, N_SAMPLES, H)
    mala_samples_rot = Q * mala_samples_prod
    bias = compute_w1_dist(ula_samples_rot[1, :], mala_samples_rot[1, :])
    return bias, acc_rate
end


function main()
    results = Dict()
    println("\n" * "="^60); println("CALIBRATING 1D BIAS (δ) FOR ROTATED EXPERIMENT REFERENCE LINE"); println("="^60)
    cal_samples = 1000000
    ula_1d = run_ula_1d_reflected(grad_V_1d, 0.0, cal_samples, H)
    mala_1d, _ = run_mala_product(V_1d, grad_V_1d, [0.0], cal_samples, H)
    delta_h = mean(ula_1d) - mean(mala_1d)
    # Checking mean bias > 0, so the assumption is valid
    @printf "Calibration complete. Estimated 1D mean bias |δ| = %.6f\n" abs(delta_h)

    for (name, func) in [("gaussian", experiment_gaussian), ("product", experiment_product), ("sparse", experiment_sparse), ("rotated", experiment_rotated)]
        println("\n" * "="^60); println("STARTING $(uppercase(name)) EXPERIMENT"); println("="^60)
        bias_results = name == "gaussian" ? zeros(length(DIMS), N_REPEATS, 2) : zeros(length(DIMS), N_REPEATS)
        acc_rate_results = name != "gaussian" ? zeros(length(DIMS), N_REPEATS) : nothing
        for (i, d) in enumerate(DIMS)
            for j in 1:N_REPEATS
                println("\n--- Repetition $j/$N_REPEATS for d=$d ---")
                Random.seed!(i * 100 + j)
                if name == "gaussian"
                    num, theory = func(d)
                    bias_results[i, j, 1], bias_results[i, j, 2] = num, theory
                else
                    bias, acc_rate = func(d)
                    bias_results[i, j] = bias
                    acc_rate_results[i, j] = acc_rate
                end
            end
        end
        if name == "gaussian"
            results["gaussian_numerical"] = (mean(bias_results[:,:,1], dims=2), std(bias_results[:,:,1], dims=2))
            results["gaussian_theoretical"] = (mean(bias_results[:,:,2], dims=2), std(bias_results[:,:,2], dims=2))
        else
            results[name*"_bias"] = (mean(bias_results, dims=2), std(bias_results, dims=2))
            results[name*"_acc_rate"] = (mean(acc_rate_results, dims=2), std(acc_rate_results, dims=2))
        end
    end

    jldsave("results/all_experiments_data.jld2"; results)
    println("\nRaw results saved to results/all_experiments_data.jld2")

    # --- Plotting: Bias ---
    plt_bias = plot(DIMS, results["gaussian_numerical"][1], yerr=results["gaussian_numerical"][2], label="Gaussian (Numerical)", marker=:circle, xaxis=:log, yaxis=:log, xlabel="Dimension (d)", ylabel="W₁ Bias of First Marginal", title="ULA Marginal Bias Scaling (h=$H)", legend=:topleft, linewidth=2)
    plot!(plt_bias, DIMS, results["gaussian_theoretical"][1], yerr=results["gaussian_theoretical"][2], label="Gaussian (Theoretical)", marker=:square, linestyle=:dash, linewidth=2)
    plot!(plt_bias, DIMS, results["product_bias"][1], yerr=results["product_bias"][2], label="Product Measure", marker=:circle, linewidth=2)
    plot!(plt_bias, DIMS, results["sparse_bias"][1], yerr=results["sparse_bias"][2], label="Sparse (Tridiagonal)", marker=:circle, linewidth=2)
    plot!(plt_bias, DIMS, results["rotated_bias"][1], yerr=results["rotated_bias"][2], label="Rotated (Dense)", marker=:circle, linewidth=2)
    plot!(plt_bias, DIMS, abs(delta_h) .* sqrt.(DIMS), label="Rotated Mean Bias Ref. (O(√d))", linestyle=:dot, color=:black, linewidth=2.5)
    savefig(plt_bias, "results/all_experiments_plot.png")
    println("Bias plot saved to results/all_experiments_plot.png")
    
    # --- Plotting: Acceptance Rate ---
    println("\nPlotting MALA acceptance rates...")
    plt_acc = plot(DIMS, results["product_acc_rate"][1], yerr=results["product_acc_rate"][2], label="Product Measure", marker=:circle, xlabel="Dimension (d)", ylabel="MALA Acceptance Rate", title="MALA Sampler Acceptance Rates (h=$H)", legend=:bottomleft, linewidth=2, ylims=(0, 1.05))
    plot!(plt_acc, DIMS, results["sparse_acc_rate"][1], yerr=results["sparse_acc_rate"][2], label="Sparse (Tridiagonal)", marker=:circle, linewidth=2)
    plot!(plt_acc, DIMS, results["rotated_acc_rate"][1], yerr=results["rotated_acc_rate"][2], label="Rotated (Dense)", marker=:circle, linewidth=2)
    savefig(plt_acc, "results/all_experiments_acceptance_rates.png")
    println("Acceptance rate plot saved to results/all_experiments_acceptance_rates.png")
end

main()