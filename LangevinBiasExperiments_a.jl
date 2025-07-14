# LangevinBiasExperiments.jl
#
# Verification script for the delocalization of bias phenomenon in ULA samplers.
#
# Author: Michael Kang


using LinearAlgebra
using Distributions
using Random
using Plots
using Plots.PlotMeasures
using ProgressMeter
using StatsBase
using JLD2
using Printf

# --- Hyperparameters ---
const N_SAMPLES = 100000
const BURN_IN = 20000
const H = 0.001
const DIMS = [8, 16, 32, 64, 128, 256, 512]
const N_REPEATS = 5

# --- Setup ---
const RESULTS_DIR = "results/standard"
if !isdir(RESULTS_DIR)
    mkpath(RESULTS_DIR)
    println("Created '$RESULTS_DIR' directory for output plots and data.")
end

# --- Utility & Sampler Functions (Expanded for Clarity) ---

function compute_w1_dist(samples1::Vector, samples2::Vector)
    s1_finite = filter(isfinite, samples1)
    s2_finite = filter(isfinite, samples2)
    if isempty(s1_finite) || isempty(s2_finite) return Inf end

    N = min(length(s1_finite), length(s2_finite))
    s1_sorted = sort(s1_finite[1:N])
    s2_sorted = sort(s2_finite[1:N])
    
    return sum(abs.(s1_sorted - s2_sorted)) / N
end

function run_ula(grad_V, x0, n_samples, h; burn_in=BURN_IN)
    d = length(x0)
    x = copy(x0)
    samples = zeros(d, n_samples)
    p = Progress(burn_in + n_samples, 1, "Running ULA for d=$d...")
    for _ in 1:burn_in
        x .-= h .* grad_V(x) .- sqrt(2*h) .* randn(d)
        update!(p, 0)
    end
    for i in 1:n_samples
        x .-= h .* grad_V(x) .- sqrt(2*h) .* randn(d)
        samples[:, i] = x
        next!(p)
    end
    return samples
end

function run_mala(V, grad_V, x0, n_samples, h; burn_in=BURN_IN)
    d = length(x0)
    x_current = copy(x0)
    samples = zeros(d, n_samples)
    accepted_count = 0
    p = Progress(burn_in + n_samples, 1, "Running MALA for d=$d...")
    for i in 1:(burn_in + n_samples)
        grad_current = grad_V(x_current)
        proposal_mean = x_current - h * grad_current
        x_proposal = proposal_mean + sqrt(2*h) * randn(d)
        
        grad_proposal = grad_V(x_proposal)
        log_pi_proposal = -V(x_proposal)
        log_pi_current = -V(x_current)
        
        log_q_proposal_to_current = -norm(x_current - (x_proposal - h * grad_proposal))^2 / (4*h)
        log_q_current_to_proposal = -norm(x_proposal - proposal_mean)^2 / (4*h)
        
        log_alpha = (log_pi_proposal + log_q_proposal_to_current) - (log_pi_current + log_q_current_to_proposal)
        
        if isfinite(log_alpha) && log(rand()) < log_alpha
            x_current = x_proposal
            if i > burn_in; accepted_count += 1; end
        end
        
        if i > burn_in; samples[:, i - burn_in] = x_current; end
        next!(p)
    end
    return samples, accepted_count / n_samples
end


# --- Experiment Implementations (Expanded for Clarity) ---

function experiment_gaussian(d)
    A = randn(d, d) / sqrt(d)
    alpha = 0.1
    P_raw = A' * A
    precision_matrix = (P_raw + P_raw') / 2 + alpha * I
    covariance_matrix = (inv(precision_matrix) + inv(precision_matrix)') / 2
    grad_V(x) = precision_matrix * x
    
    A_lyap = I - H * precision_matrix 
    C_lyap = 2 * H * Matrix(I, d, d)
    Sigma_h = copy(C_lyap)
    for _ in 1:500
        Sigma_h_new = A_lyap * Sigma_h * A_lyap' + C_lyap
        if norm(Sigma_h_new - Sigma_h) < 1e-10 * norm(Sigma_h); break; end
        Sigma_h = Sigma_h_new
    end
    Sigma_h = (Sigma_h + Sigma_h') / 2
    
    theoretical_bias = sqrt(2/pi) * abs(sqrt(covariance_matrix[1, 1]) - sqrt(Sigma_h[1, 1]))
    
    ula_samples = run_ula(grad_V, zeros(d), N_SAMPLES, H)
    true_samples = rand(MvNormal(covariance_matrix), N_SAMPLES)
    numerical_bias = compute_w1_dist(ula_samples[1, :], true_samples[1, :])
    
    return numerical_bias, theoretical_bias
end

function experiment_product(d)
    V(x) = sum(0.5 .* x.^2 + 0.25 .* x.^4)
    grad_V(x) = x .+ x.^3
    ula_samples = run_ula(grad_V, zeros(d), N_SAMPLES, H)
    mala_samples, acc_rate = run_mala(V, grad_V, zeros(d), N_SAMPLES, H)
    return compute_w1_dist(ula_samples[1, :], mala_samples[1, :]), acc_rate
end

function experiment_sparse(d)
    V(x) = 0.5 * sum(x.^2) + (d > 1 ? 0.25 * sum((x[1:d-1] .- x[2:d]).^2) : 0.0)
    function grad_V(x)
        g = copy(x)
        if d > 1
            @inbounds for i in 1:d-1
                diff_val = 0.5 * (x[i] - x[i+1])
                g[i]  += diff_val
                g[i+1] -= diff_val
            end
        end
        return g
    end
    ula_samples = run_ula(grad_V, zeros(d), N_SAMPLES, H)
    mala_samples, acc_rate = run_mala(V, grad_V, zeros(d), N_SAMPLES, H)
    return compute_w1_dist(ula_samples[1, :], mala_samples[1, :]), acc_rate
end

const K_GAMMA=2.0; const THETA_GAMMA=1.0; const MEAN_GAMMA=K_GAMMA*THETA_GAMMA; const BOUNDARY=-MEAN_GAMMA
V_1d(z::Real) = z>BOUNDARY ? (z+MEAN_GAMMA)/THETA_GAMMA - (K_GAMMA-1)*log(z+MEAN_GAMMA) : Inf
grad_V_1d(z::Real) = z>BOUNDARY ? 1.0/THETA_GAMMA - (K_GAMMA-1)/(z+MEAN_GAMMA) : Inf

function run_ula_rotated(grad_V_1d_func, Q, Qt, x0, n_samples, h)
    d = length(x0); y = copy(x0); samples = zeros(d, n_samples)
    p = Progress(BURN_IN + n_samples, 1, "Running ULA for Rotated (d=$d)...")
    for i in 1:(BURN_IN + n_samples)
        y_current = copy(y); x = Qt * y_current; grad_prod = grad_V_1d_func.(x)
        grad_rotated = Q * grad_prod; y_proposal = y_current - h * grad_rotated + sqrt(2*h) * randn(d)
        x_proposal_check = Qt * y_proposal
        if any(z -> z <= BOUNDARY, x_proposal_check); y = y_current; else; y = y_proposal; end
        if i > BURN_IN; samples[:, i - BURN_IN] = y; end
        next!(p)
    end
    return samples
end

function run_mala_product(V_1d_func, grad_V_1d_func, x0, n_samples, h; burn_in=BURN_IN)
    d = length(x0); x_current = copy(x0); samples = zeros(d, n_samples); accepted_count = 0
    p = Progress(burn_in + n_samples, 1, "Running MALA for Gamma Product (d=$d)...")
    for i in 1:(burn_in + n_samples)
        grad_current = grad_V_1d_func.(x_current); proposal_mean = x_current - h * grad_current
        x_proposal = proposal_mean + sqrt(2*h) * randn(d)
        if any(z -> z <= BOUNDARY, x_proposal)
            if i > burn_in; samples[:, i - burn_in] = x_current; end
            next!(p); continue
        end
        log_pi_proposal = -sum(V_1d_func.(x_proposal)); log_pi_current = -sum(V_1d_func.(x_current))
        log_q_proposal_to_current = -norm(x_current - (x_proposal - h*grad_V_1d_func.(x_proposal)))^2/(4*h)
        log_q_current_to_proposal = -norm(x_proposal - proposal_mean)^2 / (4*h)
        log_alpha = (log_pi_proposal + log_q_proposal_to_current) - (log_pi_current + log_q_current_to_proposal)
        if isfinite(log_alpha) && log(rand()) < log_alpha
            x_current = x_proposal
            if i > burn_in; accepted_count += 1; end
        end
        if i > burn_in; samples[:, i - burn_in] = x_current; end
        next!(p)
    end
    return samples, accepted_count / n_samples
end

get_rotation_matrix(d) = d>1 ? vcat(ones(1,d)/sqrt(d), nullspace(ones(1,d))') : Matrix(1.0I,1,1)

function experiment_rotated(d)
    x0 = zeros(d); Q = get_rotation_matrix(d)
    ula_samples_rot = run_ula_rotated(grad_V_1d, Q, Q', x0, N_SAMPLES, H)
    mala_samples_prod, acc_rate = run_mala_product(V_1d, grad_V_1d, x0, N_SAMPLES, H)
    mala_samples_rot = Q * mala_samples_prod
    return compute_w1_dist(ula_samples_rot[1, :], mala_samples_rot[1, :]), acc_rate
end

function run_ula_unrotated_gamma(grad_V_1d_func, x0, n_samples, h; burn_in=BURN_IN)
    d=length(x0); x=copy(x0); samples=zeros(d,n_samples)
    p = Progress(burn_in + n_samples, 1, "Running ULA for Unrotated Gamma (d=$d)...")
    for i in 1:(burn_in + n_samples)
        grad = grad_V_1d_func.(x); x .-= h .* grad .- sqrt(2 * h) .* randn(d)
        for j in 1:d; if x[j] <= BOUNDARY; x[j] = BOUNDARY + (BOUNDARY - x[j]); end; end
        if i > burn_in; samples[:, i - burn_in] = x; end
        next!(p)
    end
    return samples
end

function experiment_unrotated_gamma(d)
    x0 = zeros(d); ula_samples = run_ula_unrotated_gamma(grad_V_1d, x0, N_SAMPLES, H)
    mala_samples, acc_rate = run_mala_product(V_1d, grad_V_1d, x0, N_SAMPLES, H)
    return compute_w1_dist(ula_samples[1, :], mala_samples[1, :]), acc_rate
end

function run_ula_1d_reflected(grad_V_1d_func, x0, n_samples, h)
    z=Float64(x0); samples=zeros(n_samples); cal_burn_in=20000
    for i in 1:(cal_burn_in + n_samples)
        z -= h * grad_V_1d_func(z) - sqrt(2*h) * randn()
        if z <= BOUNDARY; z = BOUNDARY + (BOUNDARY - z); end
        if i > cal_burn_in; samples[i-cal_burn_in] = z; end
    end
    return samples
end

# --- Main Execution ---
function calculate_robust_stats(data::AbstractArray, name::String, quantity::String, dims_vec::Vector)
    means = zeros(size(data, 1)); stds = zeros(size(data, 1))
    for i in 1:size(data, 1)
        row_data = vec(data[i, :]); finite_data = filter(isfinite, row_data)
        num_failed = length(row_data) - length(finite_data)
        if num_failed > 0
            println("  WARNING: For $(name) ($quantity) at d=$(dims_vec[i]), $num_failed run(s) failed and were excluded.")
        end
        if isempty(finite_data); means[i]=NaN; stds[i]=NaN
        else; means[i]=mean(finite_data); stds[i]=length(finite_data)>1 ? std(finite_data) : 0.0; end
    end
    return (means, stds)
end

function main()
    results = Dict()
    println("\n" * "="^60 * "\nCALIBRATING 1D BIAS (δ)\n" * "="^60)
    cal_samples=1000000; ula_1d=run_ula_1d_reflected(grad_V_1d, 0.0, cal_samples, H)
    mala_1d, _ = run_mala_product(V_1d, grad_V_1d, [0.0], cal_samples, H; burn_in=20000)
    delta_h=compute_w1_dist(ula_1d, mala_1d[:]); @printf "Calibration complete. Estimated 1D W_1 bias |δ|=%.6f\n" delta_h

    experiment_list = [
        ("gaussian", experiment_gaussian), ("product", experiment_product), ("sparse", experiment_sparse),
        ("unrotated_gamma", experiment_unrotated_gamma), ("rotated", experiment_rotated)
    ]

    for (name, func) in experiment_list
        println("\n" * "="^60 * "\nSTARTING $(uppercase(name)) EXPERIMENT\n" * "="^60)
        bias_results = name=="gaussian" ? zeros(length(DIMS), N_REPEATS, 2) : zeros(length(DIMS), N_REPEATS)
        acc_rate_results = name!="gaussian" ? zeros(length(DIMS), N_REPEATS) : nothing
        for (i, d) in enumerate(DIMS)
            for j in 1:N_REPEATS
                println("\n--- Repetition $j/$N_REPEATS for d=$d ---")
                Random.seed!(i * 1000 + j)
                if name == "gaussian"
                    num, theory = func(d); bias_results[i, j, 1], bias_results[i, j, 2] = num, theory
                else
                    bias, acc_rate = func(d); bias_results[i, j] = bias
                    if acc_rate_results !== nothing; acc_rate_results[i, j] = acc_rate; end
                end
            end
        end
        if name=="gaussian"
            results["gaussian_numerical"]=calculate_robust_stats(bias_results[:,:,1],name,"Numerical Bias",DIMS)
            results["gaussian_theoretical"]=calculate_robust_stats(bias_results[:,:,2],name,"Theoretical Bias",DIMS)
        else
            results[name*"_bias"]=calculate_robust_stats(bias_results,name,"Bias",DIMS)
            if acc_rate_results!==nothing; results[name*"_acc_rate"]=calculate_robust_stats(acc_rate_results,name,"Acceptance Rate",DIMS); end
        end
    end

    data_path=joinpath(RESULTS_DIR, "all_experiments_data_h$(H).jld2"); jldsave(data_path; results)
    println("\nRaw results saved to $data_path")
    println("\nGenerating plots...")
    
    theme(:dark); default(linewidth=2.5, markersize=6, framestyle=:box, grid=false, legendfontsize=10, tickfontsize=10, guidefontsize=12, titlefontsize=14, left_margin=5mm, bottom_margin=5mm)
    
    # --- Bias Plot ---
    plt_bias = plot(xaxis=:log, yaxis=:log, xlabel="Dimension (d)", ylabel="W_1 Bias of First Marginal", title="ULA Marginal Bias Scaling (h=$H)", legend=:topleft, dpi=300)
    
    # NOTE: Plotting without error bars (`yerr`) to isolate the plotting package bug.
    plot!(plt_bias, DIMS, get(results, "gaussian_numerical", ([],[]))[1], label="Gaussian (Numerical)", marker=:circle)
    plot!(plt_bias, DIMS, get(results, "gaussian_theoretical", ([],[]))[1], label="Gaussian (Theoretical)", marker=:square, linestyle=:dash)
    plot!(plt_bias, DIMS, get(results, "product_bias", ([],[]))[1], label="Product (x⁴)", marker=:diamond)
    plot!(plt_bias, DIMS, get(results, "sparse_bias", ([],[]))[1], label="Sparse", marker=:utriangle)
    plot!(plt_bias, DIMS, get(results, "unrotated_gamma_bias", ([],[]))[1], label="Unrotated Gamma", marker=:star5)
    plot!(plt_bias, DIMS, get(results, "rotated_bias", ([],[]))[1], label="Rotated Gamma", marker=:hexagon)

    if !any(isnan, get(results, "rotated_bias", ([NaN],))[1]); C=results["rotated_bias"][1][1]/sqrt(DIMS[1]); plot!(plt_bias, DIMS, C.*sqrt.(DIMS), label="O(sqrt(d)) Reference", linestyle=:dot, color=:white); end
    
    bias_plot_path=joinpath(RESULTS_DIR,"bias_scaling_plot_h$(H).png"); savefig(plt_bias, bias_plot_path); println("Bias plot saved to $bias_plot_path")

    # --- Acceptance Rate Plot ---
    plt_acc = plot(xlabel="Dimension (d)", ylabel="MALA Acceptance Rate", title="MALA Sampler Acceptance Rates (h=$H)", legend=:bottomleft, ylims=(0,1.05), dpi=300)
    
    # NOTE: Plotting without error bars (`yerr`) to isolate the plotting package bug.
    plot!(plt_acc, DIMS, get(results, "product_acc_rate", ([],[]))[1], label="Product (x⁴)", marker=:diamond)
    plot!(plt_acc, DIMS, get(results, "sparse_acc_rate", ([],[]))[1], label="Sparse", marker=:utriangle)
    plot!(plt_acc, DIMS, get(results, "unrotated_gamma_acc_rate", ([],[]))[1], label="Unrotated Gamma", marker=:star5)
    plot!(plt_acc, DIMS, get(results, "rotated_acc_rate", ([],[]))[1], label="Rotated Gamma", marker=:hexagon)
    
    acc_plot_path=joinpath(RESULTS_DIR,"acceptance_rates_plot_h$(H).png"); savefig(plt_acc, acc_plot_path); println("Acceptance rate plot saved to $acc_plot_path")
end

main()