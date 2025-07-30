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

# --- Setup: Find your results and data in this file. --- 
const RESULTS_DIR = "results/standard"
if !isdir(RESULTS_DIR)
    mkpath(RESULTS_DIR)
    println("Created '$RESULTS_DIR' directory for output plots and data.")
end

# --- Utility, Sampler, and Metrics Functions ---
function compute_w1_dist(samples1::Vector, samples2::Vector)
    s1_finite = filter(isfinite, samples1)
    s2_finite = filter(isfinite, samples2)
    if isempty(s1_finite) || isempty(s2_finite) return Inf end

    N = min(length(s1_finite), length(s2_finite))
    s1_sorted = sort(s1_finite[1:N])
    s2_sorted = sort(s2_finite[1:N])
    
    return sum(abs.(s1_sorted - s2_sorted)) / N
end

# Integrated Autocorrelation Time (IAT)
function calculate_iat(time_series::Vector)
    n = length(time_series)
    if n < 10 # Not enough samples for a meaningful IAT
        return 1.0
    end

    # Calculate ACF up to a reasonable max lag
    max_lag = min(n - 1, Int(floor(n / 5)))
    acf_values = autocor(time_series, 1:max_lag)
    
    iat = 1.0
    for rho_k in acf_values
        if rho_k > 0
            iat += 2.0 * rho_k
        else
            break
        end
    end
    return iat
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


# --- Experiment Implementations ---

function experiment_gaussian(d)
    A = randn(d, d) / sqrt(d); alpha = 0.1; P_raw = A' * A
    precision_matrix = (P_raw + P_raw') / 2 + alpha * I
    covariance_matrix = (inv(precision_matrix) + inv(precision_matrix)') / 2
    grad_V(x) = precision_matrix * x
    
    A_lyap = I - H * precision_matrix; C_lyap = 2 * H * Matrix(I, d, d)
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
    
    iat_ula = calculate_iat(ula_samples[1, :])
    
    return numerical_bias, theoretical_bias, iat_ula
end

function experiment_product(d)
    V(x) = sum(0.5 .* x.^2 + 0.25 .* x.^4)
    grad_V(x) = x .+ x.^3
    ula_samples = run_ula(grad_V, zeros(d), N_SAMPLES, H)
    mala_samples, acc_rate = run_mala(V, grad_V, zeros(d), N_SAMPLES, H)
    
    bias = compute_w1_dist(ula_samples[1, :], mala_samples[1, :])
    iat_ula = calculate_iat(ula_samples[1, :])
    iat_mala = calculate_iat(mala_samples[1, :])

    return bias, acc_rate, iat_ula, iat_mala
end

function experiment_sparse(d)
    V(x) = 0.5 * sum(x.^2) + (d > 1 ? 0.25 * sum((x[1:d-1] .- x[2:d]).^2) : 0.0)
    function grad_V(x)
        g = copy(x)
        if d > 1
            @inbounds for i in 1:d-1
                diff_val = 0.5 * (x[i] - x[i+1])
                g[i]  += diff_val; g[i+1] -= diff_val
            end
        end
        return g
    end
    ula_samples = run_ula(grad_V, zeros(d), N_SAMPLES, H)
    mala_samples, acc_rate = run_mala(V, grad_V, zeros(d), N_SAMPLES, H)
    
    bias = compute_w1_dist(ula_samples[1, :], mala_samples[1, :])
    iat_ula = calculate_iat(ula_samples[1, :])
    iat_mala = calculate_iat(mala_samples[1, :])
    
    return bias, acc_rate, iat_ula, iat_mala
end

# Rotated Case setup
const K_GAMMA=2.0; const THETA_GAMMA=1.0; const MEAN_GAMMA=K_GAMMA*THETA_GAMMA; const BOUNDARY=-MEAN_GAMMA
V_1d(z::Real) = z>BOUNDARY ? (z+MEAN_GAMMA)/THETA_GAMMA - (K_GAMMA-1)*log(z+MEAN_GAMMA) : Inf
grad_V_1d(z::Real) = z>BOUNDARY ? 1.0/THETA_GAMMA - (K_GAMMA-1)/(z+MEAN_GAMMA) : Inf
get_rotation_matrix(d) = d>1 ? vcat(ones(1,d)/sqrt(d), nullspace(ones(1,d))') : Matrix(1.0I,1,1)

function run_ula_rotated(grad_V_1d_func, Q, Qt, x0, n_samples, h)
    d = length(x0); y = copy(x0); samples = zeros(d, n_samples)
    p = Progress(BURN_IN + n_samples, 1, "Running ULA for Rotated (d=$d)...")
    for i in 1:(BURN_IN + n_samples)
        y_current = copy(y); x = Qt * y_current; grad_prod = grad_V_1d_func.(x)
        grad_rotated = Q * grad_prod; y_proposal = y_current - h * grad_rotated + sqrt(2*h) * randn(d)
        if any(z -> z <= BOUNDARY, Qt * y_proposal); y = y_current; else; y = y_proposal; end
        if i > BURN_IN; samples[:, i - BURN_IN] = y; end; next!(p)
    end
    return samples
end

function run_mala_product(V_1d_func, grad_V_1d_func, x0, n_samples, h; burn_in=BURN_IN)
    d = length(x0); x_current = copy(x0); samples = zeros(d, n_samples); accepted_count = 0
    p = Progress(burn_in + n_samples, 1, "Running MALA for Gamma Product (d=$d)...")
    for i in 1:(burn_in + n_samples)
        grad_current = grad_V_1d_func.(x_current); proposal_mean = x_current - h * grad_current
        x_proposal = proposal_mean + sqrt(2*h) * randn(d)
        if any(z -> z <= BOUNDARY, x_proposal); if i > burn_in; samples[:, i-burn_in] = x_current; end; next!(p); continue; end
        log_pi_proposal = -sum(V_1d_func.(x_proposal)); log_pi_current = -sum(V_1d_func.(x_current))
        log_q_proposal_to_current = -norm(x_current - (x_proposal - h*grad_V_1d_func.(x_proposal)))^2/(4*h)
        log_q_current_to_proposal = -norm(x_proposal - proposal_mean)^2 / (4*h)
        log_alpha = (log_pi_proposal + log_q_proposal_to_current) - (log_pi_current + log_q_current_to_proposal)
        if isfinite(log_alpha) && log(rand()) < log_alpha; x_current = x_proposal; if i > burn_in; accepted_count += 1; end; end
        if i > burn_in; samples[:, i-burn_in] = x_current; end; next!(p)
    end
    return samples, accepted_count / n_samples
end

function experiment_rotated(d)
    x0 = zeros(d); Q = get_rotation_matrix(d)
    ula_samples_rot = run_ula_rotated(grad_V_1d, Q, Q', x0, N_SAMPLES, H)
    mala_samples_prod, acc_rate = run_mala_product(V_1d, grad_V_1d, x0, N_SAMPLES, H)
    mala_samples_rot = Q * mala_samples_prod
    
    bias = compute_w1_dist(ula_samples_rot[1, :], mala_samples_rot[1, :])
    iat_ula = calculate_iat(ula_samples_rot[1, :])
    iat_mala = calculate_iat(mala_samples_rot[1, :])
    
    return bias, acc_rate, iat_ula, iat_mala
end

# --- Main Execution: handling numerical nuances ---
function calculate_robust_stats(data::AbstractArray, name::String, quantity::String, dims_vec::Vector)
    means = zeros(size(data, 1)); stds = zeros(size(data, 1))
    for i in 1:size(data, 1)
        row_data = vec(data[i, :]); finite_data = filter(isfinite, row_data)
        num_failed = length(row_data) - length(finite_data)
        if num_failed > 0; println("  WARNING: For $(name) ($quantity) at d=$(dims_vec[i]), $num_failed run(s) failed and were excluded."); end
        if isempty(finite_data); means[i]=NaN; stds[i]=NaN; else; means[i]=mean(finite_data); stds[i]=length(finite_data)>1 ? std(finite_data) : 0.0; end
    end
    return (means, stds)
end

function main()
    results = Dict()
    experiment_list = [
        ("gaussian", experiment_gaussian), ("product", experiment_product), 
        ("sparse", experiment_sparse), ("rotated", experiment_rotated)
    ]

    for (name, func) in experiment_list
        println("\n" * "="^60 * "\nSTARTING $(uppercase(name)) EXPERIMENT\n" * "="^60)
        
        # Initialize storage for all metrics
        bias_results = name=="gaussian" ? zeros(length(DIMS), N_REPEATS, 2) : zeros(length(DIMS), N_REPEATS)
        acc_rate_results = name!="gaussian" ? zeros(length(DIMS), N_REPEATS) : nothing
        iat_ula_results = zeros(length(DIMS), N_REPEATS)
        iat_mala_results = name!="gaussian" ? zeros(length(DIMS), N_REPEATS) : nothing
        
        for (i, d) in enumerate(DIMS)
            for j in 1:N_REPEATS
                println("\n--- Repetition $j/$N_REPEATS for d=$d ---")
                Random.seed!(i * 1000 + j)
                if name == "gaussian"
                    num_bias, theory_bias, iat_ula = func(d)
                    bias_results[i, j, 1], bias_results[i, j, 2] = num_bias, theory_bias
                    iat_ula_results[i, j] = iat_ula
                else
                    bias, acc_rate, iat_ula, iat_mala = func(d)
                    bias_results[i, j] = bias
                    acc_rate_results[i, j] = acc_rate
                    iat_ula_results[i, j] = iat_ula
                    iat_mala_results[i, j] = iat_mala
                end
            end
        end
        
        # Store aggregated results
        if name=="gaussian"
            results["gaussian_numerical_bias"]=calculate_robust_stats(bias_results[:,:,1],name,"Numerical Bias",DIMS)
            results["gaussian_theoretical_bias"]=calculate_robust_stats(bias_results[:,:,2],name,"Theoretical Bias",DIMS)
        else
            results[name*"_bias"]=calculate_robust_stats(bias_results,name,"Bias",DIMS)
            results[name*"_acc_rate"]=calculate_robust_stats(acc_rate_results,name,"Acceptance Rate",DIMS)
            results[name*"_iat_mala"]=calculate_robust_stats(iat_mala_results,name,"IAT MALA",DIMS)
        end
        results[name*"_iat_ula"]=calculate_robust_stats(iat_ula_results,name,"IAT ULA",DIMS)
    end

    data_path=joinpath(RESULTS_DIR, "all_experiments_data_h$(H).jld2"); jldsave(data_path; results)
    println("\nRaw results saved to $data_path")
    
    # --- PLOTTING SECTION ---
    println("\nGenerating plots...")
    
    theme(:dark); default(linewidth=2, markersize=5, framestyle=:box, grid=false, legendfontsize=9, tickfontsize=9, guidefontsize=11, titlefontsize=13, left_margin=5mm, bottom_margin=5mm)


    function plot_with_ribbon!(plt, dims, data_key, label; marker, linestyle=:solid, y_log_scale=false)
        means, stds = get(results, data_key, (nothing, nothing))

        if isnothing(means) || !any(isfinite, means)
            println("  Skipping plot for '$label': No valid data.")
            return
        end

        # Make mutable copies for filtering
        plot_dims = collect(dims)
        plot_means = copy(means)
        plot_stds = isnothing(stds) ? nothing : copy(stds)

        # Filter out non-positive mean values if y-axis is log scale
        if y_log_scale
            valid_indices = findall(x -> isfinite(x) && x > 0, plot_means)
            if isempty(valid_indices)
                println("  Skipping plot for '$label': No positive mean data for log scale.")
                return
            end
            plot_dims = plot_dims[valid_indices]
            plot_means = plot_means[valid_indices]
            if !isnothing(plot_stds)
                plot_stds = plot_stds[valid_indices]
            end
        end

        ribbon_arg = nothing
        if !isnothing(plot_stds)
            if y_log_scale
                lower_error = min.(plot_stds, plot_means .* 0.999) # Stop just short of zero
                ribbon_arg = (lower_error, plot_stds)
            else
                ribbon_arg = plot_stds 
            end
        end
        
        plot!(plt, plot_dims, plot_means; ribbon=ribbon_arg, label=label, marker=marker, linestyle=linestyle, fillalpha=0.2)
    end
    
    # --- Bias Plot ---
    println("Generating Bias Plot...")
    plt_bias = plot(xaxis=:log, yaxis=:log, xlabel="Dimension (d)", ylabel="W₁ Bias of First Marginal", title="ULA Marginal Bias Scaling (h=$H)", legend=:topleft, dpi=300)
    
    plot_with_ribbon!(plt_bias, DIMS, "gaussian_numerical_bias", "Gaussian (Numerical)"; marker=:circle, y_log_scale=true)
    
    # Plot theoretical bias (no ribbon) with log-scale filtering
    means_theory, _ = get(results, "gaussian_theoretical_bias", (nothing, nothing))
    if !isnothing(means_theory)
        valid_indices = findall(x -> isfinite(x) && x > 0, means_theory)
        if !isempty(valid_indices)
            plot!(plt_bias, DIMS[valid_indices], means_theory[valid_indices]; label="Gaussian (Theoretical)", marker=:square, linestyle=:dash)
        end
    end

    plot_with_ribbon!(plt_bias, DIMS, "product_bias", "Product (x⁴)"; marker=:diamond, y_log_scale=true)
    plot_with_ribbon!(plt_bias, DIMS, "sparse_bias", "Sparse"; marker=:utriangle, y_log_scale=true)
    plot_with_ribbon!(plt_bias, DIMS, "rotated_bias", "Rotated Gamma"; marker=:hexagon, y_log_scale=true)
    
    # O(sqrt(d)) reference line
    rotated_means, _ = get(results, "rotated_bias", (nothing, nothing))
    if !isnothing(rotated_means) && any(x -> isfinite(x) && x > 0, rotated_means)
        valid_indices = findall(x -> isfinite(x) && x > 0, rotated_means)
        if !isempty(valid_indices)
            first_valid_idx = valid_indices[1]
            C = rotated_means[first_valid_idx] / sqrt(DIMS[first_valid_idx])
            ref_dims = DIMS[valid_indices]
            plot!(plt_bias, ref_dims, C .* sqrt.(ref_dims); label="O(√d) Reference", linestyle=:dot, color=:white, linewidth=1.5)
        end
    end
    bias_plot_path=joinpath(RESULTS_DIR,"bias_scaling_plot_h$(H).png"); savefig(plt_bias, bias_plot_path); println("Bias plot saved to $bias_plot_path")

    # --- Acceptance Rate Plot ---
    println("Generating Acceptance Rate Plot...")
    plt_acc = plot(xaxis=:log, xlabel="Dimension (d)", ylabel="MALA Acceptance Rate", title="MALA Sampler Acceptance Rates (h=$H)", legend=:topright, ylims=(0,1.05), dpi=300)
    plot_with_ribbon!(plt_acc, DIMS, "product_acc_rate", "Product (x⁴)"; marker=:diamond)
    plot_with_ribbon!(plt_acc, DIMS, "sparse_acc_rate", "Sparse"; marker=:utriangle)
    plot_with_ribbon!(plt_acc, DIMS, "rotated_acc_rate", "Rotated Gamma"; marker=:hexagon)
    acc_plot_path=joinpath(RESULTS_DIR,"acceptance_rates_plot_h$(H).png"); savefig(plt_acc, acc_plot_path); println("Acceptance rate plot saved to $acc_plot_path")
    
    # --- IAT Plot ---
    println("Generating IAT Plot...")
    plt_iat = plot(xaxis=:log, yaxis=:log, xlabel="Dimension (d)", ylabel="Integrated Autocorrelation Time (IAT)", title="Sampler Mixing Time Scaling (h=$H)", legend=:topleft, dpi=300)
    
    plot_with_ribbon!(plt_iat, DIMS, "gaussian_iat_ula", "ULA Gaussian"; marker=:circle, y_log_scale=true)
    plot_with_ribbon!(plt_iat, DIMS, "product_iat_ula", "ULA Product"; marker=:diamond, y_log_scale=true)
    plot_with_ribbon!(plt_iat, DIMS, "sparse_iat_ula", "ULA Sparse"; marker=:utriangle, y_log_scale=true)
    plot_with_ribbon!(plt_iat, DIMS, "rotated_iat_ula", "ULA Rotated"; marker=:hexagon, y_log_scale=true)
    plot_with_ribbon!(plt_iat, DIMS, "product_iat_mala", "MALA Product"; marker=:diamond, linestyle=:dash, y_log_scale=true)
    plot_with_ribbon!(plt_iat, DIMS, "sparse_iat_mala", "MALA Sparse"; marker=:utriangle, linestyle=:dash, y_log_scale=true)
    plot_with_ribbon!(plt_iat, DIMS, "rotated_iat_mala", "MALA Rotated"; marker=:hexagon, linestyle=:dash, y_log_scale=true)
    
    # O(log d) reference line
    rotated_iat_means, _ = get(results, "rotated_iat_ula", (nothing, nothing))
    if !isnothing(rotated_iat_means) && any(x -> isfinite(x) && x > 0, rotated_iat_means)
        valid_indices = findall(x -> isfinite(x) && x > 0, rotated_iat_means)
        if !isempty(valid_indices)
            first_valid_idx = valid_indices[1]
            C = rotated_iat_means[first_valid_idx] / log(DIMS[first_valid_idx])
            ref_dims = DIMS[valid_indices]
            ref_line = C .* log.(ref_dims)
            plottable_ref_indices = findall(y -> isfinite(y) && y > 0, ref_line)
            if !isempty(plottable_ref_indices)
                 plot!(plt_iat, ref_dims[plottable_ref_indices], ref_line[plottable_ref_indices]; label="O(log d) Reference", linestyle=:dot, color=:white, linewidth=1.5)
            end
        end
    end
    iat_plot_path=joinpath(RESULTS_DIR,"iat_scaling_plot_h$(H).png"); savefig(plt_iat, iat_plot_path); println("IAT plot saved to $iat_plot_path")
end
main()