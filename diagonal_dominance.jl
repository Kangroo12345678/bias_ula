# Verifying Bias Delocalization for Sampling Chains with Diagonal Dominant Potential-Hessians.

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
const H = 0.0001
const DIMS = [8, 16, 32, 64, 128, 256, 512]
const N_REPEATS = 5

# --- Setup ---
const RESULTS_DIR = "results/diag_dominant"
if !isdir(RESULTS_DIR)
    mkpath(RESULTS_DIR)
    println("Created '$RESULTS_DIR' directory for output plots and data.")
end


function compute_w1_dist(samples1::Vector, samples2::Vector)
    s1_finite = filter(isfinite, samples1); s2_finite = filter(isfinite, samples2)
    if isempty(s1_finite) || isempty(s2_finite) return Inf end
    N = min(length(s1_finite), length(s2_finite))
    s1_sorted = sort(s1_finite[1:N]); s2_sorted = sort(s2_finite[1:N])
    return sum(abs.(s1_sorted - s2_sorted)) / N
end

function calculate_iat(time_series::Vector)
    n = length(time_series)
    if n < 10 return 1.0 end
    max_lag = min(n - 1, Int(floor(n / 5))); acf_values = autocor(time_series, 1:max_lag)
    iat = 1.0
    for rho_k in acf_values; if rho_k > 0; iat += 2.0 * rho_k else break end; end
    return iat
end

function run_ula(grad_V, x0, n_samples, h; burn_in=BURN_IN)
    d = length(x0); x = copy(x0); samples = zeros(d, n_samples)
    p = Progress(burn_in + n_samples, 1, "Running ULA for d=$d...")
    for _ in 1:burn_in; x .-= h .* grad_V(x) .+ sqrt(2*h) .* randn(d); update!(p, 0); end
    for i in 1:n_samples; x .-= h .* grad_V(x) .+ sqrt(2*h) .* randn(d); samples[:, i] = x; next!(p); end
    return samples
end

function run_mala(V, grad_V, x0, n_samples, h; burn_in=BURN_IN)
    d = length(x0); x_current = copy(x0); samples = zeros(d, n_samples); accepted_count = 0
    p = Progress(burn_in + n_samples, 1, "Running MALA for d=$d...")
    for i in 1:(burn_in + n_samples)
        grad_current = grad_V(x_current); proposal_mean = x_current - h * grad_current
        x_proposal = proposal_mean + sqrt(2*h) * randn(d)
        log_pi_proposal = -V(x_proposal); log_pi_current = -V(x_current)
        grad_proposal = grad_V(x_proposal)
        log_q_proposal_to_current = -norm(x_current - (x_proposal - h * grad_proposal))^2 / (4*h)
        log_q_current_to_proposal = -norm(x_proposal - proposal_mean)^2 / (4*h)
        log_alpha = (log_pi_proposal + log_q_proposal_to_current) - (log_pi_current + log_q_current_to_proposal)
        if isfinite(log_alpha) && log(rand()) < log_alpha; x_current = x_proposal; if i > burn_in; accepted_count += 1; end; end
        if i > burn_in; samples[:, i - burn_in] = x_current; end; next!(p)
    end
    return samples, accepted_count / n_samples
end

function calculate_robust_stats(data::AbstractArray, name::String, quantity::String, dims_vec::Vector)
    means = zeros(size(data, 1)); stds = zeros(size(data, 1))
    for i in 1:size(data, 1)
        row_data = vec(data[i, :]); finite_data = filter(isfinite, row_data)
        if isempty(finite_data); means[i]=NaN; stds[i]=NaN; else; means[i]=mean(finite_data); stds[i]=length(finite_data)>1 ? std(finite_data) : 0.0; end
    end
    return (means, stds)
end

# --- Experiment Implementations ---

function create_diag_dominant_precision_matrix(d)
    diag_vals = exp.(range(log(1.0), log(1000.0), length=d))
    P = diagm(diag_vals)
    coupling_strength = 0.1 / d
    P .+= coupling_strength
    P -= diagm(fill(coupling_strength, d))
    return (P + P') / 2
end

function experiment_unrotated(d)
    P = create_diag_dominant_precision_matrix(d)
    V(x) = 0.5 * x' * P * x
    grad_V(x) = P * x
    ula_samples = run_ula(grad_V, zeros(d), N_SAMPLES, H)
    mala_samples, acc_rate = run_mala(V, grad_V, zeros(d), N_SAMPLES, H)
    bias = compute_w1_dist(ula_samples[1, :], mala_samples[1, :])
    iat_ula = calculate_iat(ula_samples[1, :]); iat_mala = calculate_iat(mala_samples[1, :])
    return bias, acc_rate, iat_ula, iat_mala
end

function experiment_rotated(d)
    # Start with the same diagonally dominant precision matrix
    P_unrotated = create_diag_dominant_precision_matrix(d)

    # Create a dense rotation matrix
    Q, _ = qr(randn(d, d))
    Q = Matrix(Q)

    # The new precision matrix P_rot = Q * P * Q' is now dense and NOT diagonally dominant.
    # This is the key transformation we are testing.
    P_rot = Q * P_unrotated * Q'
    
    V(y) = 0.5 * y' * P_rot * y
    grad_V(y) = P_rot * y
    ula_samples = run_ula(grad_V, zeros(d), N_SAMPLES, H)
    mala_samples, acc_rate = run_mala(V, grad_V, zeros(d), N_SAMPLES, H)
    bias = compute_w1_dist(ula_samples[1, :], mala_samples[1, :])
    iat_ula = calculate_iat(ula_samples[1, :]); iat_mala = calculate_iat(mala_samples[1, :])
    return bias, acc_rate, iat_ula, iat_mala
end

# --- Main Execution ---

function main()
    results = Dict()
    experiment_list = [
        ("unrotated", experiment_unrotated)
        ("rotated", experiment_rotated)
    ]

    for (name, func) in experiment_list
        println("\n" * "="^60 * "\nSTARTING $(uppercase(name)) EXPERIMENT\n" * "="^60)
        bias_results = zeros(length(DIMS), N_REPEATS)
        acc_rate_results = zeros(length(DIMS), N_REPEATS)
        iat_ula_results = zeros(length(DIMS), N_REPEATS)
        iat_mala_results = zeros(length(DIMS), N_REPEATS)
        
        for (i, d) in enumerate(DIMS)
            for j in 1:N_REPEATS
                println("\n--- Repetition $j/$N_REPEATS for d=$d ---")
                Random.seed!(i * 1000 + j)
                bias, acc_rate, iat_ula, iat_mala = func(d)
                bias_results[i, j] = bias; acc_rate_results[i, j] = acc_rate
                iat_ula_results[i, j] = iat_ula; iat_mala_results[i, j] = iat_mala
            end
        end
        
        results[name*"_bias"] = calculate_robust_stats(bias_results, name, "Bias", DIMS)
        results[name*"_acc_rate"] = calculate_robust_stats(acc_rate_results, name, "Acceptance Rate", DIMS)
        results[name*"_iat_ula"] = calculate_robust_stats(iat_ula_results, name, "IAT ULA", DIMS)
        results[name*"_iat_mala"] = calculate_robust_stats(iat_mala_results, name, "IAT MALA", DIMS)
    end

    data_path=joinpath(RESULTS_DIR, "conjecture_data_h$(H).jld2"); jldsave(data_path; results)
    println("\nRaw results saved to $data_path")
    
    # --- PLOTTING SECTION ---
    println("\nGenerating plots...")
    theme(:dark); default(linewidth=2, markersize=5, framestyle=:box, grid=false, legendfontsize=9)
    
    function plot_with_ribbon!(plt, dims, data_key, label; marker, linestyle=:solid, y_log_scale=false)
        means, stds = get(results, data_key, (nothing, nothing))
        if isnothing(means) || !any(isfinite, means); return; end
        plot_dims = collect(dims); plot_means = copy(means); plot_stds = isnothing(stds) ? nothing : copy(stds)
        if y_log_scale
            valid_indices = findall(x -> isfinite(x) && x > 0, plot_means)
            if isempty(valid_indices); return; end
            plot_dims = plot_dims[valid_indices]; plot_means = plot_means[valid_indices]
            if !isnothing(plot_stds) plot_stds = plot_stds[valid_indices] end
        end
        ribbon_arg = !isnothing(plot_stds) ? (y_log_scale ? (min.(plot_stds, plot_means .* 0.999), plot_stds) : plot_stds) : nothing
        plot!(plt, plot_dims, plot_means; ribbon=ribbon_arg, label=label, marker=marker, linestyle=linestyle, fillalpha=0.2)
    end
    
    # Bias Plot
    plt_bias = plot(xaxis=:log, yaxis=:log, xlabel="Dimension (d)", ylabel="W₁ Bias", title="Bias: Diagonally Dominant", legend=:topleft)
    plot_with_ribbon!(plt_bias, DIMS, "unrotated_bias", "Unrotated"; marker=:c, y_log_scale=true)
    plot_with_ribbon!(plt_bias, DIMS, "rotated_bias", "Rotated"; marker=:s, y_log_scale=true)
    savefig(plt_bias, joinpath(RESULTS_DIR, "bias_plot.png"))

    # Acceptance Rate Plot
    plt_acc = plot(xaxis=:log, xlabel="Dimension (d)", ylabel="Acceptance Rate", title="MALA Acceptance Rate", legend=:topright, ylims=(0,1.05))
    plot_with_ribbon!(plt_acc, DIMS, "unrotated_acc_rate", "Unrotated"; marker=:c)
    plot_with_ribbon!(plt_acc, DIMS, "rotated_acc_rate", "Rotated"; marker=:s)
    savefig(plt_acc, joinpath(RESULTS_DIR, "acceptance_rate_plot.png"))

    # IAT Plot
    plt_iat = plot(xaxis=:log, yaxis=:log, xlabel="Dimension (d)", ylabel="IAT", title="Sampler Mixing Time (IAT)", legend=:topleft)
    plot_with_ribbon!(plt_iat, DIMS, "unrotated_iat_ula", "ULA Unrotated"; marker=:c, y_log_scale=true)
    plot_with_ribbon!(plt_iat, DIMS, "unrotated_iat_mala", "MALA Unrotated"; marker=:c, linestyle=:dash, y_log_scale=true)
    plot_with_ribbon!(plt_iat, DIMS, "rotated_iat_ula", "ULA Rotated"; marker=:s, y_log_scale=true)
    plot_with_ribbon!(plt_iat, DIMS, "rotated_iat_mala", "MALA Rotated"; marker=:s, linestyle=:dash, y_log_scale=true)
    savefig(plt_iat, joinpath(RESULTS_DIR, "iat_plot.png"))
    
    println("All plots saved to '$RESULTS_DIR'.")
end

main()