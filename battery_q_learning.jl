using Random
using Dates
using Statistics
using DataFrames
using CSV
using XLSX

const ACTIONS = (-1, 0, 1) # -1 = charge, 0 = idle, 1 = discharge
const ACTION_NAMES = Dict(-1 => "charge", 0 => "idle", 1 => "discharge")

Base.@kwdef struct ModelParams
    # Data
    train_years::Vector{Int} = collect(2016:2024)
    test_years::Vector{Int} = [2025]
    settlement_point_name::String = "hub_north"
    consolidated_data_file::String = joinpath("data", "hub_north_2021_2026.xlsx")
    consolidated_sheet_name::String = "hub_north_all"
    output_dir::String = "results"

    # Battery operation
    eta_rt::Float64 = 0.937
    pmax_mw::Float64 = 250.0
    dt_hours::Float64 = 0.25
    operation_hours::Int = typemax(Int)

    # SOC / capacity (MWh)
    soc0_mwh::Float64 = 1000.0
    soc_min_mwh::Float64 = 0.0
    soc_max_mwh::Float64 = 1000.0

    # Lifetime and degradation
    soh0::Float64 = 1.0
    soh_min::Float64 = 0.85
    battery_lifetime_years::Float64 = 15.0
    max_cycles::Float64 = 7300.0
    soh_drop_per_cycle::Float64 = (soh0 - soh_min) / max_cycles
    cycle_pacing_tolerance_cycles::Float64 = 0.10
    lifecycle_years::Float64 = 15.0

    # Discretization (recommended starting point)
    n_price_bins::Int = 21
    n_soc_bins::Int = 15
    n_soh_bins::Int = 5
    n_life_bins::Int = 5

    # Q-learning
    episodes::Int = 50
    alpha0::Float64 = 0.15
    alpha_min::Float64 = 0.03
    gamma::Float64 = 0.995
    epsilon0::Float64 = 1.0
    epsilon_min::Float64 = 0.05
    epsilon_decay::Float64 = 0.90
    degradation_penalty_per_cycle::Float64 = 50.0
    pacing_penalty_weight::Float64 = 300.0
    enforce_terminal_soc::Bool = true
    terminal_soc_penalty_per_mwh::Float64 = 50.0
    seed::Int = 42
end

Base.@kwdef struct Discretizer
    price_cutpoints::Vector{Float64}
    n_price_bins::Int
    n_soc_bins::Int
    n_soh_bins::Int
    n_life_bins::Int
end

function load_price_data(params::ModelParams)
    file = params.consolidated_data_file
    isfile(file) || error("Data file not found: $(abspath(file))")

    ext = lowercase(splitext(file)[2])
    df = if ext == ".csv"
        CSV.read(file, DataFrame)
    elseif ext in (".xlsx", ".xlsm", ".xls")
        DataFrame(XLSX.readtable(file, params.consolidated_sheet_name))
    else
        error("Unsupported file type: $ext")
    end

    normalized = Symbol.(lowercase.(replace.(string.(names(df)), " " => "_")))
    rename!(df, normalized)

    present_cols = Set(Symbol.(names(df)))
    for col in (:timestamp, :year, :price)
        col in present_cols || error("Missing required column: $col")
    end

    if !(eltype(df.timestamp) <: TimeType)
        df.timestamp = DateTime.(df.timestamp)
    end
    df.year = Int.(df.year)
    df.price = Float64.(df.price)

    sort!(df, :timestamp)
    return df
end

function split_train_test(df::DataFrame, params::ModelParams)
    train_set = Set(params.train_years)
    test_set = Set(params.test_years)
    train_df = df[in.(df.year, Ref(train_set)), :]
    test_df = df[in.(df.year, Ref(test_set)), :]

    nrow(train_df) > 0 || error("No training rows found for years: $(params.train_years)")
    nrow(test_df) > 0 || error("No testing rows found for years: $(params.test_years)")
    return train_df, test_df
end

function price_cutpoints(prices::AbstractVector{<:Real}, n_bins::Int)
    n_bins >= 2 || error("n_price_bins must be >= 2")
    qpoints = [i / n_bins for i in 1:(n_bins - 1)]
    cuts = [quantile(prices, q) for q in qpoints]
    for i in 2:length(cuts)
        if cuts[i] <= cuts[i - 1]
            cuts[i] = nextfloat(cuts[i - 1])
        end
    end
    return Float64.(cuts)
end

function build_discretizer(prices::AbstractVector{<:Real}, params::ModelParams)
    Discretizer(
        price_cutpoints = price_cutpoints(prices, params.n_price_bins),
        n_price_bins = params.n_price_bins,
        n_soc_bins = params.n_soc_bins,
        n_soh_bins = params.n_soh_bins,
        n_life_bins = params.n_life_bins,
    )
end

@inline function cut_bin(x::Float64, cuts::Vector{Float64}, n_bins::Int)
    return clamp(searchsortedlast(cuts, x) + 1, 1, n_bins)
end

@inline function frac_bin(x::Float64, n_bins::Int)
    return clamp(floor(Int, x * n_bins) + 1, 1, n_bins)
end

function state_index(
    price::Float64,
    soc::Float64,
    soh::Float64,
    cycles_used::Float64,
    disc::Discretizer,
    params::ModelParams,
)
    price_b = cut_bin(price, disc.price_cutpoints, disc.n_price_bins)

    usable_cap = max(params.soc_max_mwh * (soh / params.soh0), 1e-9)
    soc_frac = clamp((soc - params.soc_min_mwh) / max(usable_cap - params.soc_min_mwh, 1e-9), 0.0, 1.0)
    soc_b = frac_bin(soc_frac, disc.n_soc_bins)

    soh_frac = clamp((soh - params.soh_min) / max(params.soh0 - params.soh_min, 1e-9), 0.0, 1.0)
    soh_b = frac_bin(soh_frac, disc.n_soh_bins)

    life_frac = clamp((params.max_cycles - cycles_used) / params.max_cycles, 0.0, 1.0)
    life_b = frac_bin(life_frac, disc.n_life_bins)

    idx = price_b
    idx = (idx - 1) * disc.n_soc_bins + soc_b
    idx = (idx - 1) * disc.n_soh_bins + soh_b
    idx = (idx - 1) * disc.n_life_bins + life_b
    return idx
end

function n_states(disc::Discretizer)
    return disc.n_price_bins * disc.n_soc_bins * disc.n_soh_bins * disc.n_life_bins
end

function effective_cycle_tolerance(params::ModelParams)
    annual_target = params.max_cycles / params.battery_lifetime_years
    if params.cycle_pacing_tolerance_cycles <= 1.0
        return annual_target * params.cycle_pacing_tolerance_cycles
    end
    return params.cycle_pacing_tolerance_cycles
end

function step_battery(
    price::Float64,
    action::Int,
    soc::Float64,
    soh::Float64,
    cycles_used::Float64,
    step_id::Int,
    params::ModelParams,
)
    eta_c = sqrt(params.eta_rt)
    eta_d = sqrt(params.eta_rt)
    step_mwh = params.pmax_mw * params.dt_hours

    soc_lb = params.soc_min_mwh
    soc_ub = max(params.soc_min_mwh, params.soc_max_mwh * (soh / params.soh0))
    soc = clamp(soc, soc_lb, soc_ub)

    cash = 0.0
    delta_soc = 0.0

    if action == -1
        room = max(0.0, soc_ub - soc)
        delta_soc = min(step_mwh * eta_c, room)
        if delta_soc > 0
            grid_in_mwh = delta_soc / eta_c
            cash -= price * grid_in_mwh
        end
    elseif action == 1
        available = max(0.0, soc - soc_lb)
        batt_out_mwh = min(step_mwh, available)
        delta_soc = -batt_out_mwh
        if batt_out_mwh > 0
            grid_out_mwh = batt_out_mwh * eta_d
            cash += price * grid_out_mwh
        end
    end

    soc_new = clamp(soc + delta_soc, soc_lb, soc_ub)
    cycle_inc = abs(delta_soc) / (2 * params.soc_max_mwh)
    soh_loss = params.soh_drop_per_cycle * cycle_inc
    soh_new = max(params.soh_min, soh - soh_loss)
    cycles_new = min(params.max_cycles, cycles_used + cycle_inc)

    elapsed_years = (step_id * params.dt_hours) / (24 * 365)
    target_cycles = elapsed_years * (params.max_cycles / params.battery_lifetime_years)
    overuse = max(0.0, cycles_new - target_cycles - effective_cycle_tolerance(params))
    overuse_scaled = overuse / max(params.max_cycles / params.battery_lifetime_years, 1e-9)

    pace_penalty = params.pacing_penalty_weight * overuse_scaled^2
    deg_penalty = params.degradation_penalty_per_cycle * cycle_inc
    reward = cash - deg_penalty - pace_penalty

    return soc_new, soh_new, cycles_new, cash, reward
end

function max_steps(df::DataFrame, params::ModelParams)
    if params.operation_hours == typemax(Int)
        return nrow(df)
    end
    return min(nrow(df), max(1, floor(Int, params.operation_hours / params.dt_hours)))
end

function epsilon_at_episode(params::ModelParams, episode::Int)
    return max(params.epsilon_min, params.epsilon0 * params.epsilon_decay^(episode - 1))
end

function alpha_at_episode(params::ModelParams, episode::Int)
    return max(params.alpha_min, params.alpha0 * 0.95^(episode - 1))
end

function train_q_learning(train_df::DataFrame, params::ModelParams)
    Random.seed!(params.seed)
    disc = build_discretizer(train_df.price, params)
    q = zeros(Float64, n_states(disc), length(ACTIONS))

    steps = max_steps(train_df, params)
    prices = train_df.price[1:steps]

    episode_profit = zeros(Float64, params.episodes)
    episode_reward = zeros(Float64, params.episodes)
    episode_cycles = zeros(Float64, params.episodes)
    episode_epsilon = zeros(Float64, params.episodes)
    episode_alpha = zeros(Float64, params.episodes)

    for ep in 1:params.episodes
        soc = clamp(params.soc0_mwh, params.soc_min_mwh, params.soc_max_mwh)
        soh = params.soh0
        cycles = 0.0
        total_cash = 0.0
        total_reward = 0.0

        epsilon = epsilon_at_episode(params, ep)
        alpha = alpha_at_episode(params, ep)
        episode_epsilon[ep] = epsilon
        episode_alpha[ep] = alpha

        for t in 1:steps
            s = state_index(prices[t], soc, soh, cycles, disc, params)
            a_idx = if rand() < epsilon
                rand(1:length(ACTIONS))
            else
                argmax(@view q[s, :])
            end

            action = ACTIONS[a_idx]
            soc2, soh2, cycles2, cash, reward =
                step_battery(prices[t], action, soc, soh, cycles, t, params)

            if t == steps && params.enforce_terminal_soc
                terminal_gap = abs(soc2 - params.soc0_mwh)
                terminal_penalty = params.terminal_soc_penalty_per_mwh * terminal_gap
                cash -= terminal_penalty
                reward -= terminal_penalty
            end

            target = reward
            if t < steps
                s2 = state_index(prices[t + 1], soc2, soh2, cycles2, disc, params)
                target += params.gamma * maximum(@view q[s2, :])
            end

            q[s, a_idx] += alpha * (target - q[s, a_idx])
            soc, soh, cycles = soc2, soh2, cycles2
            total_cash += cash
            total_reward += reward
        end

        episode_profit[ep] = total_cash
        episode_reward[ep] = total_reward
        episode_cycles[ep] = cycles
    end

    return (;
        q,
        disc,
        episode_profit,
        episode_reward,
        episode_cycles,
        episode_epsilon,
        episode_alpha,
    )
end

function evaluate_policy(test_df::DataFrame, q::Matrix{Float64}, disc::Discretizer, params::ModelParams)
    steps = max_steps(test_df, params)

    ts = Vector{DateTime}(undef, steps)
    price_col = Vector{Float64}(undef, steps)
    action_col = Vector{String}(undef, steps)
    soc_col = Vector{Float64}(undef, steps)
    soh_col = Vector{Float64}(undef, steps)
    cycle_col = Vector{Float64}(undef, steps)
    cash_col = Vector{Float64}(undef, steps)
    cum_col = Vector{Float64}(undef, steps)

    soc = clamp(params.soc0_mwh, params.soc_min_mwh, params.soc_max_mwh)
    soh = params.soh0
    cycles = 0.0
    cumulative = 0.0

    for t in 1:steps
        price = test_df.price[t]
        s = state_index(price, soc, soh, cycles, disc, params)
        a_idx = argmax(@view q[s, :])
        action = ACTIONS[a_idx]

        soc, soh, cycles, cash, _ = step_battery(price, action, soc, soh, cycles, t, params)
        if t == steps && params.enforce_terminal_soc
            terminal_gap = abs(soc - params.soc0_mwh)
            cash -= params.terminal_soc_penalty_per_mwh * terminal_gap
        end
        cumulative += cash

        ts[t] = test_df.timestamp[t]
        price_col[t] = price
        action_col[t] = ACTION_NAMES[action]
        soc_col[t] = soc
        soh_col[t] = soh
        cycle_col[t] = cycles
        cash_col[t] = cash
        cum_col[t] = cumulative
    end

    timeline = DataFrame(
        timestamp = ts,
        price = price_col,
        action = action_col,
        soc_mwh = soc_col,
        soh = soh_col,
        cycles_used = cycle_col,
        step_cash = cash_col,
        cumulative_profit = cum_col,
    )

    summary = (
        total_profit = cumulative,
        final_soh = soh,
        cycles_used = cycles,
        charge_steps = count(==("charge"), action_col),
        idle_steps = count(==("idle"), action_col),
        discharge_steps = count(==("discharge"), action_col),
    )

    return summary, timeline
end

function params_to_namedtuple(params::ModelParams)
    return (; (name => getfield(params, name) for name in fieldnames(ModelParams))...)
end

function quick_bin_search(
    params::ModelParams = ModelParams();
    price_candidates::Vector{Int} = [21, 31, 41],
    soc_candidates::Vector{Int} = [15, 21, 31],
    soh_candidates::Vector{Int} = [5, 7, 9],
    life_candidates::Vector{Int} = [5, 7, 9],
    tuning_episodes::Int = 8,
    validation_year::Int = maximum(params.train_years),
)
    fit_years = sort([y for y in params.train_years if y < validation_year])
    isempty(fit_years) && error("Need at least one fit year before validation_year=$validation_year")

    base = ModelParams(;
        params_to_namedtuple(params)...,
        train_years = fit_years,
        test_years = [validation_year],
        episodes = tuning_episodes,
    )

    df = load_price_data(base)
    fit_df, val_df = split_train_test(df, base)
    results = DataFrame(
        n_price_bins = Int[],
        n_soc_bins = Int[],
        n_soh_bins = Int[],
        n_life_bins = Int[],
        val_profit = Float64[],
        val_final_soh = Float64[],
        val_cycles = Float64[],
    )

    for pb in price_candidates, sb in soc_candidates, hb in soh_candidates, lb in life_candidates
        trial = ModelParams(;
            params_to_namedtuple(base)...,
            n_price_bins = pb,
            n_soc_bins = sb,
            n_soh_bins = hb,
            n_life_bins = lb,
        )

        model = train_q_learning(fit_df, trial)
        summary, _ = evaluate_policy(val_df, model.q, model.disc, trial)
        push!(
            results,
            (pb, sb, hb, lb, summary.total_profit, summary.final_soh, summary.cycles_used),
        )
    end

    sort!(results, :val_profit, rev = true)
    return results
end

function run_pipeline(params::ModelParams = ModelParams())
    df = load_price_data(params)
    train_df, test_df = split_train_test(df, params)

    model = train_q_learning(train_df, params)
    summary, timeline = evaluate_policy(test_df, model.q, model.disc, params)

    println("=== Dataset ===")
    println("Train years: $(params.train_years), rows: $(nrow(train_df))")
    println("Test years : $(params.test_years), rows: $(nrow(test_df))")
    println()
    println("=== Training ===")
    println("Episodes: $(params.episodes)")
    println("Episode 1 epsilon/alpha : $(round(model.episode_epsilon[1], digits = 4)) / $(round(model.episode_alpha[1], digits = 4))")
    println("Last episode epsilon/alpha: $(round(model.episode_epsilon[end], digits = 4)) / $(round(model.episode_alpha[end], digits = 4))")
    println("Last-episode train profit: $(round(model.episode_profit[end], digits = 2))")
    println("Last-episode cycles used : $(round(model.episode_cycles[end], digits = 3))")
    println()
    println("=== Test Result ===")
    println("Total profit: $(round(summary.total_profit, digits = 2))")
    println("Final SOH  : $(round(summary.final_soh, digits = 6))")
    println("Cycles used: $(round(summary.cycles_used, digits = 3))")
    println(
        "Action counts (charge/idle/discharge): " *
        "$(summary.charge_steps)/$(summary.idle_steps)/$(summary.discharge_steps)",
    )

    mkpath(params.output_dir)

    timeline_file = joinpath(params.output_dir, "q_learning_$(params.test_years[1])_timeline.csv")
    CSV.write(timeline_file, timeline)
    println("Saved timeline: $(timeline_file)")

    episode_log = DataFrame(
        episode = collect(1:params.episodes),
        epsilon = model.episode_epsilon,
        alpha = model.episode_alpha,
        train_profit = model.episode_profit,
        train_reward = model.episode_reward,
        train_cycles_used = model.episode_cycles,
    )
    training_log_file = joinpath(params.output_dir, "training_episode_log.csv")
    try
        CSV.write(training_log_file, episode_log)
    catch
        training_log_file = joinpath(params.output_dir, "training_episode_log_new.csv")
        CSV.write(training_log_file, episode_log)
    end
    println("Saved training log: $(training_log_file)")

    return (; params, model, summary, timeline, episode_log, training_log_file)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_pipeline()
end
