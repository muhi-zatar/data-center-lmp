"""
Usage:
    python run_two_settlement.py [config.yaml] [start_day] [n_days]
"""

import sys
import os
import pandas as pd
import numpy as np
import pypsa

from config_loader import load_config
from data_loader import (
    load_matpower_case,
    load_time_series,
    load_forecast_time_series,
    load_gen_data,
    get_generator_indices
)
from initial_conditions import (
    calculate_initial_conditions,
    load_initial_conditions_from_file,
    InitialConditions
)
from network_builder import create_network
from solver import solve_day, SolverResults


def solve_real_time(
    network: pypsa.Network,
    config,
    day: int,
    da_commitment: pd.DataFrame
) -> tuple:
    """
    Solve real-time market with fixed day-ahead commitment.
    
    Args:
        network: PyPSA network built with ACTUAL data
        config: Configuration object
        day: Day index
        da_commitment: Commitment status from DA market
    
    Returns:
        Tuple of (rt_cost, rt_lmps, rt_dispatch)
    """
    start_idx = day * 24
    end_idx = start_idx + 24
    day_snapshots = network.snapshots[start_idx:end_idx]
    
    if 'RT_Slack_Up' not in network.generators.index:
        # Find the data center bus
        dc_loads = [l for l in network.loads.index if 'DataCenter' in l]
        if dc_loads:
            dc_bus = network.loads.loc[dc_loads[0], 'bus']
        else:
            dc_bus = network.buses.index[0]  # fallback
        
        network.add(
            "Generator",
            name="RT_Slack_Up",
            bus=dc_bus,
            p_nom=99999,
            marginal_cost=9999,
            carrier="slack"
        )
        network.add(
            "Generator",
            name="RT_Slack_Down",
            bus=dc_bus,
            p_nom=99999,
            p_max_pu=0,
            p_min_pu=-1,
            marginal_cost=9999,
            carrier="slack"
        )

    committable_gens = network.generators[network.generators['committable']].index.tolist()
    
    network.generators['ramp_limit_up'] = 1.0
    network.generators['ramp_limit_down'] = 1.0
    
    for gen in committable_gens:
        if gen in da_commitment.columns:
            status_series = da_commitment[gen]
            network.generators_t.p_max_pu.loc[day_snapshots, gen] = status_series
            original_p_min_pu = network.generators.loc[gen, 'p_min_pu']
            network.generators_t.p_min_pu.loc[day_snapshots, gen] = status_series * original_p_min_pu
    
    network.generators.loc[committable_gens, 'committable'] = False
    
    network.optimize(
        snapshots=day_snapshots,
        solver_name=config.solver.name,
        solver_options={'time_limit': config.solver.lp.time_limit}
    )
    
    rt_cost = network.objective
    rt_lmps = network.buses_t.marginal_price.loc[day_snapshots].copy()
    rt_dispatch = network.generators_t.p.loc[day_snapshots].copy()
    
    return rt_cost, rt_lmps, rt_dispatch


def save_two_settlement_results(
    config,
    day: int,
    da_results: SolverResults,
    da_dispatch: pd.DataFrame,
    da_lmps: pd.DataFrame,
    rt_cost: float,
    rt_dispatch: pd.DataFrame,
    rt_lmps: pd.DataFrame,
    network_da: pypsa.Network,
    network_rt: pypsa.Network
) -> None:
    """Save results from both DA and RT markets."""
    output_dir = config.output.directory
    os.makedirs(output_dir, exist_ok=True)
    day_num = day + 1
    day_snapshots = da_results.day_snapshots
    
    da_dispatch.to_csv(f'{output_dir}/da_dispatch_day{day_num}.csv')
    da_lmps.to_csv(f'{output_dir}/da_lmps_day{day_num}.csv')
    
    rt_dispatch.to_csv(f'{output_dir}/rt_dispatch_day{day_num}.csv')
    rt_lmps.to_csv(f'{output_dir}/rt_lmps_day{day_num}.csv')
    
    da_results.commitment_status.to_csv(f'{output_dir}/commitment_day{day_num}.csv')
    
    da_load = network_da.loads_t.p_set.loc[day_snapshots].sum(axis=1)
    rt_load = network_rt.loads_t.p_set.loc[day_snapshots].sum(axis=1)
    
    summary = pd.DataFrame({
        'da_load_MW': da_load,
        'rt_load_MW': rt_load,
        'da_gen_MW': da_dispatch.sum(axis=1),
        'rt_gen_MW': rt_dispatch.sum(axis=1),
        'da_lmp_mean': da_lmps.mean(axis=1),
        'da_lmp_min': da_lmps.min(axis=1),
        'da_lmp_max': da_lmps.max(axis=1),
        'rt_lmp_mean': rt_lmps.mean(axis=1),
        'rt_lmp_min': rt_lmps.min(axis=1),
        'rt_lmp_max': rt_lmps.max(axis=1),
    })
    summary.to_csv(f'{output_dir}/summary_day{day_num}.csv')
    
    if hasattr(network_da, 'lines_t') and 'p0' in network_da.lines_t:
        da_line_flows = network_da.lines_t.p0.loc[day_snapshots].copy()
        da_line_flows.to_csv(f'{output_dir}/da_line_flows_day{day_num}.csv')
        
        # DA congestion (loading %)
        da_loading = da_line_flows.abs() / network_da.lines.s_nom
        da_loading.to_csv(f'{output_dir}/da_congestion_day{day_num}.csv')
    
    # RT line flows
    if hasattr(network_rt, 'lines_t') and 'p0' in network_rt.lines_t:
        rt_line_flows = network_rt.lines_t.p0.loc[day_snapshots].copy()
        rt_line_flows.to_csv(f'{output_dir}/rt_line_flows_day{day_num}.csv')
        
        # RT congestion (loading %)
        rt_loading = rt_line_flows.abs() / network_rt.lines.s_nom
        rt_loading.to_csv(f'{output_dir}/rt_congestion_day{day_num}.csv')
    # Price convergence analysis (DA vs RT by bus)
    price_diff = rt_lmps - da_lmps
    convergence = pd.DataFrame({
        'bus': da_lmps.columns,
        'da_lmp_avg': da_lmps.mean().values,
        'rt_lmp_avg': rt_lmps.mean().values,
        'diff_avg': price_diff.mean().values,
        'diff_abs_avg': price_diff.abs().mean().values,
        'diff_max': price_diff.abs().max().values,
    })
    convergence.to_csv(f'{output_dir}/price_convergence_day{day_num}.csv', index=False)
    

    # Final state for next day (from RT)
    _save_final_state_from_rt(output_dir, day_num, day_snapshots, 
                               da_results.commitment_status, rt_dispatch)
    

def _save_final_state_from_rt(
    output_dir: str,
    day_num: int,
    day_snapshots: pd.DatetimeIndex,
    commitment_status: pd.DataFrame,
    rt_dispatch: pd.DataFrame
) -> None:
    """Save final state from RT for next day's initial conditions."""
    final_hour = day_snapshots[-1]
    final_state = {}
    
    for gen in commitment_status.columns:
        status = commitment_status.loc[final_hour, gen]
        status_series = commitment_status[gen]
        
        if status == 1:
            hours_on = 0
            for t in reversed(day_snapshots):
                if status_series.loc[t] == 1:
                    hours_on += 1
                else:
                    break
            final_state[gen] = {'status': 1, 'up_time': hours_on, 'down_time': 0}
        else:
            hours_off = 0
            for t in reversed(day_snapshots):
                if status_series.loc[t] == 0:
                    hours_off += 1
                else:
                    break
            final_state[gen] = {'status': 0, 'up_time': 0, 'down_time': hours_off}
    
    state_df = pd.DataFrame(final_state).T
    state_df.to_csv(f'{output_dir}/final_state_day{day_num}.csv')


def print_market_comparison(
    da_lmps: pd.DataFrame,
    rt_lmps: pd.DataFrame,
    da_dispatch: pd.DataFrame,
    rt_dispatch: pd.DataFrame
) -> None:
    # For debugging and quick comparison
    # LMP comparison
    da_mean = da_lmps.mean().mean()
    rt_mean = rt_lmps.mean().mean()
    lmp_diff = rt_lmps - da_lmps
    
    print(f"\nLMP Statistics:")
    print(f"  {'':20s} {'DA':>12s} {'RT':>12s} {'Diff':>12s}")
    print(f"  {'Mean ($/MWh)':20s} {da_mean:>12.2f} {rt_mean:>12.2f} {rt_mean-da_mean:>12.2f}")
    print(f"  {'Min ($/MWh)':20s} {da_lmps.min().min():>12.2f} {rt_lmps.min().min():>12.2f}")
    print(f"  {'Max ($/MWh)':20s} {da_lmps.max().max():>12.2f} {rt_lmps.max().max():>12.2f}")
    print(f"  {'MAE ($/MWh)':20s} {lmp_diff.abs().mean().mean():>12.2f}")
    
    # Dispatch comparison
    da_total = da_dispatch.sum().sum() / 1e3  # GWh
    rt_total = rt_dispatch.sum().sum() / 1e3
    
    print(f"\nTotal Generation (GWh):")
    print(f"  DA: {da_total:.2f}")
    print(f"  RT: {rt_total:.2f}")
    print(f"  Diff: {rt_total - da_total:.2f}")


def run_two_settlement(
    config_path: str = "config.yaml",
    start_day: int = 140,
    n_days: int = 7
):
    """
    Run two-settlement market simulation.
    
    Args:
        config_path: Path to config file
        start_day: First day (0-indexed)
        n_days: Number of days to run
    """
    config = load_config(config_path)
    days = list(range(start_day, start_day + n_days))
    
    # Validate forecast files exist
    if not config.data.nodal_load_forecast_file:
        raise ValueError("Forecast files not specified in config.")
    
    
    case_data = load_matpower_case(config.data.case_file)
    
    time_series_actual = load_time_series(config.data, config.simulation.n_hours)
    time_series_forecast = load_forecast_time_series(
        config.data, config.simulation.n_hours, time_series_actual
    )
    
    gen_data = load_gen_data(config.data)
    gen_indices = get_generator_indices(case_data.genfuel)
    
    
    # Run each day
    for i, day in enumerate(days):
        print(f"\n{'#'*60}")
        print(f"DAY {day + 1}")
        print(f"{'#'*60}")
        
        config.simulation.run_day = day
        
        # Initial conditions (from previous RT state)
        if i == 0:
            # First day - calculate from scratch using forecast
            init_cond = calculate_initial_conditions(
                config, case_data, time_series_forecast, gen_data, gen_indices
            )
            initial_conditions = {
                'status': init_cond.status,
                'up_time': init_cond.up_time,
                'down_time': init_cond.down_time
            }
        else:
            # Load from previous RT final state
            prev_day = days[i - 1]
            prev_state = load_initial_conditions_from_file(
                f"{config.output.directory}/final_state_day{prev_day + 1}.csv"
            )
            initial_conditions = {
                'status': prev_state.status,
                'up_time': prev_state.up_time,
                'down_time': prev_state.down_time
            }
        
        # DAY-AHEAD MARKET (using forecasts)
        
        network_da = create_network(
            config, case_data, time_series_forecast, gen_data, gen_indices, 
            initial_conditions
        )
        
        da_results = solve_day(network_da, config, day)
        
        # Get DA dispatch and LMPs
        day_snapshots = da_results.day_snapshots
        da_dispatch = network_da.generators_t.p.loc[day_snapshots].copy()
        da_lmps = network_da.buses_t.marginal_price.loc[day_snapshots].copy()
        
        # REAL-TIME MARKET (using actuals, fixed commitment)
        
        network_rt = create_network(
            config, case_data, time_series_actual, gen_data, gen_indices,
            initial_conditions
        )
        
        rt_cost, rt_lmps, rt_dispatch = solve_real_time(
            network_rt, config, day, da_results.commitment_status
        )
        
        print_market_comparison(da_lmps, rt_lmps, da_dispatch, rt_dispatch)
        
        save_two_settlement_results(
            config, day, da_results,
            da_dispatch, da_lmps,
            rt_cost, rt_dispatch, rt_lmps,
            network_da, network_rt
        )
    


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    start_day = int(sys.argv[2]) if len(sys.argv) > 2 else 140
    n_days = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    
    run_two_settlement(config_path, start_day, n_days)
