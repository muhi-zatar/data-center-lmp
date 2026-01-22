"""
Results processing and saving module for ERCOT LMP calculation.
"""

import os
import pandas as pd
import pypsa
from typing import Dict

from config_loader import Config
from solver import SolverResults


def process_and_save_results(
    network: pypsa.Network,
    config: Config,
    solver_results: SolverResults
) -> None:
    """
    Process optimization results and save to files.
    
    Args:
        network: PyPSA network (after optimization)
        config: Configuration object
        solver_results: Results from solver
    """
    day = config.simulation.run_day
    day_snapshots = solver_results.day_snapshots
    
    # Create output directory
    os.makedirs(config.output.directory, exist_ok=True)
    
    # Extract results
    gen_p = network.generators_t.p.loc[day_snapshots]
    lmps = network.buses_t.marginal_price.loc[day_snapshots]
    line_flows = network.lines_t.p0.loc[day_snapshots]
    hourly_load = network.loads_t.p_set.loc[day_snapshots].sum(axis=1)
    hourly_gen = gen_p.sum(axis=1)
    network.storage_units_t.p_dispatch  # Discharging
    network.storage_units_t.p_store     # Charging
    network.storage_units_t.state_of_charge
    
    # Print summary
    _print_results_summary(
        config, solver_results, gen_p, lmps, hourly_load, hourly_gen
    )
    
    # Validate
    _validate_results(hourly_load, hourly_gen)
    
    # Print hourly details
    _print_hourly_details(day_snapshots, hourly_load, hourly_gen, lmps)
    
    # Save files
    _save_results(
        config, day, day_snapshots, gen_p, lmps,
        solver_results.commitment_status, line_flows,
        hourly_load, hourly_gen, network
    )


def _print_results_summary(
    config: Config,
    solver_results: SolverResults,
    gen_p: pd.DataFrame,
    lmps: pd.DataFrame,
    hourly_load: pd.Series,
    hourly_gen: pd.Series
) -> None:
    
    
    solar_gen = gen_p[[c for c in gen_p.columns if 'Solar' in c]].sum().sum() / 1e3
    wind_gen = gen_p[[c for c in gen_p.columns if 'Wind' in c]].sum().sum() / 1e3
    hydro_gen = gen_p[[c for c in gen_p.columns if 'Hydro' in c]].sum().sum() / 1e3
    
    # sanity checks
    print(f"  Solar:   {solar_gen:.2f}")
    print(f"  Wind:    {wind_gen:.2f}")
    print(f"  Hydro:   {hydro_gen:.2f}")
    
    for fuel in config.unit_commitment.committable_fuels:
        fuel_gen = gen_p[[c for c in gen_p.columns if fuel in c]].sum().sum() / 1e3
        print(f"  {fuel:8s}: {fuel_gen:.2f}")
    
    # LMP statistics as sanity checks to check quickly if results look reasonable
    # Check slack usage
    slack_up = gen_p['Slack_Up'].sum() if 'Slack_Up' in gen_p.columns else 0
    slack_down = gen_p['Slack_Down'].abs().sum() if 'Slack_Down' in gen_p.columns else 0

    if slack_up > 0.1:
        print(f"\n  WARNING: Slack_Up used {slack_up:.1f} MWh - generation shortage!")
    if slack_down > 0.1:
        print(f"\n  WARNING: Slack_Down used {slack_down:.1f} MWh - excess generation!")

    print(f"\nLMPs ($/MWh):")
    print(f"  Mean: {lmps.mean().mean():.2f}")
    print(f"  Min:  {lmps.min().min():.2f}")
    print(f"  Max:  {lmps.max().max():.2f}")
    print(f"  Std:  {lmps.std().mean():.2f}")
    
    # Unit commitment summary
    _print_uc_summary(config, solver_results.commitment_status)
    
    # Ramp analysis
    _print_ramp_analysis(config, gen_p)


def _print_uc_summary(config: Config, commitment_status: pd.DataFrame) -> None:
    """Print unit commitment summary by fuel type."""
    
    committable_gens = commitment_status.columns.tolist()
    
    for fuel in config.unit_commitment.committable_fuels:
        fuel_gens = [g for g in committable_gens if fuel in g]
        if fuel_gens:
            online_hours = commitment_status[fuel_gens].sum().sum()
            total_hours = len(fuel_gens) * 24
            startups = (commitment_status[fuel_gens].diff().fillna(0) > 0).sum().sum()
            print(f"  {fuel}: {len(fuel_gens)} units, {online_hours:.0f}/{total_hours} unit-hours, {startups:.0f} startups")


def _print_ramp_analysis(config: Config, gen_p: pd.DataFrame) -> None:
    """Print ramp rate analysis by fuel type."""
    
    for fuel in config.unit_commitment.committable_fuels:
        fuel_cols = [c for c in gen_p.columns if fuel in c]
        if fuel_cols:
            fuel_p = gen_p[fuel_cols]
            ramps = fuel_p.diff().abs()
            max_ramp = ramps.max().max()
            avg_ramp = ramps.mean().mean()
            print(f"  {fuel}: max ramp = {max_ramp:.1f} MW/h, avg ramp = {avg_ramp:.1f} MW/h")


def _validate_results(hourly_load: pd.Series, hourly_gen: pd.Series) -> None:
    """Validate load balance."""
    imbalance = (hourly_gen - hourly_load).abs()
    print(f"  Max imbalance: {imbalance.max():.2f} MW")
    print(f"  Avg imbalance: {imbalance.mean():.2f} MW")


def _print_hourly_details(
    day_snapshots: pd.DatetimeIndex,
    hourly_load: pd.Series,
    hourly_gen: pd.Series,
    lmps: pd.DataFrame
) -> None:
    """Print hourly load, generation, and LMP details."""
    
    for i, snap in enumerate(day_snapshots):
        avg_lmp = lmps.loc[snap].mean()
        min_lmp = lmps.loc[snap].min()
        max_lmp = lmps.loc[snap].max()
        print(f"  {i:<6} {hourly_load.loc[snap]:>10.0f} {hourly_gen.loc[snap]:>10.0f} "
              f"{avg_lmp:>10.2f} {min_lmp:>10.2f} {max_lmp:>10.2f}")


def _save_results(
    config: Config,
    day: int,
    day_snapshots: pd.DatetimeIndex,
    gen_p: pd.DataFrame,
    lmps: pd.DataFrame,
    commitment_status: pd.DataFrame,
    line_flows: pd.DataFrame,
    hourly_load: pd.Series,
    hourly_gen: pd.Series,
    network: pypsa.Network
) -> None:
    """Save all result files."""
    output_dir = config.output.directory
    day_num = day + 1
    
    if config.output.save_dispatch:
        # Remove slack from dispatch output
        gen_p_clean = gen_p[[c for c in gen_p.columns if 'Slack' not in c]]
        gen_p_clean.to_csv(f'{output_dir}/dispatch_day{day_num}.csv')
        # gen_p.to_csv(f'{output_dir}/dispatch_day{day_num}.csv')
    
    if config.output.save_lmps:
        lmps.to_csv(f'{output_dir}/lmps_day{day_num}.csv')
    
    if config.output.save_commitment:
        commitment_status.to_csv(f'{output_dir}/commitment_day{day_num}.csv')
    
    if config.output.save_line_flows:
        line_flows.to_csv(f'{output_dir}/line_flows_day{day_num}.csv')
    
    if config.output.save_summary:
        summary = pd.DataFrame({
            'load_MW': hourly_load,
            'generation_MW': hourly_gen,
            'lmp_mean': lmps.mean(axis=1),
            'lmp_min': lmps.min(axis=1),
            'lmp_max': lmps.max(axis=1),
        })
        summary.to_csv(f'{output_dir}/summary_day{day_num}.csv')
    
    if config.output.save_congestion:
        line_limits = network.lines['s_nom']
        congestion = pd.DataFrame({
            'line': line_flows.columns,
            'limit_MW': [line_limits[l] for l in line_flows.columns],
            'max_flow_MW': line_flows.abs().max().values,
            'utilization_pct': (line_flows.abs().max().values / line_limits.values) * 100
        })
        congestion = congestion.sort_values('utilization_pct', ascending=False)
        congestion.to_csv(f'{output_dir}/congestion_day{day_num}.csv', index=False)
    
    if config.output.save_ramp_analysis:
        ramp_analysis = pd.DataFrame()
        for fuel in config.unit_commitment.committable_fuels:
            fuel_cols = [c for c in gen_p.columns if fuel in c]
            if fuel_cols:
                fuel_ramps = gen_p[fuel_cols].diff()
                ramp_analysis[f'{fuel}_total_ramp'] = fuel_ramps.sum(axis=1)
        ramp_analysis.to_csv(f'{output_dir}/ramp_analysis_day{day_num}.csv')
    
    if config.output.save_final_state:
        _save_final_state(output_dir, day_num, day_snapshots, commitment_status)

def _save_final_state(
    output_dir: str,
    day_num: int,
    day_snapshots: pd.DatetimeIndex,
    commitment_status: pd.DataFrame
) -> None:
    """Save final state for rolling horizon (next day's initial conditions)."""
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