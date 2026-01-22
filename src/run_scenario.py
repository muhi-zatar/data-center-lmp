"""
Usage:
    python run_scenario.py scenario_config.yaml [base_config.yaml]
"""

import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

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
    load_initial_conditions_from_file
)
from network_builder import create_network
from solver import solve_day
from results import process_and_save_results
from run_two_settlement import solve_real_time, save_two_settlement_results, print_market_comparison


def load_scenario_config(path: str) -> dict:
    """Load scenario configuration from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def apply_scenario(network, scenario: dict, snapshots: pd.DatetimeIndex) -> None:
    """
    Apply scenario modifications to network.
    
    Args:
        network: PyPSA network
        scenario: Scenario configuration dict
        snapshots: Network snapshots
    """
    
    # Add loads
    if 'add_loads' in scenario and scenario['add_loads']:
        _add_scenario_loads(network, scenario['add_loads'], snapshots)
    
    # Add generators
    if 'add_generators' in scenario and scenario['add_generators']:
        _add_scenario_generators(network, scenario['add_generators'], snapshots)
    
    # Add lines
    if 'add_lines' in scenario and scenario['add_lines']:
        _add_scenario_lines(network, scenario['add_lines'])

    # Add storage        
    if 'add_storage' in scenario and scenario['add_storage']:
        _add_scenario_storage(network, scenario['add_storage'])
        
    # Modify generators
    if 'modify_generators' in scenario and scenario['modify_generators']:
        _modify_generators(network, scenario['modify_generators'])
    
    # Modify lines
    if 'modify_lines' in scenario and scenario['modify_lines']:
        _modify_lines(network, scenario['modify_lines'])
    
    # Disable components
    if 'disable' in scenario and scenario['disable']:
        _disable_components(network, scenario['disable'])


def _add_scenario_loads(network, loads: list, snapshots: pd.DatetimeIndex) -> None:
    """Add new loads to network."""

    print(f"loads: {loads}")
    for load in loads:
        bus_name = f"Bus_{load['bus']}"
        
        # Check bus exists
        if bus_name not in network.buses.index:
            print(f"  WARNING: Bus {bus_name} not found, skipping load {load['name']}")
            continue
        
        # Create load profile
        if 'profile_file' in load:
            profile = pd.read_csv(load['profile_file'], index_col=0).squeeze()
            scale = load.get('profile_scale', 1.0)
            
            # Get correct hours based on simulation day
            start_day = snapshots[0].dayofyear
            start_hour = (start_day - 1) * 24
            end_hour = start_hour + len(snapshots)
            p_set = profile.values[start_hour:end_hour] * scale
        else:
            p_set = load['p_mw']  # Constant load
        
        network.add(
            "Load",
            name=load['name'],
            bus=bus_name,
            p_set=p_set
        )
        
        p_val = load.get('p_mw', 'profile')
        print(f"  Added load: {load['name']} at {bus_name}, {p_val} MW")


def _add_scenario_generators(network, generators: list, snapshots: pd.DatetimeIndex) -> None:
    """Add new generators to network."""
    print(f"Adding {len(generators)} generators...")
    for gen in generators:
        print(f"  Processing: {gen['name']}")
        bus_name = f"Bus_{gen['bus']}"
        
        if bus_name not in network.buses.index:
            print(f"  WARNING: Bus {bus_name} not found, skipping generator {gen['name']}")
            continue
        
        print(f"  Bus {bus_name} exists: {bus_name in network.buses.index}")  # 
        print(f"  gen_params will be: {gen}")  # 
        
        gen_params = {
            'name': gen['name'],
            'bus': bus_name,
            'p_nom': gen['p_nom'],
            'carrier': gen.get('carrier', 'unknown'),
            'marginal_cost': gen.get('marginal_cost', 0),
        }
        
        # Optional parameters
        if 'p_min_pu' in gen:
            gen_params['p_min_pu'] = gen['p_min_pu']

        if 'p_max_pu' in gen and 'profile_file' not in gen:
            gen_params['p_max_pu'] = gen['p_max_pu']
        
        if 'committable' in gen and gen['committable']:
            gen_params['committable'] = True
            gen_params['min_up_time'] = gen.get('min_up_time', 1)
            gen_params['min_down_time'] = gen.get('min_down_time', 1)
            gen_params['start_up_cost'] = gen.get('start_up_cost', 0)
            gen_params['shut_down_cost'] = gen.get('shut_down_cost', 0)
        
        if 'ramp_limit_up' in gen:
            gen_params['ramp_limit_up'] = gen['ramp_limit_up']
            gen_params['ramp_limit_down'] = gen.get('ramp_limit_down', gen['ramp_limit_up'])
            # Startup ramps
            p_min_pu = gen.get('p_min_pu', 0)
            gen_params['ramp_limit_start_up'] = max(p_min_pu, gen['ramp_limit_up'])
            gen_params['ramp_limit_shut_down'] = max(p_min_pu, gen['ramp_limit_up'])
        
        if 'profile_file' in gen:
            profile = pd.read_csv(gen['profile_file'], index_col=0).squeeze()
            scale = gen.get('profile_scale', 1.0)
            
            # Get correct hours based on simulation day
            start_day = snapshots[0].dayofyear
            start_hour = (start_day - 1) * 24
            end_hour = start_hour + len(snapshots)
            p_set = profile.values[start_hour:end_hour] * scale
        
        network.add("Generator", **gen_params)
        print(f"  Calling network.add with: {gen_params}")  # 
        print(f"  Generator in network? {gen['name'] in network.generators.index}")
        


def _add_scenario_lines(network, lines: list) -> None:
    """Add new lines to network."""
    for line in lines:
        bus0_name = f"Bus_{line['bus0']}"
        bus1_name = f"Bus_{line['bus1']}"
        
        if bus0_name not in network.buses.index or bus1_name not in network.buses.index:
            print(f"  WARNING: Bus not found, skipping line {line['name']}")
            continue
        
        network.add(
            "Line",
            name=line['name'],
            bus0=bus0_name,
            bus1=bus1_name,
            s_nom=line['s_nom'],
            x=line.get('x', 0.01),
            r=line.get('r', 0.001),
            carrier="AC"
        )
        
        print(f"  Added line: {line['name']} ({bus0_name} - {bus1_name}), {line['s_nom']} MW")

def _add_scenario_storage(network, storage_units: list) -> None:
    """Add storage units to network."""
    for storage in storage_units:
        bus_name = f"Bus_{storage['bus']}"
        
        if bus_name not in network.buses.index:
            print(f"  WARNING: Bus {bus_name} not found, skipping {storage['name']}")
            continue
        
        network.add(
            "StorageUnit",
            name=storage['name'],
            bus=bus_name,
            p_nom=storage['p_nom'],
            max_hours=storage.get('max_hours', 4),
            efficiency_store=storage.get('efficiency_store', 0.95),
            efficiency_dispatch=storage.get('efficiency_dispatch', 0.95),
            cyclic_state_of_charge=storage.get('cyclic_state_of_charge', True),
            marginal_cost=storage.get('marginal_cost', 0.5),
            carrier="battery"
        )

def _modify_generators(network, modifications: list) -> None:
    """Modify existing generators."""
    for mod in modifications:
        if 'name' in mod:
            # Exact match
            gens = [mod['name']] if mod['name'] in network.generators.index else []
        elif 'pattern' in mod:
            # Pattern match
            gens = [g for g in network.generators.index if mod['pattern'] in g]
        else:
            continue
        
        for gen in gens:
            if 'marginal_cost' in mod:
                network.generators.loc[gen, 'marginal_cost'] = mod['marginal_cost']
            if 'marginal_cost_scale' in mod:
                network.generators.loc[gen, 'marginal_cost'] *= mod['marginal_cost_scale']
            if 'marginal_cost_delta' in mod:
                network.generators.loc[gen, 'marginal_cost'] += mod['marginal_cost_delta']
            if 'p_nom' in mod:
                network.generators.loc[gen, 'p_nom'] = mod['p_nom']
        
        desc = mod.get('name', f"pattern:{mod.get('pattern', '?')}")
        print(f"  Modified {len(gens)} generators matching '{desc}'")


def _modify_lines(network, modifications: list) -> None:
    """Modify existing lines."""
    for mod in modifications:
        if 'name' in mod:
            lines = [mod['name']] if mod['name'] in network.lines.index else []
        elif 'pattern' in mod:
            lines = [l for l in network.lines.index if mod['pattern'] in l]
        else:
            continue
        
        for line in lines:
            if 's_nom' in mod:
                network.lines.loc[line, 's_nom'] = mod['s_nom']
            if 's_nom_scale' in mod:
                network.lines.loc[line, 's_nom'] *= mod['s_nom_scale']
        
        desc = mod.get('name', f"pattern:{mod.get('pattern', '?')}")
        print(f"  Modified {len(lines)} lines matching '{desc}'")


def _disable_components(network, disable: dict) -> None:
    """Disable generators or lines (set capacity to 0)."""
    if 'generators' in disable and disable['generators']:
        for gen in disable['generators']:
            if gen in network.generators.index:
                network.generators.loc[gen, 'p_nom'] = 0
                print(f"  Disabled generator: {gen}")
    
    if 'lines' in disable and disable['lines']:
        for line in disable['lines']:
            if line in network.lines.index:
                network.lines.loc[line, 's_nom'] = 0
                print(f"  Disabled line: {line}")


def run_scenario(scenario_path: str, base_config_path: str = "config.yaml"):
    """
    Run scenario simulation.
    
    Args:
        scenario_path: Path to scenario YAML file
        base_config_path: Path to base configuration YAML file
    """
    # Load configs
    config = load_config(base_config_path)
    scenario = load_scenario_config(scenario_path)
    
    # Update output directory with scenario name
    scenario_name = scenario.get('scenario', {}).get('name', 'scenario')
    config.output.directory = f"results_{scenario_name}"
    
    # Check if two-settlement mode
    two_settlement = scenario.get('scenario', {}).get('two_settlement', False)
    
    # Get days to run
    days = scenario.get('simulation', {}).get('days', [config.simulation.run_day])
    
    
    case_data = load_matpower_case(config.data.case_file)
    
    time_series_actual = load_time_series(config.data, config.simulation.n_hours)
    
    if two_settlement:
        time_series_forecast = load_forecast_time_series(
            config.data, config.simulation.n_hours, time_series_actual
        )
    
    gen_data = load_gen_data(config.data)
    gen_indices = get_generator_indices(case_data.genfuel)
    
    # Run each day
    for i, day in enumerate(days):

        
        config.simulation.run_day = day
        
        # Initial conditions
        if i == 0:
            ts_for_init = time_series_forecast if two_settlement else time_series_actual
            init_cond = calculate_initial_conditions(
                config, case_data, ts_for_init, gen_data, gen_indices
            )
            initial_conditions = {
                'status': init_cond.status,
                'up_time': init_cond.up_time,
                'down_time': init_cond.down_time
            }
        else:
            prev_day = days[i - 1]
            prev_state = load_initial_conditions_from_file(
                f"{config.output.directory}/final_state_day{prev_day + 1}.csv"
            )
            initial_conditions = {
                'status': prev_state.status,
                'up_time': prev_state.up_time,
                'down_time': prev_state.down_time
            }
        
        if two_settlement:
            
            # DAY-AHEAD (with forecasts)
            network_da = create_network(
                config, case_data, time_series_forecast, gen_data, 
                gen_indices, initial_conditions
            )
            apply_scenario(network_da, scenario, network_da.snapshots)
            
            da_results = solve_day(network_da, config, day)
            day_snapshots = da_results.day_snapshots
            da_dispatch = network_da.generators_t.p.loc[day_snapshots].copy()
            da_lmps = network_da.buses_t.marginal_price.loc[day_snapshots].copy()
            
            # REAL-TIME (with actuals)
            network_rt = create_network(
                config, case_data, time_series_actual, gen_data,
                gen_indices, initial_conditions
            )
            apply_scenario(network_rt, scenario, network_rt.snapshots)
            
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
        else:
            network = create_network(
                config, case_data, time_series_actual, gen_data, 
                gen_indices, initial_conditions
            )
            apply_scenario(network, scenario, network.snapshots)
            
            solver_results = solve_day(network, config, day)
            process_and_save_results(network, config, solver_results)
    


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_scenario.py scenario_config.yaml [base_config.yaml]")
        sys.exit(1)
    
    scenario_path = sys.argv[1]
    base_config_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
    
    run_scenario(scenario_path, base_config_path)
