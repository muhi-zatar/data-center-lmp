"""
Usage:
    python main.py [config.yaml]
"""

import sys
from pathlib import Path

from config_loader import load_config
from data_loader import (
    load_matpower_case,
    load_time_series,
    load_gen_data,
    get_generator_indices
)
from initial_conditions import calculate_initial_conditions
from network_builder import create_network
from solver import solve_day
from results import process_and_save_results
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main(config_path: str = "config.yaml") -> None:
    
    config = load_config(config_path)
    
    case_data = load_matpower_case(config.data.case_file)
    
    time_series = load_time_series(config.data, config.simulation.n_hours)
    # Sanity check for time series shapes
    print(f"  Load: {time_series.nodal_load.shape}")
    print(f"  Solar: {time_series.solar_mw.shape}")
    print(f"  Wind: {time_series.wind_mw.shape}")
    print(f"  Hydro: {time_series.hydro_mw.shape}")
    
    gen_data = load_gen_data(config.data)
    
    # Get generator indices by fuel type - this will help in adding a generators to the network
    gen_indices = get_generator_indices(case_data.genfuel)
    
    # Calculate initial conditions - This includes some heuristics like 80% of minimum load
    initial_conditions = None
    if config.unit_commitment.use_initial_conditions:
        init_cond = calculate_initial_conditions(
            config, case_data, time_series, gen_data, gen_indices
        )
        initial_conditions = {
            'status': init_cond.status,
            'up_time': init_cond.up_time,
            'down_time': init_cond.down_time
        }

    network = create_network(
        config, case_data, time_series, gen_data, gen_indices, initial_conditions
    )
    
    solver_results = solve_day(network, config, config.simulation.run_day)
    
    process_and_save_results(network, config, solver_results)


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_file)