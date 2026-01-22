"""
Important note: to determine initial conditions for unit commitment, we used a heuristic of merit order, by sorting them from lower to higher marginal cost, and committing enough units to cover ~80% of minimum load, while ensuring that total Pmin of committed units does not exceed minimum load (to avoid over-generation at low load hours). This is a conservative approach to avoid infeasibilities.
"""
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from config_loader import Config
from data_loader import CaseData, TimeSeriesData, GenData, GeneratorIndices, extract_cost_from_gencost


@dataclass
class InitialConditions:
    """Container for initial unit commitment conditions."""
    status: Dict[str, int]       # gen_name -> 0 or 1
    up_time: Dict[str, int]      # gen_name -> hours already on
    down_time: Dict[str, int]    # gen_name -> hours already off


def calculate_initial_conditions(
    config: Config,
    case_data: CaseData,
    time_series: TimeSeriesData,
    gen_data: GenData,
    gen_indices: GeneratorIndices
) -> InitialConditions:
    """
    Calculate initial commitment status for generators.
    
    Uses conservative approach:
    - Only commit enough capacity to meet ~80% of minimum load
    - Ensures total Pmin doesn't exceed minimum load
    
    Args:
        config: Configuration object
        case_data: MATPOWER case data
        time_series: Time series data
        gen_data: Generator operational data
        gen_indices: Generator indices by fuel type
    
    Returns:
        InitialConditions object
    """
    run_day = config.simulation.run_day
    
    start_hour = run_day * 24
    end_hour = start_hour + 24
    day_load = time_series.nodal_load[start_hour:end_hour, :].sum(axis=1)
    min_load = day_load.min()
    avg_load = day_load.mean()
    max_load = day_load.max()

    gen_merit = _build_merit_order(
        config, case_data, gen_data, gen_indices
    )
    
    # Commit generators conservatively
    initial_status, up_time, down_time, stats = _commit_generators(
        gen_merit, min_load, gen_data
    )
        
    return InitialConditions(
        status=initial_status,
        up_time=up_time,
        down_time=down_time
    )


def _build_merit_order(
    config: Config,
    case_data: CaseData,
    gen_data: GenData,
    gen_indices: GeneratorIndices
) -> List[Dict]:
    """Build list of committable generators sorted by merit order."""
    gen_merit = []
    
    for i, (idx, row) in enumerate(case_data.gen_df.iterrows()):
        if i in gen_indices.renewable:
            continue
        
        fuel = case_data.genfuel[i].lower()
        if fuel not in config.unit_commitment.committable_fuels:
            continue
        
        p_max = float(row[case_data.pmax_col])
        p_min = float(row[case_data.pmin_col])
        
        if p_max <= 0:
            continue
        
        marginal_cost, _ = extract_cost_from_gencost(case_data.gencost, i, p_max)
        
        gen_name = f"Gen_{i}_{case_data.genfuel[i]}"
        gen_merit.append({
            'name': gen_name,
            'idx': i,
            'fuel': fuel,
            'p_max': p_max,
            'p_min': p_min,
            'marginal_cost': marginal_cost,
            'min_on': int(gen_data.min_on[i]),
            'min_off': int(gen_data.min_off[i])
        })
    
    # Sort by marginal cost (merit order)
    gen_merit.sort(key=lambda x: x['marginal_cost'])
    
    return gen_merit


def _commit_generators(
    gen_merit: List[Dict],
    min_load: float,
    gen_data: GenData
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Commit generators in merit order until target capacity is reached.
    
    Returns:
        Tuple of (status, up_time, down_time, stats)
    """
    initial_status = {}
    up_time = {}
    down_time = {}
    
    # Conservative target: 80% of minimum load
    target_capacity = min_load
    
    cumulative_capacity = 0
    cumulative_pmin = 0
    
    for gen in gen_merit:
        gen_name = gen['name']
        
        # Check if adding this unit's Pmin would exceed minimum load
        if cumulative_pmin + gen['p_min'] > min_load * 1.1:
            # Don't commit - would cause over-generation
            initial_status[gen_name] = 0
            up_time[gen_name] = 0
            down_time[gen_name] = max(gen['min_off'], 24)
        elif cumulative_capacity < target_capacity:
            # Commit this unit
            initial_status[gen_name] = 1
            up_time[gen_name] = max(gen['min_on'], 24)
            down_time[gen_name] = 0
            cumulative_capacity += gen['p_max']
            cumulative_pmin += gen['p_min']
        else:
            # Don't commit
            initial_status[gen_name] = 0
            up_time[gen_name] = 0
            down_time[gen_name] = max(gen['min_off'], 24)
    
    stats = {
        'target_capacity': target_capacity,
        'committed_pmax': cumulative_capacity,
        'committed_pmin': cumulative_pmin,
        'n_on': sum(initial_status.values()),
        'n_off': len(initial_status) - sum(initial_status.values()),
        'gen_merit': gen_merit,
        'initial_status': initial_status
    }
    
    return initial_status, up_time, down_time, stats


def load_initial_conditions_from_file(filepath: str) -> InitialConditions:
    """Load initial conditions from a previous day's final state file."""
    import pandas as pd
    
    df = pd.read_csv(filepath, index_col=0)
    
    return InitialConditions(
        status={name: int(row['status']) for name, row in df.iterrows()},
        up_time={name: int(row['up_time']) for name, row in df.iterrows()},
        down_time={name: int(row['down_time']) for name, row in df.iterrows()}
    )
