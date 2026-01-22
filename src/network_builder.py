"""
Had to write it this way for flexibility and scalability to when we need to add more generatorrs or load.
"""
import numpy as np
import pandas as pd
import pypsa
from typing import Dict, Optional

from config_loader import Config
from data_loader import (
    CaseData, TimeSeriesData, GenData, GeneratorIndices,
    extract_cost_from_gencost
)


def create_network(
    config: Config,
    case_data: CaseData,
    time_series: TimeSeriesData,
    gen_data: GenData,
    gen_indices: GeneratorIndices,
    initial_conditions: Optional[Dict] = None
) -> pypsa.Network:
    """
    Create PyPSA network with all components.
    
    Args:
        config: Configuration object
        case_data: MATPOWER case data
        time_series: Time series data
        gen_data: Generator operational data
        gen_indices: Generator indices by fuel type
        initial_conditions: Optional dict with 'status', 'up_time', 'down_time'
    
    Returns:
        Configured PyPSA network
    """
    network = pypsa.Network()
    
    snapshots = pd.date_range(
        config.simulation.start_date,
        periods=config.simulation.n_hours,
        freq='h'
    )
    network.set_snapshots(snapshots)
    
    # Add carriers
    # for carrier in ["AC", "ng", "coal", "nuclear", "solar", "wind", "hydro"]:
    # for carrier in ["AC", "ng", "coal", "nuclear", "solar", "wind", "hydro", "battery"]:
    for carrier in ["AC", "ng", "coal", "nuclear", "solar", "wind", "hydro", "battery", "slack"]:
        network.add("Carrier", carrier)
    
    # Add components
    _add_buses(network, case_data)
    _add_loads(network, case_data, time_series, snapshots)
    _add_conventional_generators(
        network, config, case_data, gen_data, gen_indices,
        snapshots, initial_conditions
    )
    _add_renewable_generators(
        network, case_data, time_series, gen_indices, snapshots
    )
    _add_lines(network, config, case_data)
    
    # Add slack generators for balancing and load shedding
    network.add(
        "Generator",
        name="Slack_Up",
        bus=network.buses.index[0],
        p_nom=99999,
        marginal_cost=9999,
        carrier="slack"
    )

    network.add(
        "Generator",
        name="Slack_Down",
        bus=network.buses.index[0],
        p_nom=99999,
        p_max_pu=0,
        p_min_pu=-1,
        marginal_cost=-9999,
        carrier="slack"
    )

    return network


def _add_buses(network: pypsa.Network, case_data: CaseData) -> None:
    """Add buses to network."""
    for idx, row in case_data.bus_df.iterrows():
        bus_id = int(row[case_data.bus_id_col])
        base_kv = row['baseKV'] if 'baseKV' in row else row.iloc[9]
        network.add("Bus", name=f"Bus_{bus_id}", v_nom=base_kv, carrier="AC")
    
    print(f"Added {len(network.buses)} buses")


def _add_loads(
    network: pypsa.Network,
    case_data: CaseData,
    time_series: TimeSeriesData,
    snapshots: pd.DatetimeIndex
) -> None:
    """Add loads to network."""
    all_bus_ids = case_data.bus_df[case_data.bus_id_col].astype(int).values
    
    for i, bus_id in enumerate(all_bus_ids):
        load_series = time_series.nodal_load[:, i]
        if load_series.max() > 0:
            network.add(
                "Load",
                name=f"Load_{bus_id}",
                bus=f"Bus_{bus_id}",
                p_set=pd.Series(load_series, index=snapshots)
            )
    
    print(f"Added {len(network.loads)} loads")


def _add_conventional_generators(
    network: pypsa.Network,
    config: Config,
    case_data: CaseData,
    gen_data: GenData,
    gen_indices: GeneratorIndices,
    snapshots: pd.DatetimeIndex,
    initial_conditions: Optional[Dict]
) -> None:
    """Add conventional (thermal) generators to network."""
    n_committable = 0
    stats = {fuel: {'count': 0, 'uc_count': 0, 'capacity': 0} 
             for fuel in config.unit_commitment.committable_fuels}
    
    for i, (idx, row) in enumerate(case_data.gen_df.iterrows()):
        if i in gen_indices.renewable:
            continue
        
        bus_id = int(row[case_data.gen_bus_col])
        p_max = float(row[case_data.pmax_col])
        p_min = float(row[case_data.pmin_col])
        fuel = case_data.genfuel[i].lower()
        gen_name = f"Gen_{i}_{case_data.genfuel[i]}"
        
        if p_max <= 0:
            continue
        
        # Get costs from gencost
        marginal_cost, startup_cost = extract_cost_from_gencost(
            case_data.gencost, i, p_max
        )
        
        # Override startup cost if in config
        if fuel in config.costs.startup:
            startup_cost = startup_cost if startup_cost > 0 else config.costs.startup[fuel]
        
        # Get ramp rate
        ramp_limit = _get_ramp_limit(config, gen_data, i, p_max, fuel)
        
        # Track stats
        if fuel in stats:
            stats[fuel]['count'] += 1
            stats[fuel]['capacity'] += p_max
        
        # Check if committable
        is_committable = (
            fuel in config.unit_commitment.committable_fuels and
            p_max >= config.unit_commitment.min_size_for_uc
        )
        
        if is_committable:
            n_committable += 1
            if fuel in stats:
                stats[fuel]['uc_count'] += 1
            
            _add_committable_generator(
                network, config, gen_name, bus_id, p_max, p_min,
                marginal_cost, startup_cost, case_data.genfuel[i],
                gen_data, i, ramp_limit, initial_conditions
            )
        else:
            _add_non_committable_generator(
                network, config, gen_name, bus_id, p_max,
                marginal_cost, case_data.genfuel[i], ramp_limit
            )
    
    print(f"\nAdded conventional generators:")
    print(f"  - {n_committable} with unit commitment")
    print(f"  - Ramp constraints: {'ENABLED' if config.ramp_rates.enabled else 'DISABLED'}")
    print(f"  - Initial conditions: {'ENABLED' if config.unit_commitment.use_initial_conditions else 'DISABLED'}")
    
    print("\nGenerator summary:")
    for fuel, s in stats.items():
        if s['count'] > 0:
            print(f"  {fuel:8s}: {s['count']:3d} units, {s['uc_count']:3d} UC, {s['capacity']:8.0f} MW")


def _get_ramp_limit(
    config: Config,
    gen_data: GenData,
    gen_idx: int,
    p_max: float,
    fuel: str
) -> float:
    """Calculate ramp limit for a generator."""
    if not config.ramp_rates.enabled:
        return 1.0
    
    if gen_data.ramp_rates is not None:
        ramp_limit = gen_data.ramp_rates[gen_idx] / p_max if p_max > 0 else 1.0
    else:
        ramp_limit = config.ramp_rates.typical.get(fuel, 0.5)
    
    ramp_limit = min(ramp_limit * config.ramp_rates.multiplier, 1.0)
    return ramp_limit


def _add_committable_generator(
    network: pypsa.Network,
    config: Config,
    gen_name: str,
    bus_id: int,
    p_max: float,
    p_min: float,
    marginal_cost: float,
    startup_cost: float,
    carrier: str,
    gen_data: GenData,
    gen_idx: int,
    ramp_limit: float,
    initial_conditions: Optional[Dict]
) -> None:
    """Add a committable generator with UC constraints."""
    p_min_pu = p_min / p_max if p_max > 0 else 0
    
    gen_params = {
        'name': gen_name,
        'bus': f"Bus_{bus_id}",
        'p_nom': p_max,
        'p_min_pu': p_min_pu,
        'marginal_cost': marginal_cost,
        'carrier': carrier,
        'committable': True,
        'min_up_time': int(gen_data.min_on[gen_idx]),
        'min_down_time': int(gen_data.min_off[gen_idx]),
        'start_up_cost': startup_cost,
        'shut_down_cost': 0,
    }
    
    # Add ramp constraints
    if config.ramp_rates.enabled:
        gen_params['ramp_limit_up'] = ramp_limit
        gen_params['ramp_limit_down'] = ramp_limit
        gen_params['ramp_limit_start_up'] = max(p_min_pu, ramp_limit)
        gen_params['ramp_limit_shut_down'] = max(p_min_pu, ramp_limit)
    
    # Add initial conditions
    if config.unit_commitment.use_initial_conditions and initial_conditions:
        if gen_name in initial_conditions['status']:
            gen_params['initial_status'] = initial_conditions['status'][gen_name]
            gen_params['up_time_before'] = initial_conditions['up_time'].get(gen_name, 0)
            gen_params['down_time_before'] = initial_conditions['down_time'].get(gen_name, 0)
    
    network.add("Generator", **gen_params)


def _add_non_committable_generator(
    network: pypsa.Network,
    config: Config,
    gen_name: str,
    bus_id: int,
    p_max: float,
    marginal_cost: float,
    carrier: str,
    ramp_limit: float
) -> None:
    """Add a non-committable generator."""
    gen_params = {
        'name': gen_name,
        'bus': f"Bus_{bus_id}",
        'p_nom': p_max,
        'p_min_pu': 0,
        'marginal_cost': marginal_cost,
        'carrier': carrier,
    }
    
    if config.ramp_rates.enabled:
        gen_params['ramp_limit_up'] = ramp_limit
        gen_params['ramp_limit_down'] = ramp_limit
    
    network.add("Generator", **gen_params)


def _add_renewable_generators(
    network: pypsa.Network,
    case_data: CaseData,
    time_series: TimeSeriesData,
    gen_indices: GeneratorIndices,
    snapshots: pd.DatetimeIndex
) -> None:
    """Add solar, wind, and hydro generators."""
    
    # Solar
    for i, gen_idx in enumerate(gen_indices.solar):
        row = case_data.gen_df.iloc[gen_idx]
        bus_id = int(row[case_data.gen_bus_col])
        profile = time_series.solar_mw[:, i]
        p_nom = profile.max()
        
        if p_nom > 0:
            network.add(
                "Generator",
                name=f"Solar_{gen_idx}",
                bus=f"Bus_{bus_id}",
                p_nom=p_nom,
                p_max_pu=pd.Series(profile / p_nom, index=snapshots),
                marginal_cost=0,
                carrier="solar"
            )
    
    # Wind
    for i, gen_idx in enumerate(gen_indices.wind):
        row = case_data.gen_df.iloc[gen_idx]
        bus_id = int(row[case_data.gen_bus_col])
        profile = time_series.wind_mw[:, i]
        p_nom = profile.max()
        
        if p_nom > 0:
            network.add(
                "Generator",
                name=f"Wind_{gen_idx}",
                bus=f"Bus_{bus_id}",
                p_nom=p_nom,
                p_max_pu=pd.Series(profile / p_nom, index=snapshots),
                marginal_cost=0,
                carrier="wind"
            )
    
    # Hydro
    for i, gen_idx in enumerate(gen_indices.hydro):
        row = case_data.gen_df.iloc[gen_idx]
        bus_id = int(row[case_data.gen_bus_col])
        profile = time_series.hydro_mw[:, i]
        p_nom = profile.max()
        
        if p_nom > 0:
            network.add(
                "Generator",
                name=f"Hydro_{gen_idx}",
                bus=f"Bus_{bus_id}",
                p_nom=p_nom,
                p_max_pu=pd.Series(profile / p_nom, index=snapshots),
                marginal_cost=0,
                carrier="hydro"
            )


def _add_lines(
    network: pypsa.Network,
    config: Config,
    case_data: CaseData
) -> None:
    """Add transmission lines to network."""
    for idx, row in case_data.branch_df.iterrows():
        from_bus = int(row[case_data.fbus_col])
        to_bus = int(row[case_data.tbus_col])
        rate_a = float(row[case_data.rate_col])
        
        if config.transmission.relax_line_limits:
            s_nom = rate_a * 1.1
        else:
            s_nom = rate_a if rate_a > 0 else 9999
        
        network.add(
            "Line",
            name=f"Line_{idx}",
            bus0=f"Bus_{from_bus}",
            bus1=f"Bus_{to_bus}",
            r=float(row[case_data.r_col]),
            x=float(row[case_data.x_col]),
            s_nom=s_nom,
            carrier="AC"
        )
    
    limit_status = "RELAXED" if config.transmission.relax_line_limits else "ACTUAL"