#!/usr/bin/env python3
"""
Generate scenario configurations from siting data - Combined Multi-Location BTM version.

In this model:
- ALL locations from a single experiment are combined into ONE scenario
- Solar/wind are behind-the-meter (subtracted from DC load)
- Each location has its own net load profile

Usage:
    python generate_siting_scenario_combined_btm.py <siting_csv> [--periods winter,summer]
    
Examples:
    python generate_siting_scenario_combined_btm.py ../data/siting_data/100MW_onlyren_nameplate_capacities.csv
"""

import sys
import os
import re
import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


MARGINAL_COSTS = {
    'smr': 10,
    'gas_turbine': 50,
    'gas': 50,
    'diesel': 100,
    'geothermal': 5,
}

SIMULATION_PERIODS = {
    'winter': {
        'name': 'Winter',
        'description': 'February 5-19',
        'days': list(range(35, 50))
    },
    'spring': {
        'name': 'Spring', 
        'description': 'April 14-28',
        'days': list(range(104, 119))
    },
    'summer': {
        'name': 'Summer',
        'description': 'July 20 - August 3',
        'days': list(range(201, 216))
    },
    'fall': {
        'name': 'Fall',
        'description': 'November 1-14',
        'days': list(range(305, 319))
    }
}

DEFAULT_PERIODS = ['winter', 'spring', 'summer', 'fall']

GEN_COUNTER = 0


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_county_info(data_dir: str) -> pd.DataFrame:
    county_info_path = os.path.join(data_dir, 'mapping_data', 'county_info.csv')
    df = pd.read_csv(county_info_path, index_col=0)
    return df


def load_hourly_cf(data_dir: str) -> pd.DataFrame:
    cf_path = os.path.join(data_dir, 'siting_data', 'total_hourly_solar_wind_cf_tx.csv')
    df = pd.read_csv(cf_path)
    return df


def load_siting_data(siting_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(siting_csv_path)
    return df


# ============================================================================
# Parsing Functions
# ============================================================================

def parse_dc_size_from_expname(exp_name: str) -> float:
    match = re.search(r'(\d+)(MW|GW)', exp_name, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2).upper()
        if unit == 'GW':
            return value * 1000
        return value
    print(f"  WARNING: Could not parse DC size from '{exp_name}', defaulting to 100 MW")
    return 100.0


def parse_dc_size_from_filename(filename: str) -> float:
    basename = os.path.basename(filename)
    match = re.search(r'^(\d+)(MW|GW)', basename, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2).upper()
        if unit == 'GW':
            return value * 1000
        return value
    return None


def get_bus_from_county(county_fips: int, county_info: pd.DataFrame) -> Optional[int]:
    match = county_info[county_info['cfips'] == county_fips]
    if len(match) == 0:
        print(f"  WARNING: No bus found for county FIPS {county_fips}")
        return None
    return int(match['bestbus'].iloc[0])


def create_net_load_profile(
    hourly_cf: pd.DataFrame,
    location: int,
    dc_size: float,
    solar_capacity: float,
    wind_capacity: float,
    bus: int,
    output_dir: str
) -> str:
    """
    Create net load profile CSV for behind-the-meter solar/wind.
    
    Net_Load[hour] = DC_size - Solar_MW[hour] - Wind_MW[hour]
    """
    location_data = hourly_cf[hourly_cf['location'] == location].sort_values('hour')
    
    if len(location_data) == 0:
        print(f"      WARNING: No CF data for location {location}, using zeros for renewables")
        solar_cf = np.zeros(8760)
        wind_cf = np.zeros(8760)
    else:
        solar_cf = location_data['hourly_cf_solar'].values
        wind_cf = location_data['hourly_cf_wind'].values
        
        if len(solar_cf) != 8760:
            print(f"      WARNING: Expected 8760 hours, got {len(solar_cf)} for location {location}")
    
    solar_mw = solar_cf * solar_capacity
    wind_mw = wind_cf * wind_capacity
    
    # Net load: DC_load - renewable_generation
    net_load = dc_size - solar_mw - wind_mw
    
    profiles_dir = os.path.join(output_dir, 'profiles')
    os.makedirs(profiles_dir, exist_ok=True)
    
    filename = f'net_load_{dc_size:.0f}MW_{bus}_{location}.csv'
    filepath = os.path.join(profiles_dir, filename)
    
    profile_series = pd.Series(net_load, name='p_mw')
    profile_series.to_csv(filepath, index=True, header=True)
    
    print(f"      Net load: min={net_load.min():.1f}, max={net_load.max():.1f}, mean={net_load.mean():.1f} MW")
    
    return os.path.abspath(filepath)

def get_next_gen_name(prefix: str = "Gen", location: int = 0) -> str:
    global GEN_COUNTER
    name = f"DC_{prefix}_{location}_{GEN_COUNTER}"
    GEN_COUNTER += 1
    return name


def create_storage_unit(row: pd.Series, bus: int, location: int) -> Dict:
    capacity = row['capacity']
    
    if capacity <= 0:
        return None
    
    return {
        'name': get_next_gen_name('Battery', location),
        'bus': bus,
        'p_nom': capacity,
        'max_hours': 4,
        'efficiency_store': 0.95,
        'efficiency_dispatch': 0.95,
        'cyclic_state_of_charge': False,
        'marginal_cost': 0.5
    }


def create_dispatchable_generator(
    row: pd.Series,
    bus: int,
    gen_type: str,
    location: int
) -> Dict:
    capacity = row['capacity']
    
    if capacity <= 0:
        return None
    
    marginal_cost = MARGINAL_COSTS.get(gen_type, 50)
    
    carrier_map = {
        'smr': 'nuclear',
        'gas_turbine': 'ng',
        'gas': 'ng',
        'diesel': 'oil',
        'geothermal': 'geothermal'
    }
    carrier = carrier_map.get(gen_type, gen_type)
    
    gen_config = {
        'name': get_next_gen_name(gen_type.upper(), location),
        'bus': bus,
        'p_nom': capacity,
        'carrier': carrier,
        'marginal_cost': marginal_cost,
    }
    
    if gen_type in ['smr', 'gas_turbine', 'gas', 'diesel']:
        gen_config.update({
            'p_min_pu': 0.3,
            'committable': True,
            'min_up_time': 4,
            'min_down_time': 4,
            'start_up_cost': 2000,
            'ramp_limit_up': 0.5,
            'ramp_limit_down': 0.5
        })
    
    return gen_config


def create_combined_btm_scenario(
    siting_df: pd.DataFrame,
    exp_name: str,
    county_info: pd.DataFrame,
    hourly_cf: pd.DataFrame,
    dc_size: float,
    period_key: str,
    base_output_dir: str
) -> Optional[str]:
    """
    Create combined BTM scenario with ALL locations from one experiment.
    Solar/wind are behind-the-meter (subtracted from each DC's load).
    """
    global GEN_COUNTER
    GEN_COUNTER = 0
    
    period = SIMULATION_PERIODS[period_key]
    simulation_days = period['days']
    
    # Get all data for this experiment
    exp_df = siting_df[siting_df['exp_name'] == exp_name]
    
    if len(exp_df) == 0:
        print(f"  No data for exp_name={exp_name}")
        return None
    
    # Get unique locations
    locations = exp_df['location'].unique()
    
    # Create scenario directory
    safe_exp_name = re.sub(r'[^\w\-_]', '_', exp_name)
    scenario_name = f"{safe_exp_name}_{period_key}_combined_btm"
    scenario_dir = os.path.join(base_output_dir, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)
    
    print(f"\n  Creating COMBINED BTM scenario: {scenario_name}")
    print(f"    Experiment: {exp_name}")
    print(f"    Locations: {len(locations)} data centers")
    print(f"    Total DC load: {dc_size * len(locations):.0f} MW ({len(locations)} x {dc_size:.0f} MW)")
    print(f"    Period: {period['name']} ({period['description']})")
    print(f"    Model: Behind-the-meter (solar/wind subtracted from load)")
    
    # Initialize scenario config
    scenario_config = {
        'scenario': {
            'name': scenario_name,
            'description': f'Combined BTM scenario. {len(locations)} DCs x {dc_size}MW. Solar/wind behind-the-meter. Period: {period["description"]}',
            'two_settlement': True
        },
        'simulation': {
            'days': simulation_days
        },
        'add_loads': [],
        'add_generators': [],
        'add_storage': [],
        'add_lines': [],
        'modify_generators': [],
        'modify_lines': [],
        'disable': {}
    }
    
    # Process each location
    for location in locations:
        bus = get_bus_from_county(location, county_info)
        if bus is None:
            continue
        
        print(f"\n    Location {location} -> Bus {bus}:")
        
        # Get assets for this location
        loc_df = exp_df[exp_df['location'] == location]
        
        # Extract solar and wind capacities for BTM calculation
        solar_capacity = 0.0
        wind_capacity = 0.0
        
        for _, row in loc_df.iterrows():
            asset_type = row['asset_type'].lower()
            if asset_type == 'solar':
                solar_capacity = row['capacity']
            elif asset_type == 'wind':
                wind_capacity = row['capacity']
        
        print(f"      DC size: {dc_size} MW")
        print(f"      Solar (BTM): {solar_capacity:.2f} MW")
        print(f"      Wind (BTM): {wind_capacity:.2f} MW")
        
        # Create net load profile for this location
        net_load_profile = create_net_load_profile(
            hourly_cf=hourly_cf,
            location=location,
            dc_size=dc_size,
            solar_capacity=solar_capacity,
            wind_capacity=wind_capacity,
            bus=bus,
            output_dir=scenario_dir
        )
        
        # Add data center with net load profile
        scenario_config['add_loads'].append({
            'name': f'DataCenter_{location}',
            'bus': bus,
            'profile_file': net_load_profile
        })
        
        # Add slack generators at this DC bus
        scenario_config['add_generators'].append({
            'name': f'DC_Slack_Up_{location}',
            'bus': bus,
            'p_nom': 99999,
            'carrier': 'slack',
            'marginal_cost': 9999
        })
        scenario_config['add_generators'].append({
            'name': f'DC_Slack_Down_{location}',
            'bus': bus,
            'p_nom': 99999,
            'carrier': 'slack',
            'marginal_cost': -9999,
            'p_max_pu': 0,
            'p_min_pu': -1
        })
        
        # Process non-renewable assets (storage, dispatchables)
        for _, row in loc_df.iterrows():
            asset_type = row['asset_type'].lower()
            capacity = row['capacity']
            
            if asset_type in ['solar', 'wind']:
                continue  # Already handled in net load profile
            
            if capacity <= 0:
                continue
                
            if asset_type == 'storage':
                print(f"      Storage: {capacity:.2f} MW")
                storage = create_storage_unit(row, bus, location)
                if storage:
                    scenario_config['add_storage'].append(storage)
                    
            elif asset_type in ['smr', 'gas_turbine', 'diesel', 'geothermal']:
                print(f"      {asset_type}: {capacity:.2f} MW")
                gen = create_dispatchable_generator(row, bus, asset_type, location)
                if gen:
                    scenario_config['add_generators'].append(gen)
    
    # Write scenario config
    config_path = os.path.join(scenario_dir, 'scenario_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(scenario_config, f, default_flow_style=None, sort_keys=False)
    
    print(f"\n    Written: {config_path}")
    print(f"    Total loads (with profiles): {len(scenario_config['add_loads'])}")
    print(f"    Total generators: {len(scenario_config['add_generators'])}")
    print(f"    Total storage: {len(scenario_config['add_storage'])}")
    
    return scenario_dir


def generate_scenarios_from_siting(
    siting_csv: str,
    data_dir: str,
    output_dir: str,
    exp_name_filter: Optional[str] = None,
    periods: Optional[List[str]] = None,
    dc_size_override: Optional[float] = None
) -> List[str]:
    """Generate combined BTM multi-location scenarios from siting CSV."""
    global GEN_COUNTER
    GEN_COUNTER = 0
    
    print(f"\n{'='*60}")
    print("Generating COMBINED BTM Multi-Location Scenarios")
    print("(All locations per experiment, solar/wind behind-the-meter)")
    print(f"{'='*60}")
    
    # Load data
    print(f"\nLoading data...")
    county_info = load_county_info(data_dir)
    hourly_cf = load_hourly_cf(data_dir)
    siting_df = load_siting_data(siting_csv)
    
    print(f"  County info: {len(county_info)} counties")
    print(f"  Hourly CF: {len(hourly_cf)} rows")
    print(f"  Siting data: {len(siting_df)} rows")
    
    # Determine DC size
    if dc_size_override:
        dc_size = dc_size_override
    else:
        dc_size = parse_dc_size_from_filename(siting_csv)
        if dc_size is None:
            first_exp = siting_df['exp_name'].iloc[0]
            dc_size = parse_dc_size_from_expname(first_exp)
    
    print(f"\nData center size (per location): {dc_size} MW")
    
    # Set periods
    if periods is None:
        periods = DEFAULT_PERIODS
    print(f"Periods: {periods}")
    
    # Get unique exp_names
    if exp_name_filter:
        exp_names = [exp_name_filter]
    else:
        exp_names = siting_df['exp_name'].unique()
    
    print(f"\nProcessing {len(exp_names)} experiment(s) x {len(periods)} period(s)...")
    
    # Show summary
    for exp_name in exp_names[:3]:
        n_locs = len(siting_df[siting_df['exp_name'] == exp_name]['location'].unique())
        print(f"  {exp_name}: {n_locs} locations -> {n_locs * dc_size:.0f} MW total")
    if len(exp_names) > 3:
        print(f"  ... and {len(exp_names) - 3} more experiments")
    
    created_scenarios = []
    
    for exp_name in exp_names:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"{'='*60}")
        
        for period_key in periods:
            scenario_dir = create_combined_btm_scenario(
                siting_df=siting_df,
                exp_name=exp_name,
                county_info=county_info,
                hourly_cf=hourly_cf,
                dc_size=dc_size,
                period_key=period_key,
                base_output_dir=output_dir
            )
            
            if scenario_dir:
                created_scenarios.append(scenario_dir)
    
    print(f"\n{'='*60}")
    print(f"Created {len(created_scenarios)} combined BTM scenario(s)")
    print(f"{'='*60}")
    
    return created_scenarios


def main():
    parser = argparse.ArgumentParser(
        description='Generate combined multi-location BTM scenario configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'siting_csv',
        help='Path to siting data CSV file'
    )
    
    parser.add_argument(
        '--data-dir',
        default='../data',
        help='Path to data directory (default: ../data)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='scenarios_combined_btm',
        help='Output directory for scenarios (default: scenarios_combined_btm)'
    )
    
    parser.add_argument(
        '--exp-name',
        default=None,
        help='Only process this experiment name (default: all)'
    )
    
    parser.add_argument(
        '--periods',
        default=None,
        help='Periods to simulate: winter,spring,summer,fall (default: all)'
    )
    
    parser.add_argument(
        '--dc-size',
        type=float,
        default=None,
        help='Override data center size in MW per location (default: parse from filename)'
    )
    
    args = parser.parse_args()
    
    # Parse periods
    periods = None
    if args.periods:
        periods = [p.strip().lower() for p in args.periods.split(',')]
        for p in periods:
            if p not in SIMULATION_PERIODS:
                print(f"ERROR: Invalid period '{p}'. Valid options: {list(SIMULATION_PERIODS.keys())}")
                sys.exit(1)
    
    # Run generation
    scenarios = generate_scenarios_from_siting(
        siting_csv=args.siting_csv,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        exp_name_filter=args.exp_name,
        periods=periods,
        dc_size_override=args.dc_size
    )
    
    # Print summary
    print("\nCreated scenarios:")
    for s in scenarios:
        print(f"  - {s}")
    
    print(f"\nTo run a scenario:")
    print(f"  python run_scenario.py <scenario_dir>/scenario_config.yaml config.yaml")


if __name__ == "__main__":
    main()
