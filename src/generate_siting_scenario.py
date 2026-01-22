#!/usr/bin/env python3
"""
Generate scenario configurations from siting data.

This script reads siting optimization results and generates:
1. scenario_config.yaml with data center load, generators, and storage
2. Solar/wind profile CSVs (CF × capacity)

Usage:
    python generate_siting_scenario.py <siting_csv> [--exp-name NAME] [--days 200,201,202]
    
Examples:
    # Process all scenarios in a siting file
    python generate_siting_scenario.py ../data/siting_data/100MW_onlyren_nameplate_capacities.csv
    
    # Process specific scenario
    python generate_siting_scenario.py ../data/siting_data/1GW_all_gen_nameplate_capacities.csv \
        --exp-name "all_gen_1GW___001__ren_penetration=0__site_gen_penetration=0"
    
    # Specify simulation days
    python generate_siting_scenario.py ../data/siting_data/nameplate_capacities.csv \
        --days 200,201,202,203,204,205,206
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


# Marginal costs for dispatchable generators 
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
        'days': list(range(35, 50))    # Feb 5 (day 35) to Feb 19 (day 49)
    },
    'spring': {
        'name': 'Spring', 
        'description': 'April 14-28',
        'days': list(range(104, 119))  # Apr 14 (day 104) to Apr 28 (day 118)
    },
    'summer': {
        'name': 'Summer',
        'description': 'July 20 - August 3',
        'days': list(range(201, 216))  # Jul 20 (day 201) to Aug 3 (day 215)
    },
    'fall': {
        'name': 'Fall',
        'description': 'November 1-14',
        'days': list(range(305, 319))  # Nov 1 (day 305) to Nov 14 (day 318)
    }
}

DEFAULT_PERIODS = ['winter', 'spring', 'summer', 'fall']


GEN_COUNTER = 0


def load_county_info(data_dir: str) -> pd.DataFrame:
    """Load county FIPS to bus mapping."""
    county_info_path = os.path.join(data_dir, 'mapping_data', 'county_info.csv')
    df = pd.read_csv(county_info_path, index_col=0)
    return df


def load_hourly_cf(data_dir: str) -> pd.DataFrame:
    """Load hourly capacity factors for solar and wind."""
    cf_path = os.path.join(data_dir, 'siting_data', 'total_hourly_solar_wind_cf_tx.csv')
    df = pd.read_csv(cf_path)
    return df


def load_siting_data(siting_csv_path: str) -> pd.DataFrame:
    """Load siting optimization results."""
    df = pd.read_csv(siting_csv_path)
    return df


def parse_dc_size_from_expname(exp_name: str) -> float:
    """
    Parse data center size from experiment name.
    
    Examples:
        "all_gen_1GW___001__ren_penetration=0__site_gen_penetration=0" -> 1000.0
        "onlyren_100MW___001__ren_penetration=0__site_gen_penetration=0" -> 100.0
    """
    # Try to match patterns like "100MW", "200MW", "500MW", "1GW"
    match = re.search(r'(\d+)(MW|GW)', exp_name, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2).upper()
        if unit == 'GW':
            return value * 1000  # Convert GW to MW
        return value
    
    print(f"  WARNING: Could not parse DC size from '{exp_name}', defaulting to 100 MW")
    return 100.0


def parse_dc_size_from_filename(filename: str) -> float:
    """
    Parse data center size from siting CSV filename.
    
    Examples:
        "100MW_onlyren_nameplate_capacities.csv" -> 100.0
        "1GW_all_gen_nameplate_capacities.csv" -> 1000.0
    """
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
    """Map county FIPS code to bus number."""
    match = county_info[county_info['cfips'] == county_fips]
    if len(match) == 0:
        print(f"  WARNING: No bus found for county FIPS {county_fips}")
        return None
    return int(match['bestbus'].iloc[0])


def create_renewable_profile(
    hourly_cf: pd.DataFrame,
    location: int,
    capacity: float,
    gen_type: str,  # 'solar' or 'wind'
    bus: int,
    output_dir: str
) -> str:
    """
    Create renewable generation profile CSV.
    
    Profile values = CF × capacity (MW output for each hour)
    
    Returns:
        Absolute path to profile file
    """
    # Filter CF data for this location and sort by hour
    cf_col = f'hourly_cf_{gen_type}'
    location_data = hourly_cf[hourly_cf['location'] == location].sort_values('hour')
    location_cf = location_data[cf_col].values
    
    # bypass cfs to make sure things work at least
    # if len(location_cf) == 0:
    #     print(f"  WARNING: No CF data for location {location}, using zeros")
    #     location_cf = np.zeros(8760)
    # elif len(location_cf) != 8760:
    #     print(f"  WARNING: Expected 8760 hours, got {len(location_cf)} for location {location}")
    
    # Calculate MW output profile
    profile_mw = location_cf * capacity
    
    # Create output directory if needed
    profiles_dir = os.path.join(output_dir, 'profiles')
    os.makedirs(profiles_dir, exist_ok=True)
    
    # Save profile
    filename = f'{gen_type}_{capacity:.2f}_{bus}_{location}.csv'
    filepath = os.path.join(profiles_dir, filename)
    
    profile_series = pd.Series(profile_mw, name='p_mw')
    profile_series.to_csv(filepath, index=True, header=True)
    
    # USE abolsute paths please
    return os.path.abspath(filepath)

def get_next_gen_name(prefix: str = "Gen", location: int = 0) -> str:
    """Get next unique generator name with location to avoid conflicts."""
    global GEN_COUNTER
    name = f"DC_{prefix}_{location}_{GEN_COUNTER}"
    GEN_COUNTER += 1
    return name


def create_solar_generator(
    row: pd.Series,
    bus: int,
    hourly_cf: pd.DataFrame,
    output_dir: str,
    location: int
) -> Dict:
    """Create solar generator configuration."""
    capacity = row['capacity']
    
    if capacity <= 0:
        return None
    
    profile_path = create_renewable_profile(
        hourly_cf, location, capacity, 'solar', bus, output_dir
    )
    
    return {
        'name': get_next_gen_name('Solar', location),
        'bus': bus,
        'p_nom': capacity,
        'carrier': 'solar',
        'marginal_cost': 0,
        'profile_file': profile_path
    }


def create_wind_generator(
    row: pd.Series,
    bus: int,
    hourly_cf: pd.DataFrame,
    output_dir: str,
    location: int
) -> Dict:
    """Create wind generator configuration."""
    capacity = row['capacity']
    
    if capacity <= 0:
        return None
    
    profile_path = create_renewable_profile(
        hourly_cf, location, capacity, 'wind', bus, output_dir
    )
    
    return {
        'name': get_next_gen_name('Wind', location),
        'bus': bus,
        'p_nom': capacity,
        'carrier': 'wind',
        'marginal_cost': 0,
        'profile_file': profile_path
    }


def create_storage_unit(row: pd.Series, bus: int, location: int) -> Dict:
    """Create battery storage configuration."""
    capacity = row['capacity']
    
    if capacity <= 0:
        return None
    
    return {
        'name': get_next_gen_name('Battery', location),
        'bus': bus,
        'p_nom': capacity,
        'max_hours': 4,  # 4-hour battery
        'efficiency_store': 0.95,
        'efficiency_dispatch': 0.95,
        'cyclic_state_of_charge': False,
        'marginal_cost': 0.5  # Small cost to avoid degeneracy
    }


def create_dispatchable_generator(
    row: pd.Series,
    bus: int,
    gen_type: str,
    location: int
) -> Dict:
    """
    Create dispatchable generator configuration.
    
    For SMR, gas turbine, diesel, geothermal.
    These have no profile - dispatch determined by PCM optimization.
    """
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

def create_scenario_for_location(
    siting_df: pd.DataFrame,
    exp_name: str,
    location: int,
    county_info: pd.DataFrame,
    hourly_cf: pd.DataFrame,
    dc_size: float,
    period_key: str,
    base_output_dir: str
) -> Optional[str]:
    """
    Create scenario configuration for a single location and period.
    
    Returns:
        Path to created scenario directory, or None if failed
    """
    global GEN_COUNTER
    
    GEN_COUNTER = 0
    
    # Get period info
    period = SIMULATION_PERIODS[period_key]
    simulation_days = period['days']
    
    df = siting_df[(siting_df['exp_name'] == exp_name) & (siting_df['location'] == location)]
    
    if len(df) == 0:
        print(f"  No data for exp_name={exp_name}, location={location}")
        return None
    
    bus = get_bus_from_county(location, county_info)
    if bus is None:
        return None
    

    safe_exp_name = re.sub(r'[^\w\-_]', '_', exp_name)
    scenario_name = f"{safe_exp_name}_loc{location}_{period_key}"
    scenario_dir = os.path.join(base_output_dir, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)

    # Print sanity check here    
    print(f"\n  Creating scenario: {scenario_name}")
    print(f"    Location: {location} -> Bus: {bus}")
    print(f"    Data center size: {dc_size} MW")
    print(f"    Period: {period['name']} ({period['description']})")
    
    scenario_config = {
        'scenario': {
            'name': scenario_name,
            'description': f'Auto-generated from siting data. DC: {dc_size}MW at county {location}. Period: {period["description"]}',
            'two_settlement': True
        },
        'simulation': {
            'days': simulation_days
        },
        'add_loads': [
            {
                'name': f'DataCenter_{location}',
                'bus': bus,
                'p_mw': dc_size
            }
        ],
        'add_generators': [
            # Add slack generators at DC bus for feasibility
            {
                'name': f'DC_Slack_Up_{location}',
                'bus': bus,
                'p_nom': 99999,
                'carrier': 'slack',
                'marginal_cost': 9999
            },
            {
                'name': f'DC_Slack_Down_{location}',
                'bus': bus,
                'p_nom': 99999,
                'carrier': 'slack',
                'marginal_cost': 9999,
                'p_max_pu': 0,
                'p_min_pu': -1
            }
        ],
        'add_storage': [],
        'add_lines': [],
        'modify_generators': [],
        'modify_lines': [],
        'disable': {}
    }
    
    for _, row in df.iterrows():
        asset_type = row['asset_type'].lower()
        capacity = row['capacity']
        
        print(f"    {asset_type}: {capacity:.2f} MW")
        
        if asset_type == 'solar':
            gen = create_solar_generator(row, bus, hourly_cf, scenario_dir, location)
            if gen:
                scenario_config['add_generators'].append(gen)
                
        elif asset_type == 'wind':
            gen = create_wind_generator(row, bus, hourly_cf, scenario_dir, location)
            if gen:
                scenario_config['add_generators'].append(gen)
                
        elif asset_type == 'storage':
            storage = create_storage_unit(row, bus, location)
            if storage:
                scenario_config['add_storage'].append(storage)
                
        elif asset_type in ['smr', 'gas_turbine', 'diesel', 'geothermal']:
            gen = create_dispatchable_generator(row, bus, asset_type, location)
            if gen:
                scenario_config['add_generators'].append(gen)
                
        else:
            print(f"      WARNING: Unknown asset type '{asset_type}'")
    
    config_path = os.path.join(scenario_dir, 'scenario_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(scenario_config, f, default_flow_style=None, sort_keys=False)
    
    print(f"    Written: {config_path}")
    
    return scenario_dir


def generate_scenarios_from_siting(
    siting_csv: str,
    data_dir: str,
    output_dir: str,
    exp_name_filter: Optional[str] = None,
    periods: Optional[List[str]] = None,
    dc_size_override: Optional[float] = None
) -> List[str]:
    """
    Generate all scenarios from a siting CSV file.
    
    Args:
        siting_csv: Path to siting data CSV
        data_dir: Path to data directory (contains mapping_data, siting_data)
        output_dir: Base output directory for scenarios
        exp_name_filter: Only process this exp_name (or all if None)
        periods: List of periods to generate ('winter', 'spring', 'summer', 'fall')
        dc_size_override: Override DC size (MW) instead of parsing from filename
    
    Returns:
        List of created scenario directories
    """
    global GEN_COUNTER
    GEN_COUNTER = 0
    
    print(f"\n{'='*60}")
    print("Generating Siting Scenarios")
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
            # Try to parse from first exp_name
            first_exp = siting_df['exp_name'].iloc[0]
            dc_size = parse_dc_size_from_expname(first_exp)
    
    print(f"\nData center size: {dc_size} MW")
    
    # Set periods
    if periods is None:
        periods = DEFAULT_PERIODS
    print(f"Periods: {periods}")
    for p in periods:
        period_info = SIMULATION_PERIODS[p]
        print(f"  {p}: {period_info['description']} (days {period_info['days'][0]}-{period_info['days'][-1]})")
    
    # Get unique exp_names
    if exp_name_filter:
        exp_names = [exp_name_filter]
    else:
        exp_names = siting_df['exp_name'].unique()
    
    print(f"\nProcessing {len(exp_names)} experiment(s) x {len(periods)} period(s)...")
    
    created_scenarios = []
    
    for exp_name in exp_names:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"{'='*60}")
        
        # Get unique locations for this experiment
        exp_df = siting_df[siting_df['exp_name'] == exp_name]
        locations = exp_df['location'].unique()
        
        print(f"Found {len(locations)} location(s)")
        
        for location in locations:
            for period_key in periods:
                scenario_dir = create_scenario_for_location(
                    siting_df=siting_df,
                    exp_name=exp_name,
                    location=location,
                    county_info=county_info,
                    hourly_cf=hourly_cf,
                    dc_size=dc_size,
                    period_key=period_key,
                    base_output_dir=output_dir
                )
                
                if scenario_dir:
                    created_scenarios.append(scenario_dir)
    
    print(f"\n{'='*60}")
    print(f"Created {len(created_scenarios)} scenario(s)")
    print(f"{'='*60}")
    
    return created_scenarios

def main():
    parser = argparse.ArgumentParser(
        description='Generate scenario configurations from siting data',
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
        default='scenarios',
        help='Output directory for scenarios (default: scenarios)'
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
        help='Override data center size in MW (default: parse from filename)'
    )
    
    args = parser.parse_args()
    
    # Parse periods
    periods = None
    if args.periods:
        periods = [p.strip().lower() for p in args.periods.split(',')]
        # Validate periods
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
    
    print("\nCreated scenarios:")
    for s in scenarios:
        print(f"  - {s}")


if __name__ == "__main__":
    main()