import re
import numpy as np
import pandas as pd
import scipy.io
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from matpowercaseframes import CaseFrames

from config_loader import DataConfig


@dataclass
class CaseData:
    """Container for MATPOWER case data."""
    bus_df: pd.DataFrame
    gen_df: pd.DataFrame
    branch_df: pd.DataFrame
    genfuel: List[str]
    gencost: Optional[List]
    
    # Column names (detected automatically)
    bus_id_col: str
    gen_bus_col: str
    pmax_col: str
    pmin_col: str
    fbus_col: str
    tbus_col: str
    r_col: str
    x_col: str
    rate_col: str


@dataclass
class TimeSeriesData:
    """Container for time series data."""
    nodal_load: np.ndarray      # (n_hours, n_buses)
    solar_mw: np.ndarray        # (n_hours, n_solar)
    wind_mw: np.ndarray         # (n_hours, n_wind)
    hydro_mw: np.ndarray        # (n_hours, n_hydro)


@dataclass
class GenData:
    """Container for generator operational data."""
    min_on: np.ndarray          # Minimum on time (hours)
    min_off: np.ndarray         # Minimum off time (hours)
    ramp_rates: Optional[np.ndarray]  # MW/hour (if available)


@dataclass
class GeneratorIndices:
    """Generator indices by fuel type."""
    solar: List[int]
    wind: List[int]
    hydro: List[int]
    renewable: set


def load_matpower_case(case_path: str) -> CaseData:
    """Load MATPOWER case file and parse all relevant data."""
    cf = CaseFrames(case_path)
    bus_df = cf.bus
    gen_df = cf.gen
    branch_df = cf.branch
    
    # Parse genfuel
    with open(case_path, 'r') as f:
        content = f.read()
    
    genfuel_match = re.search(r"mpc\.genfuel\s*=\s*\{([^}]+)\}", content, re.DOTALL)
    genfuel = re.findall(r"'([^']+)'", genfuel_match.group(1))
    
    # Parse gencost
    gencost = _parse_gencost(case_path)
    
    # Detect column names
    bus_id_col = 'bus_i' if 'bus_i' in bus_df.columns else bus_df.columns[0]
    gen_bus_col = 'bus' if 'bus' in gen_df.columns else gen_df.columns[0]
    pmax_col = 'Pmax' if 'Pmax' in gen_df.columns else gen_df.columns[8]
    pmin_col = 'Pmin' if 'Pmin' in gen_df.columns else gen_df.columns[9]
    fbus_col = 'fbus' if 'fbus' in branch_df.columns else branch_df.columns[0]
    tbus_col = 'tbus' if 'tbus' in branch_df.columns else branch_df.columns[1]
    r_col = 'r' if 'r' in branch_df.columns else branch_df.columns[2]
    x_col = 'x' if 'x' in branch_df.columns else branch_df.columns[3]
    rate_col = 'rateA' if 'rateA' in branch_df.columns else branch_df.columns[5]
    
    return CaseData(
        bus_df=bus_df,
        gen_df=gen_df,
        branch_df=branch_df,
        genfuel=genfuel,
        gencost=gencost,
        bus_id_col=bus_id_col,
        gen_bus_col=gen_bus_col,
        pmax_col=pmax_col,
        pmin_col=pmin_col,
        fbus_col=fbus_col,
        tbus_col=tbus_col,
        r_col=r_col,
        x_col=x_col,
        rate_col=rate_col
    )


def _parse_gencost(case_path: str) -> Optional[List]:
    """
    Parse MATPOWER gencost matrix.
    
    MATPOWER gencost format:
    Type 1 (piecewise linear): [1, startup, shutdown, n, p1, c1, p2, c2, ...]
    Type 2 (polynomial): [2, startup, shutdown, n, cn, cn-1, ..., c0]
    """
    with open(case_path, 'r') as f:
        content = f.read()
    
    gencost_match = re.search(r"mpc\.gencost\s*=\s*\[([\s\S]*?)\];", content)
    if not gencost_match:
        raise ValueError(f"No gencost found in case file: {case_path}")
    
    gencost_str = gencost_match.group(1)
    rows = []
    
    for line in gencost_str.strip().split('\n'):
        line = line.strip().rstrip(';').strip()
        if line and not line.startswith('%'):
            line = re.sub(r';.*', '', line).strip()
            if line:
                values = [float(x) for x in line.split()]
                rows.append(values)
    
    print(f"Parsed {len(rows)} gencost entries")
    return rows


def extract_cost_from_gencost(gencost: List, gen_idx: int, p_max: float) -> Tuple[float, float]:
    """
    Extract marginal cost and startup cost from gencost data.
    
    Returns: (marginal_cost, startup_cost)
    """
    if gen_idx >= len(gencost):
        raise ValueError(f"Generator index {gen_idx} not found in gencost")
    
    row = gencost[gen_idx]
    cost_type = int(row[0])
    startup_cost = float(row[1])
    n_points = int(row[3])
    
    if cost_type == 1:  # Piecewise linear
        points = []
        for i in range(n_points):
            p = float(row[4 + 2*i])
            c = float(row[4 + 2*i + 1])
            points.append((p, c))
        
        if len(points) >= 2:
            total_cost_diff = points[-1][1] - points[0][1]
            total_mw_diff = points[-1][0] - points[0][0]
            marginal_cost = total_cost_diff / total_mw_diff if total_mw_diff > 0 else 0
        else:
            marginal_cost = 0
            
    elif cost_type == 2:  # Polynomial
        coeffs = [float(row[4 + i]) for i in range(n_points)]
        
        if n_points == 3:  # Quadratic: c2*p^2 + c1*p + c0
            c2, c1, c0 = coeffs
            p_mid = p_max / 2
            marginal_cost = 2 * c2 * p_mid + c1
        elif n_points == 2:  # Linear: c1*p + c0
            c1, c0 = coeffs
            marginal_cost = c1
        else:
            marginal_cost = coeffs[-2] if len(coeffs) >= 2 else 0
        
        marginal_cost = max(0, marginal_cost)
    else:
        raise ValueError(f"Unknown cost type {cost_type} for generator {gen_idx}")
    
    return marginal_cost, startup_cost


def load_time_series(config: DataConfig, n_hours: int) -> TimeSeriesData:
    """Load all time series data (actuals)."""
    nodal_data = scipy.io.loadmat(config.nodal_load_file)
    solar = scipy.io.loadmat(config.solar_file)
    wind = scipy.io.loadmat(config.wind_file)
    hydro = scipy.io.loadmat(config.hydro_file)
    
    return TimeSeriesData(
        nodal_load=nodal_data['nodal_load_hourly'][:n_hours, :],
        solar_mw=solar['solar_MW'][:n_hours, :],
        wind_mw=wind['wind_MW'][:n_hours, :],
        hydro_mw=hydro['hydro_MW'][:n_hours, :]
    )


def load_forecast_time_series(config: DataConfig, n_hours: int, 
                               hydro_from_actuals: TimeSeriesData) -> TimeSeriesData:
    """
    Load forecast time series data.
    
    Args:
        config: Data configuration with forecast file paths
        n_hours: Number of hours to load
        hydro_from_actuals: Hydro data from actuals (hydro is deterministic)
    
    Returns:
        TimeSeriesData with forecasted load/solar/wind and actual hydro
    """
    nodal_data = scipy.io.loadmat(config.nodal_load_forecast_file)
    solar = scipy.io.loadmat(config.solar_forecast_file)
    wind = scipy.io.loadmat(config.wind_forecast_file)
    
    return TimeSeriesData(
        nodal_load=nodal_data['nodal_load_hourly'][:n_hours, :],
        solar_mw=solar['solar_MW'][:n_hours, :],
        wind_mw=wind['wind_MW'][:n_hours, :],
        hydro_mw=hydro_from_actuals.hydro_mw  # Hydro stays deterministic
    )


def load_gen_data(config: DataConfig) -> GenData:
    """Load generator operational data."""
    gendata = scipy.io.loadmat(config.gendata_file)
    
    ramp_rates = None
    if 'ramp_rate' in gendata:
        ramp_rates = gendata['ramp_rate'].flatten()
    
    return GenData(
        min_on=gendata['min_on'].flatten(),
        min_off=gendata['min_off'].flatten(),
        ramp_rates=ramp_rates
    )


def get_generator_indices(genfuel: List[str]) -> GeneratorIndices:
    """Get generator indices by fuel type."""
    solar_idx = [i for i, fuel in enumerate(genfuel) if fuel.lower() == 'solar']
    wind_idx = [i for i, fuel in enumerate(genfuel) if fuel.lower() == 'wind']
    hydro_idx = [i for i, fuel in enumerate(genfuel) if fuel.lower() == 'hydro']
    renewable_idx = set(solar_idx + wind_idx + hydro_idx)
    
    
    return GeneratorIndices(
        solar=solar_idx,
        wind=wind_idx,
        hydro=hydro_idx,
        renewable=renewable_idx
    )
