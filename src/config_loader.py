import yaml
from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path


@dataclass
class DataConfig:
    case_file: str
    nodal_load_file: str
    solar_file: str
    wind_file: str
    hydro_file: str
    gendata_file: str
    # Forecast files (optional - for two-settlement market)
    nodal_load_forecast_file: str = None
    solar_forecast_file: str = None
    wind_forecast_file: str = None


@dataclass
class SimulationConfig:
    run_day: int
    n_hours: int
    start_date: str


@dataclass
class TransmissionConfig:
    relax_line_limits: bool


@dataclass
class UnitCommitmentConfig:
    committable_fuels: List[str]
    min_size_for_uc: float
    use_initial_conditions: bool


@dataclass
class RampRatesConfig:
    enabled: bool
    multiplier: float
    typical: Dict[str, float]


@dataclass
class CostsConfig:
    startup: Dict[str, float]


@dataclass
class MIPConfig:
    rel_gap: float
    time_limit: int


@dataclass
class LPConfig:
    time_limit: int


@dataclass
class SolverConfig:
    name: str
    mip: MIPConfig
    lp: LPConfig


@dataclass
class OutputConfig:
    directory: str
    save_dispatch: bool
    save_lmps: bool
    save_commitment: bool
    save_line_flows: bool
    save_summary: bool
    save_congestion: bool
    save_ramp_analysis: bool
    save_final_state: bool


@dataclass
class Config:
    data: DataConfig
    simulation: SimulationConfig
    transmission: TransmissionConfig
    unit_commitment: UnitCommitmentConfig
    ramp_rates: RampRatesConfig
    costs: CostsConfig
    solver: SolverConfig
    output: OutputConfig


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    
    # Handle forecast files
    # import pdb; pdb.set_trace()
    data_config = raw['data'].copy()
    data_config.setdefault('nodal_load_forecast_file', None)
    data_config.setdefault('solar_forecast_file', None)
    data_config.setdefault('wind_forecast_file', None)
    
    return Config(
        data=DataConfig(**data_config),
        simulation=SimulationConfig(**raw['simulation']),
        transmission=TransmissionConfig(**raw['transmission']),
        unit_commitment=UnitCommitmentConfig(**raw['unit_commitment']),
        ramp_rates=RampRatesConfig(**raw['ramp_rates']),
        costs=CostsConfig(**raw['costs']),
        solver=SolverConfig(
            name=raw['solver']['name'],
            mip=MIPConfig(**raw['solver']['mip']),
            lp=LPConfig(**raw['solver']['lp'])
        ),
        output=OutputConfig(**raw['output'])
    )
