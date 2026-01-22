# Production Cost Model (PCM) Pipeline

A PyPSA-based production cost model for simulating electricity markets with data center siting scenarios. Supports day-ahead and real-time two-settlement market simulations.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Data Requirements](#data-requirements)
6. [Running Simulations](#running-simulations)
7. [Scenario Generation](#scenario-generation)
8. [Output Files](#output-files)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This pipeline simulates wholesale electricity markets with the following capabilities:

- **Single-settlement**: Day-ahead market only
- **Two-settlement**: Day-ahead market with forecasts, real-time market with actuals
- **Scenario-based**: Add data centers, generators, storage, and modify network components
- **Behind-the-meter (BTM)**: Model renewables as reducing data center net load
- **Grid-connected**: Model renewables as separate generators injecting to the grid

The model uses unit commitment (UC) with constraints including:
- Minimum up/down times
- Ramp rate limits
- Start-up costs
- Transmission constraints (DC power flow)

---

## Directory Structure

```
project/
├── src/
│   ├── config.yaml                 # Base configuration
│   ├── run_scenario.py             # Main entry point for scenarios
│   ├── run_two_settlement.py       # Two-settlement market simulation
│   ├── solver.py                   # Optimization solver
│   ├── network_builder.py          # PyPSA network construction
│   ├── data_loader.py              # Data loading utilities
│   ├── config_loader.py            # Configuration parser
│   ├── initial_conditions.py       # Generator initial state calculation
│   │
│   ├── scenarios/                  # Generated scenario directories
│   │   └── <scenario_name>/
│   │       ├── scenario_config.yaml
│   │       └── profiles/           # Time-series profiles
│   │
│   ├── results_*/                  # Output directories
│   │
│   ├── generate_specific_scenarios.py      # Targeted scenario generator
│   ├── generate_siting_scenario.py         # Grid-connected scenarios
│   ├── generate_siting_scenario_btm.py     # BTM scenarios
│   ├── generate_siting_scenario_combined.py        # Multi-location grid-connected
│   ├── generate_siting_scenario_combined_btm.py    # Multi-location BTM
│
├── data/
│   ├── case_ACTIVSg2000.mat        # MATPOWER case file (network topology)
│   ├── nodal_load.csv              # Actual hourly load by bus (8760 x N_buses)
│   ├── nodal_load_forecast.csv     # Forecasted hourly load (8760 x N_buses)
│   ├── solar_cf.csv                # Solar capacity factors (8760 x N_solar)
│   ├── wind_cf.csv                 # Wind capacity factors (8760 x N_wind)
│   ├── solar_forecast.csv          # Solar CF forecast (8760 x N_solar)
│   ├── wind_forecast.csv           # Wind CF forecast (8760 x N_wind)
│   ├── gen_data.csv                # Generator parameters
│   │
│   ├── mapping_data/
│   │   └── county_info.csv         # County FIPS to bus mapping
│   │
│   └── siting_data/
│       ├── 100MW_all_gen_nameplate_capacities.csv
│       ├── 1GW_all_gen_nameplate_capacities.csv
│       └── total_hourly_solar_wind_cf_tx.csv   # Location-specific CF data
```

---

## Installation

### Requirements

- Python 3.8+
- Gurobi solver (with valid license)

### Python Dependencies

```bash
pip install pypsa pandas numpy scipy matplotlib pyyaml gurobipy
```

### Gurobi License

Ensure Gurobi is properly licensed. For academic use:
```bash
grbgetkey <your-key>
```

---

## Configuration

### Base Configuration (config.yaml)

```yaml
data:
  case_file: "../data/case_ACTIVSg2000.mat"
  nodal_load_file: "../data/nodal_load.csv"
  nodal_load_forecast_file: "../data/nodal_load_forecast.csv"
  solar_file: "../data/solar_cf.csv"
  wind_file: "../data/wind_cf.csv"
  solar_forecast_file: "../data/solar_forecast.csv"
  wind_forecast_file: "../data/wind_forecast.csv"
  gen_data_file: "../data/gen_data.csv"

simulation:
  n_hours: 8760
  run_day: 0

solver:
  name: gurobi
  lp:
    time_limit: 300
  milp:
    time_limit: 600
    mip_gap: 0.01

output:
  directory: "results"
```

### Scenario Configuration (scenario_config.yaml)

```yaml
scenario:
  name: example_scenario
  description: "Example data center scenario"
  two_settlement: true          # Enable two-settlement market

simulation:
  days: [201, 202, 203]         # Days to simulate (1-indexed day of year)

add_loads:
  - name: DataCenter_48371
    bus: 3001070
    p_mw: 200.0                 # Constant load
    # OR use profile:
    # profile_file: /path/to/net_load.csv

add_generators:
  - name: DC_Slack_Up_48371
    bus: 3001070
    p_nom: 99999
    carrier: slack
    marginal_cost: 9999

  - name: DC_Slack_Down_48371
    bus: 3001070
    p_nom: 99999
    carrier: slack
    marginal_cost: -9999
    p_max_pu: 0
    p_min_pu: -1

  - name: DC_Solar_48371
    bus: 3001070
    p_nom: 50.0
    carrier: solar
    marginal_cost: 0
    profile_file: /path/to/solar_profile.csv

add_storage:
  - name: DC_Battery_48371
    bus: 3001070
    p_nom: 100.0
    max_hours: 4
    efficiency_store: 0.95
    efficiency_dispatch: 0.95
    marginal_cost: 0.5

add_lines: []
modify_generators: []
modify_lines: []
disable: {}
```

---

## Data Requirements

### Time-Series Data Format

All CSV files should have 8760 rows (hourly data for one year).

**nodal_load.csv** / **nodal_load_forecast.csv**:
```csv
,Bus_3001001,Bus_3001002,...
0,150.5,200.3,...
1,145.2,195.1,...
...
8759,160.0,210.5,...
```

**solar_cf.csv** / **wind_cf.csv**:
```csv
,Gen_1_solar,Gen_2_solar,...
0,0.0,0.0,...
1,0.0,0.0,...
...
12,0.85,0.78,...
...
```

### Siting Data Format

**nameplate_capacities.csv**:
```csv
exp_name,location,asset_type,asset_name,capacity
all_gen_1GW___001__ren_penetration=0__site_gen_penetration=0,48189,solar,solar,104.43
all_gen_1GW___001__ren_penetration=0__site_gen_penetration=0,48189,wind,wind,223.67
all_gen_1GW___001__ren_penetration=0__site_gen_penetration=0,48189,storage,lithium_ion,100.0
all_gen_1GW___001__ren_penetration=0__site_gen_penetration=0,48189,smr,smr,100.0
```

### County to Bus Mapping

**county_info.csv**:
```csv
,cfips,bestbus
0,48001,3001001
1,48003,3001002
...
```

---

## Running Simulations

### Single Scenario

```bash
cd src/

# Single-settlement (DA only)
python run_scenario.py scenarios/my_scenario/scenario_config.yaml config.yaml

# Two-settlement is enabled in scenario_config.yaml with:
#   two_settlement: true
```

### Baseline (No Data Center)

Create a minimal scenario config:
```yaml
scenario:
  name: baseline_summer
  description: "No data center baseline"
  two_settlement: true

simulation:
  days: [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215]

add_loads: []
add_generators: []
add_storage: []
add_lines: []
modify_generators: []
modify_lines: []
disable: {}
```

### HPC Batch Runs

Generate configs.txt with scenario paths:
```bash
ls scenarios_specific_btm/*/scenario_config.yaml > configs.txt
```

Use with job arrays:
```bash
# In SLURM script
CONFIG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" configs.txt)
python run_scenario.py $CONFIG config.yaml
```

---

## Scenario Generation

### Targeted Scenarios (Specific Locations/Sizes)

Edit `generate_specific_scenarios.py` to set:
```python
LOCATIONS = [48139, 48371, 48383, 48479]
TARGET_SIZES = [100, 200, 500, 1000]  # MW
EXP_FILTER = 'site_gen_penetration=0'
```

Run:
```bash
python generate_specific_scenarios.py
# Output: scenarios_specific_btm/
```

### Grid-Connected Scenarios

Solar/wind inject power to the grid as separate generators:
```bash
python generate_siting_scenario.py ../data/siting_data/100MW_all_gen_nameplate_capacities.csv
# Output: scenarios/
```

### Behind-the-Meter (BTM) Scenarios

Solar/wind reduce data center net load:
```bash
python generate_siting_scenario_btm.py ../data/siting_data/100MW_all_gen_nameplate_capacities.csv
# Output: scenarios_btm/
```

Net load calculation:
```
Net_Load[hour] = DC_size_MW - (Solar_CF[hour] * Solar_MW) - (Wind_CF[hour] * Wind_MW)
```

### Multi-Location (Combined) Scenarios

All locations from one experiment in a single scenario:
```bash
# Grid-connected
python generate_siting_scenario_combined.py ../data/siting_data/100MW_all_gen_nameplate_capacities.csv

# BTM
python generate_siting_scenario_combined_btm.py ../data/siting_data/100MW_all_gen_nameplate_capacities.csv
```

### Seasonal Periods

All generators support `--periods` flag:
```bash
python generate_siting_scenario.py data.csv --periods summer
python generate_siting_scenario.py data.csv --periods winter,summer
```

Default periods:
| Period | Days | Date Range |
|--------|------|------------|
| winter | 35-49 | Feb 5-19 |
| spring | 104-118 | Apr 14-28 |
| summer | 201-215 | Jul 20 - Aug 3 |
| fall | 305-318 | Nov 1-14 |

## Output Files

Each simulation day produces:

| File | Description |
|------|-------------|
| `da_dispatch_dayN.csv` | Day-ahead generator dispatch (MW) |
| `da_lmps_dayN.csv` | Day-ahead LMPs by bus ($/MWh) |
| `da_line_flows_dayN.csv` | Day-ahead line flows (MW) |
| `da_congestion_dayN.csv` | Day-ahead line loading (%) |
| `rt_dispatch_dayN.csv` | Real-time generator dispatch (MW) |
| `rt_lmps_dayN.csv` | Real-time LMPs by bus ($/MWh) |
| `rt_line_flows_dayN.csv` | Real-time line flows (MW) |
| `rt_congestion_dayN.csv` | Real-time line loading (%) |
| `commitment_dayN.csv` | Generator commitment status (0/1) |
| `summary_dayN.csv` | Daily summary statistics |
| `price_convergence_dayN.csv` | DA vs RT price comparison |
| `final_state_dayN.csv` | Generator state for next day initialization |

---

## Troubleshooting

### Infeasible Optimization

**Symptom**: "Model is infeasible or unbounded"

**Common Causes**:

1. **Missing slack generators**: Ensure scenario has `DC_Slack_Up` and `DC_Slack_Down` with:
   ```yaml
   - name: DC_Slack_Down_XXXXX
     marginal_cost: -9999
     p_max_pu: 0      # CRITICAL - must be 0
     p_min_pu: -1     # CRITICAL - must be -1
   ```

2. **Profile indexing bug**: Ensure profiles are indexed by simulation day, not starting from hour 0:
   ```python
   start_day = snapshots[0].dayofyear
   start_hour = (start_day - 1) * 24
   end_hour = start_hour + len(snapshots)
   p_set = profile.values[start_hour:end_hour]
   ```

3. **Summer congestion**: Summer peak load may cause transmission infeasibility. Try relaxing line constraints:
   ```python
   network.lines['s_nom'] *= 1.2  # 20% increase
   ```

4. **Initial conditions conflict**: Generator min up/down times may conflict with initial state. Check `initial_conditions.py`.

### Generator Name Conflicts

**Symptom**: "WARNING: Generator already defined and will be skipped"

**Solution**: Use unique naming with location prefix:
```python
name = f"DC_{asset_type}_{location}_{counter}"
```

### Slack Generator Unbounded

**Symptom**: Optimization unbounded with large negative costs

**Cause**: `Slack_Down` without `p_max_pu=0` can generate positive power at negative cost.

**Solution**: Always include both constraints:
```yaml
p_max_pu: 0
p_min_pu: -1
```

### Missing Data for Season

**Symptom**: Specific seasons fail while others work

**Check**:
1. Data files have 8760 rows
2. Profile loading uses correct hour indexing
3. Load/generation data exists for those hours

```python
# Verify data availability
import pandas as pd
load = pd.read_csv('nodal_load.csv', index_col=0)
print(f"Rows: {len(load)}, Summer row 4824: {load.iloc[4824].sum()}")
```

### Real-Time Infeasibility

**Symptom**: DA succeeds but RT fails

**Cause**: RT uses actual data which may differ significantly from DA forecast, but commitment is fixed.

**Solution**: Add RT slack generators in `solve_real_time()`:
```python
if 'RT_Slack_Up' not in network.generators.index:
    network.add("Generator", name="RT_Slack_Up", ...)
    network.add("Generator", name="RT_Slack_Down", ...)
```
