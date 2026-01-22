import pandas as pd
import pypsa
from dataclasses import dataclass
from typing import Tuple

from config_loader import Config


@dataclass
class SolverResults:
    """Container for solver results."""
    uc_cost: float
    ed_cost: float
    commitment_status: pd.DataFrame
    day_snapshots: pd.DatetimeIndex


def solve_day(
    network: pypsa.Network,
    config: Config,
    day: int
) -> SolverResults:
    """
    Solve unit commitment and economic dispatch for a single day.
    
    Two-stage approach:
    1. Stage 1 (MIP): Solve UC to determine commitment schedule
    2. Stage 2 (LP): Fix commitment, solve ED to get LMPs
    
    Args:
        network: PyPSA network
        config: Configuration object
        day: Day index (0-indexed)
    
    Returns:
        SolverResults object
    """
    # Get day snapshots
    start_idx = day * 24
    end_idx = start_idx + 24
    day_snapshots = network.snapshots[start_idx:end_idx]
    
    # Stage 1: Unit Commitment (MIP)
    uc_cost = _solve_unit_commitment(network, config, day_snapshots, day)
    
    # Get commitment status before modifying network
    committable_gens = network.generators[network.generators['committable']].index.tolist()
    commitment_status = network.generators_t.status.loc[day_snapshots, committable_gens].copy()
    
    # Stage 2: Economic Dispatch (LP)
    ed_cost = _solve_economic_dispatch(
        network, config, day_snapshots, committable_gens, commitment_status
    )
    
    return SolverResults(
        uc_cost=uc_cost,
        ed_cost=ed_cost,
        commitment_status=commitment_status,
        day_snapshots=day_snapshots
    )


def _solve_unit_commitment(
    network: pypsa.Network,
    config: Config,
    day_snapshots: pd.DatetimeIndex,
    day: int
) -> float:
    """
    Stage 1: Solve unit commitment (MIP).
    
    Returns:
        Objective value (total cost)
    """
    
    network.optimize(
        snapshots=day_snapshots,
        solver_name=config.solver.name,
        # solver_options={
        #     'mip_rel_gap': config.solver.mip.rel_gap,
        #     'time_limit': config.solver.mip.time_limit
        # }
        # For Gurobi:
        solver_options={
            'MIPGap': config.solver.mip.rel_gap,
            'TimeLimit': config.solver.mip.time_limit
        }
    )
    # model = network.model.solver_model
    # model.computeIIS()
    # model.write("summer_infeasible.ilp")
    # print("IIS written - check summer_infeasible.ilp")
    uc_cost = network.objective
    
    return uc_cost


def _solve_economic_dispatch(
    network: pypsa.Network,
    config: Config,
    day_snapshots: pd.DatetimeIndex,
    committable_gens: list,
    commitment_status: pd.DataFrame
) -> float:
    """
    Stage 2: Fix commitment and solve economic dispatch (LP) for LMPs.
    
    Returns:
        Objective value (total cost)
    """
    
    # Disable ramp constraints for LP (already enforced in MIP)
    network.generators['ramp_limit_up'] = 1.0
    network.generators['ramp_limit_down'] = 1.0
    
    # Fix commitment by setting p_max_pu and p_min_pu based on status
    for gen in committable_gens:
        status_series = commitment_status[gen]
        network.generators_t.p_max_pu.loc[day_snapshots, gen] = status_series
        original_p_min_pu = network.generators.loc[gen, 'p_min_pu']
        network.generators_t.p_min_pu.loc[day_snapshots, gen] = status_series * original_p_min_pu
    
    # Make generators non-committable for LP
    network.generators.loc[committable_gens, 'committable'] = False
    
    # Solve LP
    network.optimize(
        snapshots=day_snapshots,
        solver_name=config.solver.name,
        solver_options={
            'time_limit': config.solver.lp.time_limit
        }
    )
    # model = network.model.solver_model
    # model.computeIIS()
    # model.write("summer_infeasible.ilp")
    # print("IIS written - check summer_infeasible_ED.ilp")
    ed_cost = network.objective
    
    return ed_cost
