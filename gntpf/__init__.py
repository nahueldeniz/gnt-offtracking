"""Off-tracking-minimising path following for Generalised N-Trailer vehicles.

A Python/CasADi reimplementation of the two-stage framework: an optimisation-based
reference generator that produces a kinematically feasible posture for every
segment of the chain, followed by a tracking NMPC, with a moving-horizon estimator
reconstructing the states that are not measured.
"""

from .model import GNT, atan2c, wrap_to_pi
from .paths import Path, LocalFit, make_path, corridor_bounds, PATHS
from .refgen import ReferenceGenerator, RefGenWeights
from .nmpc import TrackingNMPC
from .nmhe import NMHE
from .ekf import EKF
from .sim import SensorModel, SimConfig, SimResult, simulate, initial_state_on_path
from .baselines import (MichalekCascade, PurePursuit, chain_condition,
                        simulate_baseline)

__version__ = "0.1.0"

__all__ = [
    "GNT", "atan2c", "wrap_to_pi",
    "Path", "LocalFit", "make_path", "corridor_bounds", "PATHS",
    "ReferenceGenerator", "RefGenWeights",
    "TrackingNMPC", "NMHE", "EKF",
    "SensorModel", "SimConfig", "SimResult", "simulate", "initial_state_on_path",
    "MichalekCascade", "PurePursuit", "chain_condition", "simulate_baseline",
]
