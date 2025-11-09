"""塔径估算工具。"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DiameterEstimate:
    vapor_flow: float
    superficial_velocity: float
    diameter: float
    basis: str


def from_superficial_velocity(vapor_vol_flow: float, superficial_velocity: float) -> DiameterEstimate:
    """根据空塔气速初估塔径。

    vapor_vol_flow 需为 m^3/s，superficial_velocity 为 m/s。
    """

    if vapor_vol_flow <= 0 or superficial_velocity <= 0:
        raise ValueError("流量与气速需为正值")
    diameter = math.sqrt(4 * vapor_vol_flow / (math.pi * superficial_velocity))
    return DiameterEstimate(
        vapor_flow=vapor_vol_flow,
        superficial_velocity=superficial_velocity,
        diameter=diameter,
        basis="direct",
    )


def ideal_gas_diameter(
    vapor_molar_flow: float,
    temperature: float,
    pressure: float,
    superficial_velocity: float,
    *,
    gas_constant: float = 8.314,
) -> DiameterEstimate:
    """理想气体假设下以摩尔流量估算塔径。

    vapor_molar_flow: kmol/h
    temperature: K
    pressure: kPa
    superficial_velocity: m/s
    """

    if vapor_molar_flow <= 0:
        raise ValueError("汽相摩尔流量需 > 0")
    # 转为 mol/s
    mol_s = vapor_molar_flow * 1000 / 3600
    pressure_pa = pressure * 1000
    volumetric_flow = mol_s * gas_constant * temperature / pressure_pa
    diameter_est = from_superficial_velocity(volumetric_flow, superficial_velocity)
    diameter_est.basis = "ideal_gas"
    return diameter_est

