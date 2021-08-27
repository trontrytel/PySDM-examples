"""
Created at 29.11.2019
"""

from PySDM.physics.constants import si
from PySDM.physics import constants as const, formulae as phys
from PySDM.dynamics import condensation
from PySDM.physics.formulae import Formulae
import numpy as np
from pystrict import strict


@strict
class Settings:
    def __init__(self, w_avg: float, N_STP: float, r_dry: float, mass_of_dry_air: float, coord: str = 'VolumeLogarithm'):
        self.formulae = Formulae(saturation_vapour_pressure='AugustRocheMagnus', condensation_coordinate=coord)

        self.p0 = 1000 * si.hectopascals
        self.RH0 = .98
        self.kappa = .2  # TODO #441
        self.T0 = 300 * si.kelvin
        self.z_half = 150 * si.metres

        pvs = self.formulae.saturation_vapour_pressure.pvs_Celsius(self.T0 - const.T0)
        self.q0 = const.eps / (self.p0 / self.RH0 / pvs - 1)
        self.w_avg = w_avg
        self.r_dry = r_dry
        self.N_STP = N_STP
        self.n_in_dv = N_STP / const.rho_STP * mass_of_dry_air
        self.mass_of_dry_air = mass_of_dry_air
        self.n_output = 500

        self.rtol_x = condensation.default_rtol_x
        self.rtol_thd = condensation.default_rtol_thd
        self.coord = 'volume logarithm'
        self.dt_cond_range = condensation.default_cond_range

    @property
    def dt_max(self):
        t_total = 2 * self.z_half / self.w_avg
        result = t_total / self.n_output
        if result < 1 * si.centimetre / si.second:
            result /= 100  # TODO #411
        return result

    def w(self, t):
        return self.w_avg * np.pi / 2 * np.sin(np.pi * t / self.z_half * self.w_avg)


w_avgs = (
    100 * si.centimetre / si.second,
    # 50 * si.centimetre / si.second,
    .2 * si.centimetre / si.second
)

N_STPs = (
    50 / si.centimetre ** 3,
    500 / si.centimetre ** 3
)

r_drys = (
    .1 * si.micrometre,
    .05 * si.micrometre
)

setups = []
for w_i in range(len(w_avgs)):
    for N_i in range(len(N_STPs)):
        for rd_i in range(len(r_drys)):
            if not rd_i == N_i == 1:
                setups.append(Settings(
                    w_avg=w_avgs[w_i],
                    N_STP=N_STPs[N_i],
                    r_dry=r_drys[rd_i],
                    mass_of_dry_air=1000 * si.kilogram
                ))
