from typing import Iterable
import numba
import numpy as np
import scipy
from pystrict import strict
from PySDM.dynamics import condensation, coalescence
from PySDM.physics import spectra, si, Formulae, constants as const
from PySDM.backends.numba.conf import JIT_FLAGS
from PySDM_examples.Morrison_and_Grabowski_2007.strato_cumulus import StratoCumulus


@strict
class Settings(StratoCumulus):
    def __dir__(self) -> Iterable[str]:
        return 'dt', 'grid', 'size', 'n_spin_up', 'versions', 'steps_per_output_interval', 'formulae', \
            'initial_dry_potential_temperature_profile', 'initial_vapour_mixing_ratio_profile'

    def __init__(self, fastmath: bool = JIT_FLAGS['fastmath']):
        super().__init__(fastmath)

        self.grid = (25, 25)
        self.size = (1500 * si.metres, 1500 * si.metres)
        self.rho_w_max = .6 * si.metres / si.seconds * (si.kilogram / si.metre ** 3)

        # output steps
        self.simulation_time = 90 * si.minute
        self.dt = 5 * si.second
        self.spin_up_time = 1 * si.hour

        self.mode_1 = spectra.Lognormal(
            norm_factor=60 / si.centimetre ** 3 / const.rho_STP,
            m_mode=0.04 * si.micrometre,
            s_geom=1.4
        )
        self.mode_2 = spectra.Lognormal(
          norm_factor=40 / si.centimetre**3 / const.rho_STP,
          m_mode=0.15 * si.micrometre,
          s_geom=1.6
        )
        self.spectrum_per_mass_of_dry_air = spectra.Sum((self.mode_1, self.mode_2))

        self.th_std0 = 289 * si.kelvins
        self.qv0 = 7.5 * si.grams / si.kilogram
        self.p0 = 1015 * si.hectopascals
        self.kappa = 1  # TODO #441!
