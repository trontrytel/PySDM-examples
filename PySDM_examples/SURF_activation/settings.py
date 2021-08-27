from PySDM.backends import CPU
from PySDM.physics import si, spectra
from PySDM.initialisation import spectral_sampling
from PySDM.dynamics import condensation
import numpy as np
from pystrict import strict


@strict
class Settings:

    def __init__(self, n_sd: int = 100, dt_output: float = 1 * si.second, dt_max: float = 1 * si.second):

        self.total_time = 15 * si.minutes #3 * si.hours
        self.mass_of_dry_air = 1000 * si.kilogram  # TODO #335 doubled with jupyter si unit

        self.n_steps = int(self.total_time / (5 * si.second))  # TODO #334 rename to n_output
        self.n_sd = n_sd

        self.mode_1 = spectra.Lognormal(
            norm_factor=1000 / si.milligram * self.mass_of_dry_air,
            m_mode=0.04 * si.micrometre,
            s_geom=1.4
        )
        self.mode_2 = spectra.Lognormal(
          norm_factor=1000 / si.milligram * self.mass_of_dry_air,
          m_mode=0.15 * si.micrometre,
          s_geom=1.6
        )
        self.r_dry, self.n = spectral_sampling.Logarithmic(
            spectrum=spectra.Sum((self.mode_1, self.mode_2)),
            size_range=(10.633 * si.nanometre, 513.06 * si.nanometre)
        ).sample(n_sd)

        self.dt_max = dt_max

        self.dt_output = dt_output
        self.r_bins_edges = np.linspace(0 * si.micrometre, 10 * si.micrometre, 1001, endpoint=True)

        self.backend = CPU
        self.coord = 'VolumeLogarithm'
        self.adaptive = True
        self.rtol_x = condensation.default_rtol_x
        self.rtol_thd = condensation.default_rtol_thd
        self.dt_cond_range = condensation.default_cond_range

        self.T0 = 284.3 * si.kelvin
        self.q0 = 7.6 * si.grams / si.kilogram
        self.p0 = 938.5 * si.hectopascals
        self.z0 = 600 * si.metres
        self.kappa = 0.53  # Petters and S. M. Kreidenweis mean growth-factor derived

        self.t0 = 1200 * si.second
        self.f0 = 1 / 1000 * si.hertz

        self.w = 0.5 *si.metre / si.second
