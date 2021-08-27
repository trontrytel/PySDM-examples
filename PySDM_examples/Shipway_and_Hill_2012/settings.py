import numpy as np
from PySDM.physics import si
import PySDM.physics.constants as const
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from PySDM.dynamics import condensation
from PySDM.physics import Formulae, spectra
from pystrict import strict


@strict
class Settings:
    def __init__(self, n_sd_per_gridbox: int, w_1: float = 2*si.m/si.s, dt: float = 1*si.s,
                 dz: float = 25*si.m, precip: bool = True):
        self.formulae = Formulae()
        self.n_sd_per_gridbox = n_sd_per_gridbox
        self.kappa = .9  # TODO #414: not in the paper
        self.wet_radius_spectrum_per_mass_of_dry_air = spectra.Lognormal(
            norm_factor=50/si.cm**3,  # TODO #414: / self.rho,
            m_mode=.08/2 * si.um,
            s_geom=1.4
        )

        self.dt = dt
        self.dz = dz
        self.precip = precip

        self.z_max = 3000 * si.metres
        self.t_max = 60 * si.minutes

        t_1 = 600 * si.s
        self.w = lambda t: w_1 * np.sin(np.pi * t/t_1) if t < t_1 else 0

        self._th = interp1d((0, 740, 3260), (297.9, 297.9, 312.66))
        self.qv = interp1d((0, 740, 3260), (.015, .0138, .0024))  # TODO #414: is initial particle water included in initial qv? (q1 logic)
        self.thd = lambda z: self.formulae.state_variable_triplet.th_dry(self._th(z), self.qv(z))

        p0 = 975 * si.hPa  # TODO #414: not in the paper?
        g = const.g_std
        self.rhod0 = self.formulae.state_variable_triplet.rho_d(p0, self.qv(0), self._th(0))

        def drhod_dz(z, rhod):
            T = self.formulae.state_variable_triplet.T(rhod[0], self.thd(z))
            p = self.formulae.state_variable_triplet.p(rhod[0], T, self.qv(z))
            lv = self.formulae.latent_heat.lv(T)
            return self.formulae.hydrostatics.drho_dz(g, p, T, self.qv(z), lv)

        z_points = np.arange(0, self.z_max, self.dz / 2)
        rhod_solution = solve_ivp(
            fun=drhod_dz,
            t_span=(0, self.z_max),
            y0=np.asarray((self.rhod0,)),
            t_eval=z_points
        )
        assert rhod_solution.success
        self.rhod = interp1d(z_points, rhod_solution.y[0])

        self.mpdata_settings = {'n_iters': 3, 'iga': True, 'fct': True, 'tot': True}
        self.condensation_rtol_x = condensation.default_rtol_x
        self.condensation_rtol_thd = condensation.default_rtol_thd
        self.condensation_adaptive = True
        self.coalescence_adaptive = True

        self.r_bins_edges = np.logspace(np.log10(0.001 * si.um), np.log10(100 * si.um), 101, endpoint=True)
        self.cloud_water_radius_range = [1 * si.um, 50 * si.um]
        self.rain_water_radius_range = [50 * si.um, np.inf * si.um]

    @property
    def n_sd(self):
        return self.nz * self.n_sd_per_gridbox

    @property
    def nz(self):
        nz = self.z_max / self.dz
        assert nz == int(nz)
        return int(nz)

    @property
    def nt(self):
        nt = self.t_max / self.dt
        assert nt == int(nt)
        return int(nt)
