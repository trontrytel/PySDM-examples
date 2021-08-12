from PySDM.physics import si, Formulae, constants as const
from PySDM.dynamics import condensation, coalescence
import numpy, numpy as np
from PySDM.physics.coalescence_kernels import Geometric
import PySDM, PyMPDATA
import numba
import scipy


class Common:
    def __init__(self, fastmath):
        self.formulae = Formulae(condensation_coordinate='VolumeLogarithm', fastmath=fastmath)
        self.g = const.g_std

        self.condensation_rtol_x = condensation.default_rtol_x
        self.condensation_rtol_thd = condensation.default_rtol_thd
        self.condensation_adaptive = True
        self.condensation_substeps = -1
        self.condensation_dt_cond_range = condensation.default_cond_range
        self.condensation_schedule = condensation.default_schedule

        self.coalescence_adaptive = True
        self.coalescence_dt_coal_range = coalescence.default_dt_coal_range
        self.coalescence_optimized_random = True
        self.coalescence_substeps = 1
        self.kernel = Geometric(collection_efficiency=1)

        self.n_sd_per_gridbox = 20
        self.aerosol_radius_threshold = .5 * si.micrometre
        self.drizzle_radius_threshold = 25 * si.micrometre
        self.r_bins_edges = np.logspace(np.log10(0.001 * si.micrometre),
                                        np.log10(100 * si.micrometre),
                                        101, endpoint=True)
        self.output_interval = 1 * si.minute
        self.spin_up_time = 0

        self.processes = {
            "particle advection": True,
            "fluid advection": True,
            "coalescence": True,
            "condensation": True,
            "sedimentation": True,
            # "relaxation": False  # TODO #338
        }

        self.mpdata_iters = 2
        self.mpdata_iga = True
        self.mpdata_fct = True
        self.mpdata_tot = True

        key_packages = (PySDM, PyMPDATA, numba, numpy, scipy)
        self.versions = {}
        for pkg in key_packages:
            try:
                self.versions[pkg.__name__] = pkg.__version__
            except AttributeError:
                pass
        self.versions = str(self.versions)

    def rhod(self, zZ):
        p = self.formulae.hydrostatics.p_of_z_assuming_const_th_and_qv(self.g, self.p0, self.th_std0, self.qv0, z=zZ * self.size[-1])
        rhod = self.formulae.state_variable_triplet.rho_d(p, self.qv0, self.th_std0)
        return rhod

    @property
    def n_steps(self) -> int:
        return int(self.simulation_time / self.dt)  # TODO #413

    @property
    def steps_per_output_interval(self) -> int:
        return int(self.output_interval / self.dt)

    @property
    def n_spin_up(self) -> int:
        return int(self.spin_up_time / self.dt)

    @property
    def output_steps(self) -> np.ndarray:
        return np.arange(0, self.n_steps + 1, self.steps_per_output_interval)

    @property
    def n_sd(self):
        return self.grid[0] * self.grid[1] * self.n_sd_per_gridbox
