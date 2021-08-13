import numpy as np
from PySDM_examples.Morrison_and_Grabowski_2007.common import Common


class StratoCumulus(Common):
    def stream_function(self, xX, zZ, _):
        X = self.size[0]
        return - self.rho_w_max * X / np.pi * np.sin(np.pi * zZ) * np.cos(2 * np.pi * xX)

    @property
    def initial_vapour_mixing_ratio_profile(self):
        return np.full(self.grid[-1], self.qv0)

    @property
    def initial_dry_potential_temperature_profile(self):
        return np.full(self.grid[-1], self.formulae.state_variable_triplet.th_dry(self.th_std0, self.qv0))
