import numpy as np
from PySDM_examples.Morrison_and_Grabowski_2007.common import Common
from pystrict import strict
from PySDM.physics import si
from PySDM.backends.numba.conf import JIT_FLAGS


class Cumulus(Common):
    def __init__(self, fastmath: bool = JIT_FLAGS['fastmath']):
        super().__init__(fastmath)
        self.size = (9 * si.km, 2.7 * si.km)
        self.hx = 1.8 * si.km
        self.x0 = 3.6 * si.km
        self.xc = self.size[0]/2
        self.grid = tuple(s//50*si.m for s in self.size)
        self.dt = 1 * si.s
        self.simulation_time = 60 * si.min

    @staticmethod
    def z0(z):
        return 0 if z <= 1.7 * si.km else 0.7 * si.km

    @staticmethod
    def hz(z):
        return 3.4 * si.km if z <= 1.7 * si.km else 2.0 * si.km

    @staticmethod
    def alpha(x):
        return 1 if abs(x-self.xc) <= 0.9 * si.km else 0

    @staticmethod
    def beta(x):
        return 1 if x <= 5.4 * si.km else -1

    @staticmethod
    def A1(t):
        if t <= 900 * si.s:
            return 5.73e2
        elif t <= 1500 * si.s:
            return 5.73e2 + 2.02e3 * (1 + np.cos(np.pi*(t-900)/600 + 1))
        else:
            return 1.15e3 + 1.72e3 * (1 + np.cos(np.pi*(min(t,2400)-1500)/900 + 1))

    @staticmethod
    def A2(t):
        if t <= 300 * si.s:
            return 0
        elif t <= 1500 * si.s:
            return 6e2 * (1 + np.cos(np.pi * (t - 300)/600 - 1))
        else:
            return 5e2 * (1 + np.cos(np.pi * (min(2400, t) - 1500)/900 - 1))

    # see Appendix (page 2859)
    def stream_function(self, xX, zZ, t):
        x = xX * self.size[0]
        z = zZ * self.size[-1]
        return (
                - self.A1(t) * np.sin(self.beta(x) * np.pi * (z-self.z0(z))/hz(z)) * np.cos(self.alpha(x) * np.pi * (x-self.x0)/hx)
                + self.A2(t)/2*np.pow(zZ, 2)
        )