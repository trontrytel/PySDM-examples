from .table import Table
from PySDM.physics.spectra import Lognormal
from PySDM.physics import si


class Table2(Table):
    def label(self, key):
        return f"r={self[key]['cooling_rate']/(si.K/si.min)} K/min"

    def __init__(self, *, dv=1*si.cm**3):
        super().__init__(dv)
        self._data = {
            'Cr1': {
                'ISA': Lognormal(norm_factor=1000 / dv, s_geom=10, m_mode=1e-5*si.cm**2),
                'cooling_rate': .5 * si.K / si.min,
                'color': 'orange'
            },
            'Cr2': {
                'ISA': Lognormal(norm_factor=1000 / dv, s_geom=10, m_mode=1e-5*si.cm**2),
                'cooling_rate': 5 * si.K / si.min,
                'color': 'blue'
            }
        }
