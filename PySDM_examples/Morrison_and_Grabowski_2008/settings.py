from PySDM_examples.Morrison_and_Grabowski_2007.cumulus import Cumulus
from PySDM.physics import si
from pystrict import strict


@strict
class ColdCumulus(Cumulus):
    @property
    def p0(self):
        return 1013 * si.hPa

    @property
    def kappa(self):
        return 1

    @property
    def spectrum_per_mass_of_dry_air(self):
        return None
