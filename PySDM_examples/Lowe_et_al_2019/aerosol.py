from pystrict import strict
from PySDM.physics import spectra, si

class _Aerosol:
    pass


@strict
class AerosolMarine(_Aerosol):
    dry_radius = spectra.Sum(
        {
            'Aitken': spectra.Lognormal(
                norm_factor=226 / si.cm ** 3,
                m_mode=19.6 * si.nm,
                s_geom=1.71
            ),
            'Accumulation': spectra.Lognormal(
                norm_factor=134 / si.cm ** 3,
                m_mode=69.5 * si.nm,
                s_geom=1.7
            ),
        }.values()
    )
    color = 'dodgerblue'


@strict
class AerosolBoreal(_Aerosol):
    dry_radius = spectra.Sum(
        {
            'Aitken': spectra.Lognormal(
                norm_factor=1100 / si.cm ** 3,
                m_mode=22.7 * si.nm,
                s_geom=1.75
            ),
            'Accumulation': spectra.Lognormal(
                norm_factor=540 / si.cm ** 3,
                m_mode=82.2 * si.nm,
                s_geom=1.62
            ),
        }.values()
    )
    color = 'yellowgreen'


@strict
class AerosolNascent(_Aerosol):
    dry_radius = spectra.Sum(
        {
            'Aitken': spectra.Lognormal(
                norm_factor=2000 / si.cm ** 3,
                m_mode=11.5 * si.nm,
                s_geom=1.71
            ),
            'Accumulation': spectra.Lognormal(
                norm_factor=30 / si.cm ** 3,
                m_mode=100 * si.nm,
                s_geom=1.70
            ),
        }.values()
    )
    color = 'orangered'
