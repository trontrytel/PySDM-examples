from pystrict import strict
from PySDM.physics import spectra, si
from chempy import Substance

# densities
rho_pal = 0.852 * si.g / si.cm**3  # palmitic acid density
rho_soa = 1.24 * si.g / si.cm**3  # SOA 1 density
rho1 = 1.77 * si.g / si.cm**3  # ammonium sulfate density
rho2 = 1.72 * si.g / si.cm**3  # ammonium nitrate density
M1 = Substance.from_formula("(NH4)2SO4").mass * si.gram / si.mole
M2 = Substance.from_formula("NH4NO3").mass * si.gram / si.mole


def _f_org(f_org_mass, rho_org, frac1=1.):
    frac2 = 1 - frac1
    f_org = f_org_mass / rho_org / (
            f_org_mass / rho_org +
            frac1 * (1 - f_org_mass) / rho1 +
            frac2 * (1 - f_org_mass) / rho2
    )
    return f_org


def _nu_inorg(frac1):
    frac2 = 1 - frac1
    return frac1 * (M1 / rho1) + frac2 * (M2 / rho2)


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
    f_org = _f_org(f_org_mass=.2, rho_org=rho_pal)
    nu_inorg = _nu_inorg(frac1=1)
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
    _frac1 = .5
    f_org = _f_org(f_org_mass=.668, rho_org=rho_soa, frac1=_frac1)
    nu_inorg = _nu_inorg(frac1=_frac1)
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
    f_org = _f_org(f_org_mass=.52, rho_org=rho_soa)
    nu_inorg = _nu_inorg(frac1=1)
    color = 'orangered'
