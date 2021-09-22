from typing import Union
from PySDM import Builder
from PySDM.dynamics import Freezing
from PySDM.environments import Box
from PySDM.physics import constants as const, Formulae
from PySDM.physics.heterogeneous_ice_nucleation_rate import constant, abifm
from PySDM.initialisation import spectral_sampling
from PySDM.initialisation.multiplicities import discretise_n
from PySDM.products import IceWaterContent
from PySDM.backends import CPU
from PySDM.physics import si
from matplotlib import pylab
import matplotlib
from packaging import version
import numpy as np


class Simulation:
    # note: dv and droplet_volume are dummy multipliers (multiplied and then divided by)
    #       will become used if coalescence or other processes are turned on
    def __init__(self, *, cases, n_runs_per_case=10, multiplicity=1, dt, droplet_volume=1*si.um**3,
                 heterogeneous_ice_nucleation_rate='Constant',
                 total_time: Union[None, float] = None, temperature_range: Union[None, tuple] = None
                 ):
        self.cases = cases
        self.n_runs_per_case = n_runs_per_case
        self.multiplicity = multiplicity
        self.dv = cases.dv
        self.dt = dt
        self.droplet_volume = droplet_volume
        self.heterogeneous_ice_nucleation_rate = heterogeneous_ice_nucleation_rate
        self.output = None
        self.total_time = total_time
        self.temperature_range = temperature_range

    def run(self, keys):
        self.output = {}
        for key in keys:
            case = self.cases[key]

            assert (self.total_time is None) + (self.temperature_range is None) == 1
            if self.total_time is not None:
                total_time = self.total_time
            else:
                total_time = np.diff(np.asarray(self.temperature_range)) / case['cooling_rate']

            if 'J_het' not in case:
                case['J_het'] = None
            if 'cooling_rate' not in case:
                case['cooling_rate'] = 0

            self.output[key] = []
            for i in range(self.n_runs_per_case):
                number_of_real_droplets = case['ISA'].norm_factor * self.dv
                n_sd = number_of_real_droplets / self.multiplicity
                np.testing.assert_approx_equal(n_sd, int(n_sd))
                n_sd = int(n_sd)

                data = simulation(seed=i, n_sd=n_sd, dt=self.dt, dv=self.dv, spectrum=case['ISA'],
                                  droplet_volume=self.droplet_volume, multiplicity=self.multiplicity, J_het=case['J_het'],
                                  total_time=total_time, number_of_real_droplets=number_of_real_droplets,
                                  cooling_rate=self.cases[key]['cooling_rate'],
                                  heterogeneous_ice_nucleation_rate=self.heterogeneous_ice_nucleation_rate,
                                  T0=self.temperature_range[1] if self.temperature_range else np.nan)

                self.output[key].append(data)

    def plot(self, ylim, grid=None):
        pylab.rc('font', size=10)
        for key in self.output.keys():
            for run in range(self.n_runs_per_case):
                time = self.dt * np.arange(len(self.output[key][run]))
                if self.cases[key]['cooling_rate'] == 0:
                    x = time / si.min
                    y = self.output[key][run]
                else:
                    x = self.temperature_range[1] - time * self.cases[key]['cooling_rate']
                    y = 1 - np.asarray(self.output[key][run])
                pylab.step(
                    x,
                    y,
                    label=self.cases.label(key) if run == 0 else None,
                    color=self.cases[key]['color'],
                    linewidth=.666
                )
        key = None
        if version.parse(matplotlib.__version__) >= version.parse('3.3.0'):
            pylab.gca().set_box_aspect(1)
        pylab.legend()
        if grid is not None:
            pylab.grid(which=grid)
        pylab.ylim(ylim)
        if self.temperature_range:
            pylab.xlim(*self.temperature_range)
            pylab.xlabel("T / K")
            pylab.ylabel("$f_{frz}$")
        else:
            pylab.xlim(0, self.total_time / si.min)
            pylab.xlabel("t / min")
            pylab.ylabel("$f_{ufz}$")
            pylab.yscale('log')


def simulation(*, seed, n_sd, dt, dv, spectrum, droplet_volume, multiplicity, J_het, total_time,
               number_of_real_droplets, cooling_rate=0, heterogeneous_ice_nucleation_rate='Constant', T0=np.nan):
    constant.J_het = J_het
    abifm.m = 54.48
    abifm.c = -10.67
    formulae = Formulae(seed=seed, heterogeneous_ice_nucleation_rate=heterogeneous_ice_nucleation_rate)
    builder = Builder(n_sd=n_sd, backend=CPU, formulae=formulae)
    env = Box(dt=dt, dv=dv)
    builder.set_environment(env)
    builder.add_dynamic(Freezing(singular=False))

    if hasattr(spectrum, 's_geom') and spectrum.s_geom == 1:
        _isa, _conc = np.full(n_sd, spectrum.m_mode), np.full(n_sd, multiplicity / dv)
    else:
        _isa, _conc = spectral_sampling.ConstantMultiplicity(spectrum).sample(n_sd)
    attributes = {
        'n': discretise_n(_conc * dv),
        'immersed surface area': _isa,
        'volume': np.full(n_sd, droplet_volume)
    }
    np.testing.assert_almost_equal(attributes['n'], multiplicity)
    products = [IceWaterContent(specific=False)]
    particulator = builder.build(attributes=attributes, products=products)

    temperature = T0
    env['a_w_ice'] = np.nan

    cell_id = 0
    data = []
    for i in range(int(total_time / dt) + 1):
        if cooling_rate != 0:
            temperature -= cooling_rate * dt/2
            env['a_w_ice'] = particulator.formulae.saturation_vapour_pressure.a_w_ice.py_func(temperature)
        particulator.run(0 if i == 0 else 1)
        if cooling_rate != 0:
            temperature -= cooling_rate * dt/2

        ice_mass_per_volume = particulator.products['qi'].get()[cell_id]
        ice_mass = ice_mass_per_volume * dv
        ice_number = ice_mass / (const.rho_w * droplet_volume)
        unfrozen_fraction = 1 - ice_number / number_of_real_droplets
        data.append(unfrozen_fraction)
    return data
