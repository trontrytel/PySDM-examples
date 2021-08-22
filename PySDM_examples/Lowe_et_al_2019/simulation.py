from PySDM.environments import Parcel
from PySDM import Builder
from PySDM.backends import CPU
from PySDM.physics import si
from PySDM.dynamics import AmbientThermodynamics, Condensation
import PySDM.products as PySDM_products
import numpy as np


class Simulation:
    def __init__(self, settings, products=None):
        env = Parcel(dt=settings.dt, mass_of_dry_air=settings.mass_of_dry_air, p0=settings.p0, q0=settings.q0,
                     T0=settings.T0, w=settings.w, g=settings.g)

        builder = Builder(n_sd=settings.n_sd, backend=CPU, formulae=settings.formulae)
        builder.set_environment(env)

        attributes = env.init_attributes(
            n_in_dv=settings.n_in_dv,
            kappa=settings.kappa,
            r_dry=settings.r_dry
        )

        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(Condensation())

        products = products or (
            # PySDM_products.RelativeHumidity(),
            PySDM_products.WaterMixingRatio(name='ql', description_prefix='liquid', radius_range=[1*si.um, np.inf]),
            PySDM_products.ParcelDisplacement(),
            # PySDM_products.Pressure(),
            # PySDM_products.Temperature(),
            # PySDM_products.DryAirDensity(),
            # PySDM_products.WaterVapourMixingRatio(),
            PySDM_products.Time(),
            # PySDM_products.TotalDryMassMixingRatio(settings.DRY_RHO),
            PySDM_products.PeakSupersaturation(),
            PySDM_products.CloudDropletConcentration(radius_range=settings.cloud_radius_range),
            PySDM_products.AerosolConcentration(radius_threshold=settings.cloud_radius_range[0]),
            PySDM_products.ParticlesWetSizeSpectrum(radius_bins_edges=settings.wet_radius_bins_edges),
            PySDM_products.ParticlesDrySizeSpectrum(radius_bins_edges=settings.dry_radius_bins_edges),
        )

        self.core = builder.build(attributes=attributes, products=products)
        self.settings = settings

    def _save(self, output):
        for k, v in self.core.products.items():
            value = v.get()
            if isinstance(value, np.ndarray) and value.size == 1:
                value = value[0]
            output[k].append(value)

    def run(self):
        output = {k: [] for k in self.core.products.keys()}
        self._save(output)
        for _ in range(0, self.settings.nt+1, self.settings.steps_per_output_interval):
            self.core.run(steps=self.settings.steps_per_output_interval)
            self._save(output)
        return output
