"""
Created at 02.10.2019
"""

from ..utils.widgets import IntSlider, FloatSlider, VBox, Checkbox, Accordion, Dropdown
from PySDM_examples.Arabas_et_al_2015.settings import Settings
from PySDM import physics
import numpy as np
import numba
import os
import inspect


class DemoSettings:
    __dir__ = Settings.__dir__

    def __init__(self):
        settings = Settings()

        self.ui_th_std0 = FloatSlider(description="$\\theta_0$ [K]", value=settings.th_std0, min=280, max=300)
        self.ui_qv0 = FloatSlider(description="q$_{v0}$ [g/kg]", value=settings.qv0 * 1000, min=5, max=10)
        self.ui_p0 = FloatSlider(description="p$_0$ [hPa]", value=settings.p0 / 100, min=900, max=1100)
        self.ui_kappa = FloatSlider(description="$\\kappa$ [1]", value=settings.kappa, min=0, max=1.5)
        self.ui_amplitude = FloatSlider(description="$\\psi_{_{mx}}$[kg/s/m$^{_2}$]", value=settings.rho_w_max, min=-1, max=1)
        self.ui_nx = IntSlider(value=settings.grid[0], min=10, max=100, description="nx")
        self.ui_nz = IntSlider(value=settings.grid[1], min=10, max=100, description="nz")
        self.ui_dt = FloatSlider(value=settings.dt, min=.5, max=60, description="dt (Eulerian)")
        self.ui_simulation_time = IntSlider(
            value=settings.simulation_time,
            min=1800,
            max=7200,
            description="simulation time $[s]$")
        self.ui_condensation_rtol_x = IntSlider(
            value=np.log10(settings.condensation_rtol_thd),
            min=-9,
            max=-3,
            description="log$_{10}$(rtol$_x$)")
        self.ui_condensation_rtol_thd = IntSlider(
            value=np.log10(settings.condensation_rtol_thd),
            min=-9,
            max=-3,
            description="log$_{10}$(rtol$_\\theta$)")
        self.ui_condensation_adaptive = Checkbox(
            value=settings.condensation_adaptive,
            description='condensation adaptive time-step')
        self.ui_coalescence_adaptive = Checkbox(
            value=settings.condensation_adaptive,
            description='coalescence adaptive time-step')
        self.ui_processes = [Checkbox(value=settings.processes[key], description=key) for key in settings.processes.keys()]
        self.ui_sdpg = IntSlider(value=settings.n_sd_per_gridbox, description="n_sd/gridbox", min=1, max=1000)
        self.fct_description = "MPDATA: flux-corrected transport option"
        self.tot_description = "MPDATA: third-order terms option"
        self.iga_description = "MPDATA: infinite gauge option"
        self.nit_description = "MPDATA: number of iterations (1=UPWIND)"
        self.ui_mpdata_options = [
            Checkbox(value=settings.mpdata_fct, description=self.fct_description),
            Checkbox(value=settings.mpdata_tot, description=self.tot_description),
            Checkbox(value=settings.mpdata_iga, description=self.iga_description),
            IntSlider(value=settings.mpdata_iters, description=self.nit_description, min=1, max=5)
        ]
        self.ui_formulae_options = [
            Dropdown(
                description=k,
                options=physics.formulae._choices(getattr(physics, k)).keys(),
                value=v.default
            )
            for k, v in inspect.signature(physics.Formulae.__init__).parameters.items()
            if k not in ('self', 'fastmath', 'seed')
        ]
        self.ui_formulae_options.append(
            Checkbox(
                value=inspect.signature(physics.Formulae.__init__).parameters['fastmath'].default,
                description='fastmath'
            )
        )
        self.ui_output_options = {
            'interval': IntSlider(description='interval [s]', value=settings.output_interval, min=30, max=60*15),
            'aerosol_radius_threshold': FloatSlider(
                description='aerosol/cloud threshold [um]',
                value=settings.aerosol_radius_threshold / physics.si.um,
                min=.1, max=1, step=.1),
            'drizzle_radius_threshold': IntSlider(
                description='cloud/drizzle threshold [um]',
                value=settings.drizzle_radius_threshold / physics.si.um,
                min=10, max=100, step=5)
        }

        # TODO #37
        self.r_bins_edges = settings.r_bins_edges
        self.size = settings.size
        self.condensation_substeps = settings.condensation_substeps
        self.condensation_dt_cond_range = settings.condensation_dt_cond_range
        self.condensation_schedule = settings.condensation_schedule
        self.kernel = settings.kernel
        self.spectrum_per_mass_of_dry_air = settings.spectrum_per_mass_of_dry_air
        self.coalescence_dt_coal_range = settings.coalescence_dt_coal_range
        self.coalescence_optimized_random = settings.coalescence_optimized_random
        self.coalescence_substeps = settings.coalescence_substeps

        for attr in ('n_sd', 'rhod', 'field_values', 'versions', 'n_spin_up', 'stream_function'):
            setattr(self, attr, getattr(settings, attr))

    @property
    def aerosol_radius_threshold(self):
        return self.ui_output_options['aerosol_radius_threshold'].value * physics.si.um

    @property
    def drizzle_radius_threshold(self):
        return self.ui_output_options['drizzle_radius_threshold'].value * physics.si.um

    @property
    def output_interval(self):
        return self.ui_output_options['interval'].value

    @property
    def formulae(self) -> physics.Formulae:
        return physics.Formulae(
            **{widget.description: widget.value for widget in self.ui_formulae_options}
        )

    @property
    def steps_per_output_interval(self) -> int:
        return int(self.output_interval / self.ui_dt.value)

    @property
    def output_steps(self) -> np.ndarray:
        return np.arange(0, self.n_steps + 1, self.steps_per_output_interval)

    @property
    def th_std0(self):
        return self.ui_th_std0.value

    @property
    def qv0(self):
        return self.ui_qv0.value/1000

    @property
    def p0(self):
        return self.ui_p0.value*100

    @property
    def kappa(self):
        return self.ui_kappa.value

    @property
    def amplitude(self):
        return self.ui_amplitude.value

    @property
    def grid(self):
        return self.ui_nx.value, self.ui_nz.value

    @property
    def dt(self):
        return self.ui_dt.value

    @property
    def n_steps(self):
        return int(self.ui_simulation_time.value / self.ui_dt.value)  # TODO #413

    @property
    def condensation_rtol_x(self):
        return 10**self.ui_condensation_rtol_x.value

    @property
    def condensation_rtol_thd(self):
        return 10**self.ui_condensation_rtol_thd.value

    @property
    def condensation_adaptive(self):
        return self.ui_condensation_adaptive.value

    @property
    def coalescence_adaptive(self):
        return self.ui_coalescence_adaptive.value

    @property
    def processes(self):
        result = {}
        for checkbox in self.ui_processes:
            result[checkbox.description] = checkbox.value
        return result

    @property
    def n_sd_per_gridbox(self):
        return self.ui_sdpg.value

    @property
    def mpdata_tot(self):
        for widget in self.ui_mpdata_options:
            if widget.description == self.tot_description:
                return widget.value
        raise Exception()

    @property
    def mpdata_fct(self):
        for widget in self.ui_mpdata_options:
            if widget.description == self.fct_description:
                return widget.value
        raise Exception()

    @property
    def mpdata_iga(self):
        for widget in self.ui_mpdata_options:
            if widget.description == self.iga_description:
                return widget.value
        raise Exception()

    @property
    def mpdata_iters(self):
        for widget in self.ui_mpdata_options:
            if widget.description == self.nit_description:
                return widget.value
        raise Exception()

    def box(self):
        layout = Accordion(children=[
            VBox([self.ui_th_std0, self.ui_qv0, self.ui_p0, self.ui_kappa, self.ui_amplitude]),
            VBox([*self.ui_processes]),
            VBox([self.ui_nx, self.ui_nz, self.ui_sdpg, self.ui_dt, self.ui_simulation_time,
                  self.ui_condensation_rtol_x, self.ui_condensation_rtol_thd,
                  self.ui_condensation_adaptive, self.ui_coalescence_adaptive,
                  *self.ui_mpdata_options]),
            VBox([*self.ui_formulae_options]),
            VBox([*self.ui_output_options.values()])
        ])
        layout.set_title(0, 'environment')
        layout.set_title(1, 'processes')
        layout.set_title(2, 'discretisation')
        layout.set_title(3, 'physics')
        layout.set_title(4, 'output')
        return layout
