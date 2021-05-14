from PySDM_examples.Arabas_et_al_2015.netcdf_exporter import NetCDFExporter
from PySDM_examples.Arabas_et_al_2015.storage import Storage
from PySDM_examples.Arabas_et_al_2015.demo_settings import DemoSettings
from PySDM_examples.Arabas_et_al_2015.simulation import Simulation
from PySDM_examples.utils.temporary_file import TemporaryFile
from PySDM_examples.utils.widgets import IntSlider
from PySDM.backends import CPU
from scipy.io import netcdf
import ast


def test_Arabas_et_al_2015_export():
    # Arrange
    settings = DemoSettings()
    settings.ui_simulation_time = IntSlider(value=10)
    settings.ui_dt = IntSlider(value=10)
    settings.ui_output_options['interval'] = IntSlider(value= settings.ui_dt.value)
    assert settings.n_steps == 1
    assert len(settings.output_steps) == 2 and settings.output_steps[-1] == 1

    storage = Storage()
    simulator = Simulation(settings=settings, storage=storage, backend=CPU)
    file = TemporaryFile()
    exporter = NetCDFExporter(storage=storage, settings=settings, simulator=simulator, filename=file.absolute_path)


    # Act
    simulator.reinit()
    simulator.run()
    exporter.run()

    # Assert
    versions = netcdf.netcdf_file(file.absolute_path).versions
    assert 'PyMPDATA' in str(versions)
