import sys
from PySDM_examples.Arabas_et_al_2015.demo_controller import DemoController
from PySDM_examples.Arabas_et_al_2015.storage import Storage
from PySDM_examples.Arabas_et_al_2015.demo_settings import DemoSettings
from PySDM_examples.Arabas_et_al_2015.simulation import Simulation
from PySDM_examples.Arabas_et_al_2015.demo_viewer import DemoViewer
from PySDM_examples.Arabas_et_al_2015.netcdf_exporter import NetCDFExporter
from PySDM_examples.utils.temporary_file import TemporaryFile
from PySDM_examples.utils.widgets import display, Tab, VBox, HTML


def launch():
    settings = DemoSettings()
    storage = Storage()
    simulator = Simulation(settings, storage)
    temporary_file = TemporaryFile('.nc')
    exporter = NetCDFExporter(storage, settings, simulator, temporary_file.absolute_path)

    viewer = DemoViewer(storage, settings)

    controller = DemoController(simulator, viewer, exporter, temporary_file)

    tabs = Tab([VBox([controller.box(), viewer.box()]), settings.box()])
    tabs.set_title(1, "Settings")
    tabs.set_title(0, "Simulation")
    tabs.observe(controller.reinit, 'selected_index')

    # https://github.com/googlecolab/colabtools/issues/1302
    if 'google.colab' in sys.modules:
        display(HTML('''<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"> '''))
    display(tabs)
