"""
Created at 25.11.2019
"""

from PySDM_examples.SURF_activation.settings import Settings
from PySDM_examples.SURF_activation.simulation import Simulation


if __name__ == '__main__':
    Simulation(settings=Settings()).run()
