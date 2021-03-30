"""
Created at 25.11.2019
"""

from PySDM_examples.Yang_et_al_2018.settings import Settings
from PySDM_examples.Yang_et_al_2018.simulation import Simulation


if __name__ == '__main__':
    Simulation(settings=Settings()).run()
