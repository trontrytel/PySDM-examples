from PySDM.physics.spectra import Lognormal


class Table:
    def __getitem__(self, item):
        return self._data[item]

    def items(self):
        return self._data.items()


    def __init__(self, dv):
        self._data = None
        self.dv = dv
