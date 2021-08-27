"""
Created at 20.08.2020
"""

from PySDM_examples.Arabas_and_Shima_2017.simulation import Simulation
from PySDM_examples.Arabas_and_Shima_2017.settings import setups
from PySDM.backends.numba.test_helpers import bdf
from PySDM.backends import CPU, GPU
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection


def data(n_output, rtols, schemes, setups_num):
    resultant_data = {}
    for scheme in schemes:
        resultant_data[scheme] = {}
        if scheme == 'BDF':
            for rtol in rtols:
                resultant_data[scheme][rtol] = []
            for settings_idx in range(setups_num):
                settings = setups[settings_idx]
                settings.n_output = n_output
                simulation = Simulation(settings)
                bdf.patch_core(simulation.core)
                results = simulation.run()
                for rtol in rtols:
                    resultant_data[scheme][rtol].append(results)
        else:
            for rtol in rtols:
                resultant_data[scheme][rtol] = []
                for settings_idx in range(setups_num):
                    settings = setups[settings_idx]
                    settings.rtol_x = rtol
                    settings.rtol_thd = rtol
                    settings.n_output = n_output
                    simulation = Simulation(settings, backend=CPU if scheme=='CPU' else GPU)
                    results = simulation.run()
                    resultant_data[scheme][rtol].append(results)
    return resultant_data


def add_color_line(fig, ax, x, y, z):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    z = np.array(z)
    vmin = min(np.nanmin(z), np.nanmax(z)/2)
    lc = LineCollection(segments, cmap=plt.get_cmap('plasma'),
                        norm=matplotlib.colors.LogNorm(vmax=1, vmin=vmin))
    lc.set_array(z)
    lc.set_linewidth(3)

    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax)


def plot(data, rtols, schemes, setups_num, path=None):
    _rtol = '$r_{tol}$'

    fig, axs = plt.subplots(setups_num, len(rtols),
                            sharex=True, sharey=True, figsize=(10, 13))
    for settings_idx in range(setups_num):
        BDF_S = None
        PySDM_S = None
        for rtol_idx in range(len(rtols)):
            ax = axs[settings_idx, rtol_idx]
            for scheme in schemes:
                datum = data[scheme][rtols[rtol_idx]][settings_idx]
                S = datum['S']
                z = datum['z']
                dt = datum['dt_cond_min']
                if scheme == 'BDF':
                    ax.plot(S, z, label=scheme, color='grey')
                    BDF_S = np.array(S)
                else:
                    add_color_line(fig, ax, S, z, dt)
                    PySDM_S = np.array(S)
            if BDF_S is not None and PySDM_S is not None:
                mae = np.mean(np.abs(BDF_S - PySDM_S))
                ax.set_title(f"MAE: {mae:.4E}")
            ax.set_xlim(-7.5e-3, 7.5e-3)
            ax.set_ylim(0, 180)
            ax.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            ax.grid()
            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    for i, ax in enumerate(axs[:, 0]):
        ax.set(ylabel=r'$\bf{settings: ' + str(i) + '}$\ndisplacement [m]')
    for i, ax in enumerate(axs[-1, :]):
        ax.set(xlabel='supersaturation\n' + r'$\bf{r_{tol}: ' + str(rtols[i]) + '}$')

    plt.tight_layout()

    if path is not None:
        plt.savefig(path + '.pdf', format='pdf')
    plt.show()


def main(save=None):
    rtols = [1e-7, 1e-11]
    schemes = ['CPU', 'BDF']
    setups_num = len(setups)
    input_data = data(80, rtols, schemes, setups_num)
    plot(input_data, rtols, schemes, setups_num, save)


if __name__ == '__main__':
    main('BDF_VS_ADAPTIVE')
