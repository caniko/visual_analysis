import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from elephant.statistics import isi
import seaborn as sns
import numpy as np

from visualstiumulation.utils import make_orientation_trials
from visualstiumulation.plot import polar_tuning_curve, plot_raster
from visualstiumulation.analysis import (compute_orientation_tuning, compute_osi,
                                         compute_dsi, compute_circular_variance
                                        )


def plot_tuning_overview(trials, unit_spiketrain, spontan_rate=None, weights=(1, 0.6)):
    """
    Makes orientation tuning plots (line and polar plot)
    for each stimulus orientation.

    Parameters
    ----------
    trials : list
        list of neo.SpikeTrain
    spontan_rates : defaultdict(dict), optional
        rates[channel_index_name][unit_id] = spontaneous firing rate trials.
    """

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 2)
    trials = make_orientation_trials(trials)
    
    """ Analytical parameters """
    # Non-Weighed
    rates, orients = compute_orientation_tuning(trials)

    pref_or = orients[np.argmax(rates)]
    osi = compute_osi(rates, orients)
    n_osi = compute_osi(rates, orients, normalise=True)
    dsi = compute_dsi(rates, orients)
    cv = compute_circular_variance(rates, orients)

    # Weighed
    w_rates, orients = compute_orientation_tuning(trials, weigh=True, weights=weights)

    w_pref_or = orients[np.argmax(w_rates)]
    w_osi = compute_osi(w_rates, orients)
    w_n_osi = compute_osi(w_rates, orients, normalise=True)
    w_dsi = compute_dsi(w_rates, orients)
    w_cv = compute_circular_variance(w_rates, orients)

    title_1 = "Preferred orientation={}  Weighed PO={}\n".format(pref_or, w_pref_or)
    title_2 = "Non-weighed: OSI={:.2f}  nOSI={:.2f}  CV={:.2f}  DSI={:.2f}\n".format(osi, n_osi, cv, dsi)
    title_3 = "Weighed:     OSI={:.2f}  nOSI={:.2f}  CV={:.2f}  DSI={:.2f}".format(w_osi, w_n_osi, w_cv, w_dsi)

    fig.suptitle(title_1 + title_2 + title_3, fontsize=17)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(orients, rates, "-o", label="with bkg")
    ax1.set_xticks(orients.magnitude)
    ax1.set_xlabel("Orientation angle (deg)")
    ax1.set_ylabel("Rate (Hz)")
    if spontan_rate is not None:
        ax1.plot(orients, rates - spontan_rate, "-o", label="without bkg")
        ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1], projection="polar")
    polar_tuning_curve(orients.rescale("rad"), rates, ax=ax2)

    ax3 = fig.add_subplot(gs[1, :])
    sns.distplot(isi(unit_spiketrain), ax=ax3)

    fig.tight_layout(rect=[0, 0.03, 1, 0.92])

    return fig


def orient_raster_plots(trials):
    """
    Makes raster plot for each stimulus orientation

    Parameters
    ----------
    trials : list
        list of neo.SpikeTrain
    """
    m_orient_trials = make_orientation_trials(trials)
    orients = list(m_orient_trials.keys())

    col_count = 4
    row_count = int(np.ceil(len(m_orient_trials)/col_count))
    fig, ax = plt.subplots(row_count, col_count, figsize=(10*col_count, 4*row_count))

    i = 0
    for r in range(0, row_count):
        for c in range(col_count):
            orient = orients[i]
            orient_trials = m_orient_trials[orient]

            ax[r, c] = plot_raster(orient_trials, ax=ax[r, c])
            ax[r, c].set_title(orient)
            ax[r, c].grid(False)

            i += 1
    
    plt.tight_layout()
    return fig
