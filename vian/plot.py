from elephant.statistics import isi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


from visualstiumulation.utils import make_orientation_trials


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
    from visualstiumulation.analysis import (compute_orientation_tuning, compute_osi,
                                             compute_dsi, compute_circular_variance
                                            )

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 2)
    trials = make_orientation_trials(trials)
    
    """ Analytical parameters """
    # Non-Weighed
    rates, orients = compute_orientation_tuning(trials)

    pref_or = orients[np.argmax(rates)]
    osi = compute_osi(rates, orients)
    rosi = compute_osi(rates, orients, relative=True)
    dsi = compute_dsi(rates, orients)
    cv = compute_circular_variance(rates, orients)

    # Weighed
    w_rates, orients = compute_orientation_tuning(trials, weigh=True, weights=weights)

    w_pref_or = orients[np.argmax(w_rates)]
    w_osi = compute_osi(w_rates, orients)
    w_rosi = compute_osi(w_rates, orients, relative=True)
    w_dsi = compute_dsi(w_rates, orients)
    w_cv = compute_circular_variance(w_rates, orients)

    title_1 = "Preferred orientation={}  Weighed PO={}\n".format(pref_or, w_pref_or)
    title_2 = "Non-weighed: OSI={:.2f}  CV={:.2f}  DSI={:.2f}  rOSI={:.2f}\n".format(osi, cv, dsi, rosi)
    title_3 = "Weighed:     OSI={:.2f}  CV={:.2f}  DSI={:.2f}  rOSI={:.2f}".format(w_osi, w_cv, w_dsi, w_rosi)

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
