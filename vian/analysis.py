import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from elephant.statistics import isi
import seaborn as sns
import numpy as np

from visualstimulation.utils import make_orientation_trials
from visualstimulation.plot import polar_tuning_curve, plot_raster
from visualstimulation.analysis import (compute_orientation_tuning, compute_osi,
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

    col_count = 2
    row_count = int(np.ceil(len(m_orient_trials)/col_count))
    fig, ax = plt.subplots(row_count, col_count, figsize=(4*col_count, 2*row_count))

    i = 0
    for r in range(0, row_count):
        for c in range(col_count):
            orient = orients[i]
            orient_trials = m_orient_trials[orient]

            ax[r, c] = plot_raster(orient_trials, ax=ax[r, c])
            ax[r, c].set_title(orient)
            ax[r, c].grid(False)
            ax[r, c].axvline(0, color='r')

            i += 1

    plt.tight_layout()
    return fig


def spike_raster(ax, sptr, T=[0., 10.], epochs=None):
    '''
    Arguments
    ---------
    ax : matplotlib.axes._subplots.AxesSubplot
    spiketrains : list of neo.SpikeTrain objects
    T : length 2 list/tuple of floats
        time interval in seconds
    epo : None or neo.Epoch object
        show onset/offset times of stimuli
    '''
    yticklabels = []
    for i, spiketrain in enumerate(sptr):
        yticklabels.append('{} ({})'.format(spiketrain.name, spiketrain.description))
        ax.plot(spiketrain, np.zeros(spiketrain.size)+i, 'C0|')
    if epochs is not None:
        axis = ax.axis('tight')
        ax.vlines(epochs[0].times, axis[2], axis[3], 'g')
        ax.vlines((epochs[0].times+epochs[0].durations), axis[2], axis[3], 'r')
    ax.set_yticks(range(len(sptr)))
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel('unit id')
    ax.set_xlim(T)
    ax.set_xlabel('t (s)')
    ax.set_title('Spike Raster')



def plot_psth(st, epoch, fig=None, axes=None, lags=None, bin_size=None,
              marker='|', color='C0', n_trials=10,  histtype='bar'):
    '''
    Parameters:
    st : neo.SpikeTrain
    epoch : neo.Epoch
    lags : tuple of Quantity scalars
    bin_size : Quantity scalar
    color : mpl color
    n_trials : int
        number of trials to include in PSTH
    '''
    labels = np.unique(epoch.labels, axis=-1)
    bins = np.linspace(lags[0], lags[1], int((lags[1]-lags[0])//bin_size)+1)
    flattenlist = lambda lst: [item for sublist in lst for item in sublist]

    if fig is None:
        fig, axes = plt.subplots(2, len(labels), sharex=True, sharey='row', figsize=(10, 5))
        fig.suptitle('unit {} ({})'.format(st.name, st.description))
    for i, label in enumerate(labels):
        axes[0, i].set_xlim(lags)
        axes[1, i].set_xlim(lags)

        sts = []
        for h, epo in enumerate(epoch[epoch.labels == label]):
            if h < n_trials:
                st_ = st.time_slice(t_start=(epo+lags[0]).simplified,
                                    t_stop=(epo+lags[1]).simplified)
                sts.append((st_.times.simplified - epo.simplified).tolist())
                axes[0, i].plot(sts[h], np.zeros(len(sts[h])) + h, marker, color=color)

        axes[0, i].set_title('{}'.format(label), fontsize='x-small')
        axes[1, i].set_xlabel('lag (s)')
        axes[1, i].hist(flattenlist(sts), bins=bins, color=color, histtype=histtype)
        axes[1, i].axvline(0, color='r')
        axes[0, i].axvline(0, color='r')

        if i == 0:
            axes[0, i].set_ylabel('trial #')
            axes[1, i].set_ylabel('#')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, axes


# def some plotting functions
def remove_axis_junk(ax, lines=['right', 'top']):
    for loc, spine in ax.spines.items():
        if loc in lines:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def draw_lineplot(
        ax, data, dt=0.1,
        T=(0, 200),
        scaling_factor=1.,
        vlimround=None,
        label='local',
        scalebar=True,
        unit='mV',
        ylabels=True,
        color='k',
        ztransform=True,
        filter=False,
        filterargs=dict(N=2, Wn=0.02, btype='lowpass')
        ):
    ''' draw some nice lines'''

    tvec = np.arange(data.shape[1])*dt
    if T[0] < 0:
        tvec += T[0]
    try:
        tinds = (tvec >= T[0]) & (tvec <= T[1])
    except TypeError:
        print(data.shape, T)
        raise Exception

    # apply temporal filter
    if filter:
        b, a = ss.butter(**filterargs)
        data = ss.filtfilt(b, a, data, axis=-1)

    #subtract mean in each channel
    if ztransform:
        dataT = data.T - data.mean(axis=1)
        data = dataT.T

    zvec = np.arange(data.shape[0])
    vlim = abs(data[:, tinds]).max()
    if vlimround is None:
        vlimround = 2.**np.round(np.log2(vlim)) / scaling_factor
    else:
        pass

    yticklabels=[]
    yticks = []

    for i, z in enumerate(zvec):
        if i == 0:
            ax.plot(tvec[tinds], data[i][tinds] / vlimround + z, lw=.5,
                    rasterized=False, label=label, clip_on=False,
                    color=color)
        else:
            ax.plot(tvec[tinds], data[i][tinds] / vlimround + z, lw=.5,
                    rasterized=False, clip_on=False,
                    color=color)
        yticklabels.append('ch. %i' % (i+1))
        yticks.append(z)

    if scalebar:
        ax.plot([tvec[tinds][-1], tvec[tinds][-1]],
                [zvec[-1], zvec[-2]], lw=2, color='k', clip_on=False)
        ax.text(tvec[tinds][-1]+np.diff(T)*0.0, np.mean([zvec[-1], zvec[-2]]),
                '$2^{' + '{}'.format(int(np.log2(vlimround))) + '}$ ' + '{0}'.format(unit),
                color='r', rotation='vertical',
                va='center', zorder=100)

    ax.axis(ax.axis('tight'))
    if ylabels:
        ax.yaxis.set_ticks(yticks)
        ax.yaxis.set_ticklabels(yticklabels)
        #ax.set_ylabel('channel', labelpad=0.1)
    else:
        ax.yaxis.set_ticklabels([])
    remove_axis_junk(ax, lines=['right', 'top'])
    ax.set_xlabel(r'time (ms)', labelpad=0.1)

    return vlimround
