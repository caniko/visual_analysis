from elephant.statistics import isi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np


def plot_spiketrain_isi(trials, height=0.5, ax=None):
    """
    Plot median and mean of interspike intervals in spiketrain
    Parameters
    ----------
    trials : list of neo.SpikeTrains
    heigth: float. Thickness of bars
    ax : matplotlib axes
    Returns
    -------
    out : axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    trial_mean_isis = []
    trial_median_isis = []
    trial_std = []
    x_axis = []
    for i, sptr in enumerate(trials):
        if len(sptr) >= 4:
            sptr_isi = isi(sptr)
            trial_median_isis.append(np.median(sptr_isi).magnitude)
            trial_mean_isis.append(np.mean(sptr_isi).magnitude)
            trial_std.append(np.std(sptr_isi).magnitude)
            x_axis.append(i+1)
        elif 0 <= len(sptr) < 4:
            pass
        else:
            msg = "Something went wrong len(<sptr>) is negative"
            raise RuntimeError(msg)
    
    x_axis = np.array(x_axis)
    with sns.axes_style("whitegrid"):
        median = ax.barh(x_axis-height/2, trial_median_isis, height=height*1.03, color='b', align='center')
        mean = ax.barh(x_axis+height/2, trial_mean_isis, height=height*1.03, xerr=trial_std, color='r', align='center')
        ax.legend(('Median', 'Mean'))
    return ax


def plot_waveforms(sptr, color='r', fig=None, title='waveforms', lw=2, gs=None):
    """
    Visualize waveforms on respective channels
    Parameters
    ----------
    sptr : neo.SpikeTrain
    color : color of waveforms
    title : figure title
    fig : matplotlib figure
    Returns
    -------
    out : fig
    """
    import matplotlib.gridspec as gridspec
    
    nrc = sptr.waveforms.shape[1]
    if fig is None:
        fig = plt.figure()
        fig.suptitle(title)
    axs = []
    for c in range(nrc):
        if gs is None:
            ax = fig.add_subplot(1, nrc, c+1, sharex=ax, sharey=ax)
        else:
            gs0 = gridspec.GridSpecFromSubplotSpec(1, nrc, subplot_spec=gs)
            ax = fig.add_subplot(gs0[:, c], sharex=ax, sharey=ax)
        axs.append(ax)
    for c in range(nrc):
        wf = sptr.waveforms[:, c, :]
        m = np.mean(wf, axis=0)
        stime = np.arange(m.size, dtype=np.float32)/sptr.sampling_rate
        stime.units = 'ms'
        sd = np.std(wf, axis=0)
        axs[c].plot(stime, m, color=color, lw=lw)
        axs[c].fill_between(stime, m-sd, m+sd, alpha=.1, color=color)
        if sptr.left_sweep is not None:
            sptr.left_sweep.units = 'ms'
            axs[c].axvspan(sptr.left_sweep, sptr.left_sweep, color='k',
                           ls='--')
        axs[c].set_xlabel(stime.dimensionality)
        axs[c].set_xlim([stime.min(), stime.max()])
        if c > 0:
            plt.setp(axs[c].get_yticklabels(), visible=False)
    axs[0].set_ylabel(r'amplitude $\pm$ std [%s]' % wf.dimensionality)

    return fig
