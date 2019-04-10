import exdir.plugins.git_lfs
import pathlib
import exdir
import exdir.plugins.quantities
import quantities as pq
import neo
import numpy as np
import os


def get_data_path(action):
    action_path = action._backend.path
    project_path = action_path.parent.parent
    #data_path = action.data['main']
    data_path = str(pathlib.Path(pathlib.PureWindowsPath(action.data['main'])))

    print("Project path: {}\nData path: {}".format(project_path, data_path))
    return project_path / data_path

def load_lfp(data_path):
    f = exdir.File(str(data_path), 'r', plugins=[exdir.plugins.quantities])
    # LFP
    t_stop = f.attrs['session_duration']
    _lfp = f['processing']['electrophysiology']['channel_group_0']['LFP']
    keys = list(_lfp.keys())
    electrode_value = [_lfp[key]['data'].value.flatten() for key in keys]
    electrode_idx = [_lfp[key].attrs['electrode_idx'] for key in keys]
    sampling_rate = _lfp[keys[0]].attrs['sample_rate']
    units = _lfp[keys[0]]['data'].attrs['unit']
    LFP = np.r_[[_lfp[key]['data'].value.flatten() for key in keys]].T
    #LFP = (LFP.T - np.median(np.array(LFP), axis=-1)).T #CMR reference
    #LFP = (LFP.T - LFP[:, 0]).T # use topmost channel as reference
    LFP = LFP[:, np.argsort(electrode_idx)]

    LFP = neo.AnalogSignal(LFP,
                           units=units, t_stop=t_stop, sampling_rate=sampling_rate)
    LFP = LFP.rescale('mV')
    return LFP

def load_epochs(data_path):
    f = exdir.File(str(data_path), 'r', plugins=[exdir.plugins.quantities])
    epochs_group = f['epochs']
    epochs = []
    for group in epochs_group.values():
        if 'timestamps' in group.keys():
            epo = _read_epoch(f, group.name)
            epochs.append(epo)
        else:
            for g in group.values():
                if 'timestamps' in g.keys():
                    epo = _read_epoch(f, g.name)
                    epochs.append(epo)
    # io = neo.ExdirIO(str(data_path), plugins=[exdir.plugins.quantities, exdir.plugins.git_lfs])
    # blk = io.read_block()
    # seg = blk.segments[0]
    # epochs = seg.epochs
    return epochs


def load_spiketrains(data_path, channel_idx=None, remove_label='noise'):
    io = neo.ExdirIO(str(data_path), plugins=[exdir.plugins.quantities, exdir.plugins.git_lfs.Plugin(verbose=True)])
    blk = io.read_block()
    channels = blk.channel_indexes
    if channel_idx is None:
        blk = io.read_block()
        sptr = blk.segments[0].spiketrains
    else:
        blk = io.read_block(channel_group_idx=channel_idx)
        channels = blk.channel_indexes
        chx = channels[0]
        sptr = [u.spiketrains[0] for u in chx.units]
    if remove_label is not None:
        if 'cluster_group' in sptr[0].annotations.keys():
            sptr = [s for s in sptr if remove_label not in s.annotations['cluster_group']]
        else:
            print("Data have to be curated with phy to remove noise. Returning all spike trains")
    return sptr

def _read_epoch(exdir_file, path, cascade=True, lazy=False):
    group = exdir_file[path]
    if lazy:
        times = []
    else:
        times = pq.Quantity(group['timestamps'].data,
                            group['timestamps'].attrs['unit'])

    if "durations" in group and not lazy:
        durations = pq.Quantity(group['durations'].data, group['durations'].attrs['unit'])
    elif "durations" in group and lazy:
        durations = []
    else:
        durations = None

    if 'data' in group and not lazy:
        if 'unit' not in group['data'].attrs:
            labels = group['data'].data
        else:
            labels = pq.Quantity(group['data'].data,
                                 group['data'].attrs['unit'])
    elif 'data' in group and lazy:
        labels = []
    else:
        labels = None
    annotations = {'exdir_path': path}
    annotations.update(group.attrs.to_dict())

    if lazy:
        lazy_shape = (group.attrs['num_samples'],)
    else:
        lazy_shape = None
    epo = neo.Epoch(times=times, durations=durations, labels=labels,
                lazy_shape=lazy_shape, **annotations)

    return epo
