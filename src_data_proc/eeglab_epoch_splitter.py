from mne import Epochs, find_events
from mne.datasets import sample
from mne.io import read_raw_fif


directory = sample.data_path() / "MEG" / "sample"
raw = read_raw_fif(directory / "sample_audvis_raw.fif", preload=False)
raw.pick_types(eeg=True, stim=True)
raw.load_data()

event_id = dict(left=1, right=2)
events = find_events(raw, stim_channel="STI 014")
epochs = Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=-0.2,
    tmax=0.5,
    reject=None,
    preload=True,
)