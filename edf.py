import datetime
import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from mne.time_frequency import psd_array_multitaper
from sklearn.linear_model import LinearRegression

plt.rcParams.update({'font.size': 16})

class EEGFeatureComputation:
    def __init__(self, edf_path, channels, stagepath, output_dir):
        self.edf_path = edf_path
        self.channels = channels
        self.stagepath = stagepath
        self.output_dir = output_dir
        self.bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'sigma': (11, 16),
            'beta': (12, 30),
            'slow_oscillation': (0.3, 1)
        }
        self.channels_dict = {
            'central': ['C3-M2', 'C4-M1'],
            'frontal': ['F3-M2', 'F4-M1'],
            'occipital': ['O1-M2', 'O2-M1']
        }
        self.psds = None
        self.freqs = None
        self.epochs = None
        self.start_time = None
        self.stages = self.load_stages()

    def load_stages(self):
        # Load sleep stages
        annotations_df = pd.read_csv(self.stagepath)
        annotations_df['time'] = pd.to_datetime(annotations_df['time'], format='%H:%M:%S').dt.time
        sleep_stage_df = annotations_df[annotations_df['event'].str.contains('Sleep_stage_')]  # isolating only the sleep 'stages'
        sleep_stage_df = sleep_stage_df.groupby('epoch').first().reset_index()
        return sleep_stage_df

    def preprocess_data(self):
        raw = mne.io.read_raw_edf(self.edf_path, preload=False)  # do this to get raw.ch_names
        raw = mne.io.read_raw_edf(self.edf_path, preload=True, exclude=[x for x in raw.ch_names if x not in self.channels])
        raw.notch_filter(freqs=60)
        raw.filter(l_freq=0.3, h_freq=35)
        
        sfreq = raw.info['sfreq']
        events = mne.make_fixed_length_events(raw, id=1, duration=30)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=30 - 1/raw.info['sfreq'], baseline=None, preload=True)
        
        epochs_data = epochs.get_data(picks=self.channels) * 1e6
        self.psds, self.freqs = psd_array_multitaper(epochs_data, sfreq=sfreq, fmin=0.3, fmax=35, n_jobs=1, bandwidth=0.5)
        self.psds = 10 * np.log10(self.psds)
        self.epochs = epochs
        self.start_time = raw.info['meas_date']

    def avg(self, psds, channel_names, channels):
        indices = [channel_names.index(ch) for ch in channels if ch in channel_names]
        if not indices:
            print(f"channel {channels} not found in data")
        else:
            print(f"channel {channels} found at indices {indices}")
        return np.nanmean(psds[:, indices, :], axis=1)  # use nanmean instead of mean to ignore nan (due to log(0))

    def spect(self, psds, freqs, title, epoch_length, start_time, output_file):
        plt.close()
        plt.figure(figsize=(14, 5))
        
        # Average the PSDs across channels for each epoch
        avg_psds = np.nanmean(psds, axis=1)
        
        T = avg_psds.shape[0] * epoch_length / 3600
        plt.imshow(avg_psds.T, aspect='auto', origin='lower', extent=[0, T, freqs[0], freqs[-1]], cmap='turbo', vmin=15, vmax=30)
        
        xticks = np.arange(int(T) + 1)
        xticklabels = [(start_time + datetime.timedelta(hours=int(x))).strftime('%H:%M') for x in xticks]

        plt.colorbar(label='Power Spectral Density (dB/Hz)')
        plt.title(title)
        plt.xticks(xticks, labels=xticklabels)
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (hours)')
        plt.savefig(output_file, dpi=500, bbox_inches='tight')


    def get_band_power(self, psds_db, freqs, band_mask):
        """
        psds_db: PSD in decibel/Hz
        freqs: the frequency bins
        band_mask: a boolean array to indicate the band of interest
        """
        # for band power, one must convert decibel/Hz back to uV^2/Hz, since power is defined in uV^2
        psds = np.power(10, psds_db / 10)
        
        # band power (bp) is defined as the area under the spectrum, we should use rectangle approximation to calculate the area under curve
        # each rectangle area is width x height = frequency bin width x power density at that frequency bin
        dfreq = freqs[1] - freqs[0]  # frequency bin width, assumes constant interval
        bp = np.nansum(psds[..., band_mask], axis=-1) * dfreq
        
        # then convert band power to decibel (log-scale) for better numerical behavior
        return 10 * np.log10(bp)

    def compute_band_powers(self):
        band_powers = {}
        for band, (fmin, fmax) in self.bands.items():
            band_mask = (self.freqs >= fmin) & (self.freqs <= fmax)
            band_powers[band] = {}
            for region, channels in self.channels_dict.items():
                avg_psds = self.avg(self.psds, self.epochs.ch_names, channels)
                band_powers[band][region] = self.get_band_power(avg_psds, self.freqs, band_mask)
        
        sleep_stages = self.stages['event'].values
        
        # Compute overnight slopes - only care about the nrem, gaps in the nrem range - time axis should contain the gap - when x is created the gap in x should be maintain
        # we just need a plain old simple lin reg - adjust the coefficient - might've been removed
        overnight_slopes = {}
        for band in self.bands.keys():
            overnight_slopes[band] = {}
            for region in self.channels_dict.keys():
                nrem_mask = (sleep_stages == 'Sleep_stage_N1') | (sleep_stages == 'Sleep_stage_N2') | (sleep_stages == 'Sleep_stage_N3')
                if np.any(nrem_mask):
                    X = (np.arange(np.sum(nrem_mask)).reshape(-1, 1))/(3600)
                    y = band_powers[band][region][nrem_mask]
                    model = LinearRegression().fit(X, y)
                    overnight_slopes[band][region] = model.coef_[0]
        
        return band_powers, sleep_stages, overnight_slopes

    def plot_alpha(self, alpha_power, title):
        plt.close()  # moved from bottom
        plt.figure()
        plt.plot(alpha_power)
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Alpha Power (8-12Hz) (dB)')
        output_file = os.path.join(self.output_dir, title.replace(' ', '_').lower() + '.png')
        plt.savefig(output_file, dpi=500)

    def save_results(self, band_powers, sleep_stages, overnight_slopes):
        # Save results to CSV
        results = {'epoch': np.arange(1, len(self.epochs) + 1)}
        for band in self.bands.keys():
            for region in self.channels_dict.keys():
                results[f'{band}_{region}'] = band_powers[band][region]
                results[f'{band}_{region}_slope'] = overnight_slopes[band][region]
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, 'band_powers1.csv'), index=False)
        
        alpha_band = (self.freqs >= 8) & (self.freqs <= 12)
        central_alpha = self.get_band_power(self.avg(self.psds, self.epochs.ch_names, self.channels_dict['central']), self.freqs, alpha_band)
        self.plot_alpha(central_alpha, 'Central Alpha Power')
        
        stage_labels = self.stages['event'].unique()
        stage_powers = {}
        
        for stage in stage_labels:
            stage_mask = (sleep_stages == stage)
            if np.any(stage_mask):
                stage_powers[stage] = {}
                for band in self.bands.keys():
                    stage_powers[stage][band] = {}
                    for region in self.channels_dict.keys():
                        stage_powers[stage][band][region] = np.nanmean(band_powers[band][region][stage_mask])
        
        stage_results = {'stage': stage_labels}
        for band in self.bands.keys():
            for region in self.channels_dict.keys():
                stage_results[f'{band}_{region}'] = [stage_powers[stage][band][region] for stage in stage_labels]
        
        stage_results_df = pd.DataFrame(stage_results)
        stage_results_df.to_csv(os.path.join(self.output_dir, 'stage_band_powers.csv'), index=False)

    def run(self):
        self.preprocess_data()
        band_powers, sleep_stages, overnight_slopes = self.compute_band_powers()
        self.spect(self.psds, self.freqs, 'Spectrogram', 30, self.start_time, os.path.join(self.output_dir, 'spectrogram.png'))
        self.save_results(band_powers, sleep_stages, overnight_slopes)
        print("All computations are completed and results are saved.")

if __name__ == "__main__":
    edf_path = "/Users/raosamvr/Downloads/sub-S0001111189359_ses-1_task-psg_eeg.edf"
    stagepath = "/Users/raosamvr/Downloads/sub-S0001111189359_ses-1_task-psg_annotations.csv"
    output_dir = "/Users/raosamvr/Downloads/Spec2/6_27"
    channels = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1']
    
    eeg_feature_computation = EEGFeatureComputation(edf_path, channels, stagepath, output_dir)
    eeg_feature_computation.run()
