import datetime
import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from mne.time_frequency import psd_array_multitaper
from scipy.stats import linregress
import re
import yasa



plt.rcParams.update({'font.size': 16})

class EEGFeatureComputation: #now returns a pandas
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
        df = pd.read_csv(self.stagepath)
        df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
        sleep_stage_df = df[df['event'].str.contains('Sleep_stage_')]
        sleep_stage_df = sleep_stage_df.groupby('epoch').first().reset_index()
        sleep_stage_df['stage'] = sleep_stage_df['event'].str.replace('Sleep_stage_', '')
        return sleep_stage_df

    def filter_stages(self, results_df): #new filter stages function - filters based on a regex pattern; right now, it filters postprocessing but can be altered to filter preprocessing
        df = pd.read_csv(self.stagepath)
        pattern = r'(?i)(lights?\s*(?:out?|0|on|1))'
        matches = df['event'].str.extract(pattern, expand=False)
        match_indices = matches.dropna().index
        start_index = match_indices[0] if match_indices[0] != None else None
        end_index = match_indices[1] if match_indices[1] != None else None
        filtered_df = results_df.iloc[start_index + 1:end_index].reset_index(drop=True)
        return filtered_df
        


    def preprocess_data(self):
        raw = mne.io.read_raw_edf(self.edf_path, preload=False)
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
        return np.nanmean(psds[:, indices, :], axis=1)

    def get_band_power(self, psds_db, freqs, band_mask):
        psds = np.power(10, psds_db / 10)
        dfreq = freqs[1] - freqs[0]
        bp = np.nansum(psds[..., band_mask], axis=-1) * dfreq
        return 10 * np.log10(bp)

    def compute_band_powers(self):
        band_powers = {}
        for band, (fmin, fmax) in self.bands.items():
            band_mask = (self.freqs >= fmin) & (self.freqs <= fmax)
            band_powers[band] = {}
            for region, channels in self.channels_dict.items():
                avg_psds = self.avg(self.psds, self.epochs.ch_names, channels)
                band_powers[band][region] = self.get_band_power(avg_psds, self.freqs, band_mask)
        
        return band_powers

    def compute_slopes(self, band_powers):
        slopes = {}
        for band in self.bands.keys():
            slopes[band] = {}
            for region in self.channels_dict.keys():
                slopes[band][region] = {}
                for stage in ['W', 'N1', 'N2', 'N3', 'REM', 'NREM', 'R']:
                    if stage == 'NREM':
                        stage_mask = self.stages['stage'].isin(['N1', 'N2', 'N3'])
                    elif stage == 'R':
                        stage_mask = self.stages['stage'] == 'REM'
                    else:
                        stage_mask = self.stages['stage'] == stage
                    
                    if np.sum(stage_mask) > 1:
                        # Calculate the cumulative time in hours with gaps maintained
                        times = np.arange(len(stage_mask)) * 30 / 3600  # Convert 30-second epochs to hours
                        cumulative_time = np.cumsum(stage_mask * 30 / 3600)
                        
                        y = band_powers[band][region][stage_mask]
                        slope, _, _, _, _ = linregress(cumulative_time[stage_mask], y)
                        slopes[band][region][stage] = slope
                    else:
                        slopes[band][region][stage] = np.nan
        
        return slopes

    def create_unified_dataframe(self, band_powers, slopes):
        results = []
        for epoch in range(len(self.epochs)):
            row = {
                'epoch': epoch + 1,
                'time': (self.start_time + datetime.timedelta(seconds=30*epoch)).strftime('%H:%M:%S')
            }
            stage = self.stages.loc[self.stages['epoch'] == epoch + 1, 'stage']
            if stage.empty:
                row['stage'] = 'Unknown'
            else:
                row['stage'] = stage.values[0]
            
            for band in self.bands.keys():
                for region in self.channels_dict.keys():
                    row[f'{band}_{region}_power'] = band_powers[band][region][epoch]
                    
                    if row['stage'] in ['NREM', 'W', 'N1', 'N2', 'N3', 'REM', 'R']:
                        row[f'{band}_{region}_slope_{row["stage"]}'] = slopes[band][region][row['stage']]
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def detect_spindles(self, data, sfreq):
        """
        Detect sleep spindles using YASA and integrate results with existing data.
        
        Parameters:
        data (np.ndarray): Numpy array of shape (n_channels, n_epochs, n_times)
        sfreq (float): Sampling frequency of the EEG data
        
        Returns:
        pd.DataFrame: DataFrame containing detected spindles integrated with existing data
        """
        n_epochs, n_channels, n_times = data.shape
        spindles_df = pd.DataFrame(index=range(1, n_epochs + 1))
        spindles_df['epoch'] = spindles_df.index
        
        data_reshaped = data.reshape(n_channels, -1)
        
        # Scale data to microvolts
        data_uv = data_reshaped * 1e6
        
        for ch_idx in range(n_channels):
            ch_name = f'channel_{ch_idx}' 
            
            sp = yasa.spindles_detect(data_uv[ch_idx], sfreq)
            
            if sp is not None:
                summary = sp.summary()
                if not summary.empty:
                    summary['Start'] = pd.to_timedelta(summary['Start'], unit='s')
                    summary['Start_time'] = self.start_time + summary['Start']
                    summary['epoch'] = (summary['Start'].dt.total_seconds() / 30).astype(int) + 1
                    x=1
                    agg_dict = {}
                    for col in ['Duration', 'Amplitude', 'Frequency']:
                        if col in summary.columns:
                            agg_dict[col] = 'mean'
                        else:
                            print(f"Warning: Column '{col}' not found in spindle summary for {ch_name}")
                    
                    agg_dict['Oscillations'] = 'size'
                    
                    grouped = summary.groupby('epoch').agg(agg_dict).reset_index()
                    
                    grouped.columns = [f'{col}_{ch_name}' if col != 'epoch' else col for col in grouped.columns]
                    
                    spindles_df = pd.merge(spindles_df, grouped, on='epoch', how='left')
                else:
                    print(f"No spindles detected for channel {ch_idx}")
            else:
                print(f"Spindle detection failed for channel {ch_idx}")
        
        spindles_df = spindles_df.fillna(0)
        
        results_df = pd.merge(self.create_unified_dataframe(self.compute_band_powers(), self.compute_slopes(self.compute_band_powers())), 
                            spindles_df, on='epoch', how='left')
        
        return results_df



    def run(self):
        self.preprocess_data()
        band_powers = self.compute_band_powers()
        slopes = self.compute_slopes(band_powers)
        results_df = self.detect_spindles(self.epochs.get_data(), self.epochs.info['sfreq'])  
        filtered_results_df = self.filter_stages(results_df) #calling filtered rsults
        print("Filtered results are saved.")
        return filtered_results_df

if __name__ == "__main__":
    edf_path = "/Users/raosamvr/Downloads/sub-S0001111189359_ses-1_task-psg_eeg.edf"
    stagepath = "/Users/raosamvr/Downloads/sub-S0001111189359_ses-1_task-psg_annotations.csv"
    output_dir = "/Users/raosamvr/Downloads/Spec2/6_27"
    channels = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1']
    
    eeg_feature_computation = EEGFeatureComputation(edf_path, channels, stagepath, output_dir)
    results_df = eeg_feature_computation.run()
    results_df.to_csv(os.path.join(output_dir, 'filtered_eeg_features1.csv'), index=False)
    x=1
    print(results_df.head())