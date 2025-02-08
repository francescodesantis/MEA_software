from PyQt5.QtCore import pyqtSignal, QThread
from electrode_config import ElectrodeConfig
import pandas as pd
import numpy as np

# Class that computes the mean time lags with respect to reference electrode
class MeanValues(QThread):
    propagation = pyqtSignal(dict)

    def __init__(self, electrode_data, peaks, plot_canvas, window_size, percentage, parent=None):
        super().__init__(parent)
        self.electrode_data = electrode_data
        self.peaks = peaks
        self.window_size = window_size
        self.percentage = percentage
        self.pacemaker = {}
        self.electrode_config = ElectrodeConfig(self.electrode_data)  
        self.plot_canvas = plot_canvas 

    def run(self):
        self.df_normalized_list = self.compute_pacemaker()
        if not self.df_normalized_list:
            print("No valid peaks were found.")
            return
        self.mean_values = self.initial_plot()
        self.propagation.emit(self.mean_values)

    def compute_pacemaker(self):
        # Filter electrodes with less than percentage of peaks inserted by user (if th = 100, be aware of empty electrodes)
        peak_count = max([len(self.peaks[header]) for header in self.electrode_data.columns])
        filtered_headers = [header for header in self.electrode_data.columns if len(self.peaks[header]) >= (self.percentage/100)*peak_count and len(self.peaks[header]) > 0]
        if not filtered_headers:
            print("No electrodes passed the threshold criteria.")
            return []
        self.electrode_data = self.electrode_data[filtered_headers]
        self.peaks = {header: self.peaks[header] for header in filtered_headers}


        self.df_normalized_list = []
        # Find electrode with most R peaks
        self.reference_electrode = max(self.peaks, key = lambda header: len(self.peaks[header])) # Find the electrode with the most spikes 
        reference_peaks = self.peaks[self.reference_electrode]
        #print(f"Reference electrode: {reference_electrode} with reference peaks: {reference_peaks}")

        for i, ref_peak_time in enumerate(reference_peaks):
            if ref_peak_time is None:
                continue
        
            valid_r_peak_times = {self.reference_electrode: 0} # Set reference as zero value
            for header in self.electrode_data.columns:
                if header == self.reference_electrode:
                    continue
                # Find peaks from other electrodes in window size defined by user
                other_peaks = [peak for peak in self.peaks[header] if ref_peak_time - self.window_size <= peak <= ref_peak_time + self.window_size]
                if other_peaks:
                    valid_r_peak_times[header] = other_peaks[0] - ref_peak_time # Normalize peaks with respect to reference electrode
                    #print(f"Electrode {header} has a valid peak in {other_peaks[0]} normalized to {valid_r_peak_times[header]}")

            if len(valid_r_peak_times) > 1:
                # Sort electrodes by normalized peaks
                sorted_r_peaks = {k: v for k, v in sorted(valid_r_peak_times.items(), key=lambda item: item[1])}
                # Create dataframe with normalized peaks
                df_normalized = pd.DataFrame({
                    'Electrode': list(sorted_r_peaks.keys()),
                    'R_Peak_Time': list(sorted_r_peaks.values())
                })
                #print(f"Dataframe normalized for beat {i}: {df_normalized}")
                # Add coordinates to dataframe
                df_normalized['Coordinates'] = df_normalized['Electrode'].apply(self.electrode_config.header_to_coordinate)
                df_normalized[['X', 'Y']] = pd.DataFrame(df_normalized['Coordinates'].tolist(), index=df_normalized.index)
                self.df_normalized_list.append(df_normalized.dropna())
            else:
                print("No valid peaks were found.")

        if not self.df_normalized_list:
            print("No valid peaks were found.")
        
        return self.df_normalized_list
    
    def initial_plot(self):
        mean_lags = {}
        for df in self.df_normalized_list:
            for electrode, time in df[['Electrode', 'R_Peak_Time']].values:
                if electrode not in mean_lags:
                    mean_lags[electrode] = []
                mean_lags[electrode].append(time) # Append normalized peaks to list
        # Compute mean values for each electrode
        mean_values = {electrode: np.mean(times) for electrode, times in mean_lags.items()}
        return mean_values
    
# Class to compute the propagation of the signal with respect to pacemaker
class Propagation(QThread):
    propagation = pyqtSignal(list, str)

    def __init__(self, electrode_data, peaks, window_size, percentage, pacemaker, plot_canvas, parent=None):
        super().__init__(parent)
        self.electrode_data = electrode_data
        self.peaks = peaks
        self.pacemaker = pacemaker # Pacemaker electrode found in main code as first electrode in table
        self.electrode_config = ElectrodeConfig(self.electrode_data)  
        self.plot_canvas = plot_canvas 
        self.window_size = window_size
        self.percentage = percentage

    def run(self):
        self.df_normalized_list, self.pacemaker = self.compute_pacemaker()
        self.propagation.emit(self.df_normalized_list, self.pacemaker)
    
    def compute_pacemaker(self):
        self.df_normalized_list = []
        time_window = self.window_size
        # Filter electrodes with less than percentage of peaks inserted by user
        peak_count = max([len(self.peaks[header]) for header in self.electrode_data.columns])
        filtered_headers = [header for header in self.electrode_data.columns if len(self.peaks[header]) >= (self.percentage/100)*peak_count]
        self.electrode_data = self.electrode_data[filtered_headers]
        self.peaks = {header: self.peaks[header] for header in filtered_headers}
        
        self.reference_electrode = self.pacemaker # Set pacemaker electrode as reference
        reference_peaks = self.peaks[self.reference_electrode]
        #print(f"Reference electrode: {reference_electrode} with reference peaks: {reference_peaks}")

        for i, ref_peak_time in enumerate(reference_peaks):
            if ref_peak_time is None:
                continue
        
            valid_r_peak_times = {self.reference_electrode: 0} # Set reference as zero value
            for header in self.electrode_data.columns:
                if header == self.reference_electrode:
                    continue

                other_peaks = [peak for peak in self.peaks[header] if ref_peak_time - time_window <= peak <= ref_peak_time + time_window]
                if other_peaks:
                    valid_r_peak_times[header] = other_peaks[0] - ref_peak_time
                    #print(f"Electrode {header} has a valid peak in {other_peaks[0]} normalized to {valid_r_peak_times[header]}")

            if len(valid_r_peak_times) > 1:
                sorted_r_peaks = {k: v for k, v in sorted(valid_r_peak_times.items(), key=lambda item: item[1])}

                #normalized_r_peaks = {k: v for k, v in sorted_r_peaks.items()}
                df_normalized = pd.DataFrame({
                    'Electrode': list(sorted_r_peaks.keys()),
                    'R_Peak_Time': list(sorted_r_peaks.values())
                })
                #print(f"Dataframe normalized for beat {i}: {df_normalized}")
                df_normalized['Coordinates'] = df_normalized['Electrode'].apply(self.electrode_config.header_to_coordinate)
                df_normalized[['X', 'Y']] = pd.DataFrame(df_normalized['Coordinates'].tolist(), index=df_normalized.index)
                self.df_normalized_list.append(df_normalized.dropna())
            else:
                print("No valid peaks were found.")

        if not self.df_normalized_list:
            print("No valid peaks were found.")
        
        return self.df_normalized_list, self.pacemaker