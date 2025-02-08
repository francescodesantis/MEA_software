import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QFrame
import numpy as np
from scipy import stats
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QLabel, QSlider, QCheckBox, QDialog,
                           QTableWidget, QTableWidgetItem, QSpinBox, QRadioButton, QDialogButtonBox, QMessageBox, QLineEdit)
from PyQt5.QtCore import Qt
from scipy.signal import find_peaks
from electrode_config import ElectrodeConfig
from plot_global import PlotCanvas

class IntervalSelectionDialog(QDialog):
    def __init__(self, parent=None, time_range=None):
        super().__init__(parent)
        self.setWindowTitle("Interval Selection")
        self.time_range = time_range or [0, 1000]  # Default 1 second range
        self.intervals = []
        
        layout = QVBoxLayout(self)
        
        # Intervals table
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Start (ms)", "End (ms)", ""])
        layout.addWidget(self.table)
        
        # Add interval button
        self.add_button = QPushButton("Add Interval")
        self.add_button.clicked.connect(self.add_interval)
        layout.addWidget(self.add_button)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def add_interval(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        start_spin = QSpinBox()
        start_spin.setRange(self.time_range[0], self.time_range[1])
        end_spin = QSpinBox()
        end_spin.setRange(self.time_range[0], self.time_range[1])
        
        start_spin.setValue(self.time_range[0])
        end_spin.setValue(self.time_range[1])
        
        self.table.setCellWidget(row, 0, start_spin)
        self.table.setCellWidget(row, 1, end_spin)
        
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(lambda: self.table.removeRow(self.table.row(delete_btn)))
        self.table.setCellWidget(row, 2, delete_btn)
        
    def get_intervals(self):
        intervals = []
        for row in range(self.table.rowCount()):
            start = self.table.cellWidget(row, 0).value()
            end = self.table.cellWidget(row, 1).value()
            intervals.append((start, end))
        return intervals or [(self.time_range[0], self.time_range[1])]

class PeakNameDialog(QDialog):
    def __init__(self, parent=None, custom_names=None):
        super().__init__(parent)
        self.setWindowTitle("Select Peak Name")
        layout = QVBoxLayout(self)
        
        # Standard peak names
        self.standard_names = ['P', 'Q', 'R', 'S', 'T']
        
        # Radio buttons for standard names
        self.name_group = QWidget()
        name_layout = QVBoxLayout(self.name_group)
        self.radio_buttons = []
        
        for name in self.standard_names:
            radio = QRadioButton(name)
            self.radio_buttons.append(radio)
            name_layout.addWidget(radio)
        
        layout.addWidget(self.name_group)
        
        # Custom name option
        self.custom_radio = QRadioButton("Custom Name")
        self.custom_input = QLineEdit()
        self.custom_input.setEnabled(False)
        
        self.custom_radio.toggled.connect(self.custom_input.setEnabled)
        
        layout.addWidget(self.custom_radio)
        layout.addWidget(self.custom_input)
        
        # OK/Cancel buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def get_selected_name(self):
        if self.custom_radio.isChecked():
            return self.custom_input.text().strip()
        
        for radio in self.radio_buttons:
            if radio.isChecked():
                return radio.text()
        return None

class WaveformAnalyzer(QMainWindow):
    def __init__(self, electrode_data=None, filtered_data=None, peaks_info=None, windows=None, sampling_rate=10000):
        super().__init__()
        # Initialize data
        self.raw_data = electrode_data.copy() if electrode_data is not None else {}
        self.filtered_data = filtered_data.copy() if filtered_data is not None else {}
        self.electrode_data = self.raw_data
        self.peaks_info = peaks_info or {}
        self.windows = windows or {}
        self.sampling_rate = sampling_rate
        
        # Analysis parameters
        self.pre_event_ms = 50
        self.post_event_ms = 100
        self.positive_prominence = 10
        self.negative_prominence = 10
        self.show_amplitude_labels = False
        self.using_filtered_data = False
        self.analysis_type = None
        self.analysis_performed = False

        # Initialize peak naming attributes
        self.peak_naming_mode = False
        self.peak_names = {}
        
        
        # Plotting attributes
        self.global_plot_window = None
        self.global_plot_canvas = None
        
        # Results storage
        self.results = {}
        self.average_peaks = {}
        self.intervals = None
        
        # Electrode configuration
        self.electrode_config = ElectrodeConfig(electrode_data or {})
        
        # Initialize UI
        self.setup_ui()
        self._determine_analysis_type()
        self._set_window_parameters()

    
    def display_plot_window(self):
        """Display averaged waveforms in the global plot window"""
        if not self.results:
            QMessageBox.warning(self, "No Results", "Please perform analysis first.")
            return
            
        # Create plot window if it doesn't exist
        if self.global_plot_window is None:
            self.global_plot_window = QWidget()
            self.global_plot_window.setWindowTitle("Global Waveform Analysis")
            layout = QVBoxLayout()
            
            # Add controls at the top
            controls_widget = QWidget()
            controls_layout = QHBoxLayout(controls_widget)
            
            # Add peak naming button
            self.name_peaks_button = QPushButton("Assign Peak Names")
            self.name_peaks_button.clicked.connect(self.toggle_peak_naming_mode)
            controls_layout.addWidget(self.name_peaks_button)
            
            # Add show/hide amplitude labels checkbox
            amplitude_checkbox = QCheckBox("Show Amplitude Labels")
            amplitude_checkbox.setChecked(self.show_amplitude_labels)
            amplitude_checkbox.stateChanged.connect(self.toggle_amplitude_labels)
            controls_layout.addWidget(amplitude_checkbox)
            
            layout.addWidget(controls_widget)
            
            # Create PlotCanvas
            self.global_plot_canvas = PlotCanvas(self.global_plot_window)
            layout.addWidget(self.global_plot_canvas)
            
            self.global_plot_window.setLayout(layout)
            self.global_plot_window.resize(1200, 800)

        # Prepare data for plotting
        plot_data = {}
        peaks_for_plot = {}
        confidence_intervals = {}
        time_axis = None

        for electrode, result in self.results.items():
            # Handle multiple intervals case
            if result.get('multiple_intervals'):
                for i, interval_result in enumerate(result['interval_results']):
                    if interval_result and 'average' in interval_result:
                        if time_axis is None:
                            time_axis = interval_result['time_axis']
                        
                        # Get the average waveform and confidence intervals from the interval
                        mean_waveform, lower_ci, upper_ci = interval_result['average']
                        if mean_waveform is not None:
                            # Create unique key for each interval
                            interval_key = f"{electrode}_interval_{i+1}"
                            plot_data[interval_key] = mean_waveform
                            confidence_intervals[interval_key] = (lower_ci, upper_ci)
                            
                            # Handle peaks for this interval
                            if electrode in self.average_peaks:
                                interval_peaks = self.average_peaks[electrode]
                                if isinstance(interval_peaks, list) and i < len(interval_peaks):
                                    peaks_for_plot[interval_key] = interval_peaks[i]['peaks']
            
            # Handle single interval case
            else:
                if 'average' in result and result['average'][0] is not None:
                    if time_axis is None:
                        time_axis = result['time_axis']
                    mean_waveform, lower_ci, upper_ci = result['average']
                    plot_data[electrode] = mean_waveform
                    confidence_intervals[electrode] = (lower_ci, upper_ci)
                    
                    # Process peaks if they exist
                    if electrode in self.average_peaks:
                        peaks_for_plot[electrode] = self.average_peaks[electrode]

        if not plot_data:
            QMessageBox.warning(self, "No Plottable Data", "No valid data found to plot.")
            return

        # Create DataFrame from plot data
        data_df = pd.DataFrame(plot_data)

        # Plot using PlotCanvas with confidence intervals
        self.global_plot_canvas.plot(
            time_axis, 
            data_df,
            self.electrode_config,
            show_peaks=bool(peaks_for_plot),
            peaks_info=peaks_for_plot if peaks_for_plot else None,
            show_amplitudes=self.show_amplitude_labels,
            confidence_intervals=confidence_intervals
        )

        # Show the plot window
        self.global_plot_window.show()
        
    
    def update_plots(self, show_amplitudes=None):
        """Update plots with peak markers and optional amplitude labels based on analysis type."""
        if show_amplitudes is None:
            show_amplitudes = self.show_amplitude_labels

        # For single-channel analysis
        if self.analysis_type != 'global':
            # If we don't have a single channel plot window, create it
            if not hasattr(self, 'single_channel_window'):
                self.single_channel_window = QWidget()
                self.single_channel_window.setWindowTitle("Single Channel Analysis")
                layout = QVBoxLayout()
                
                # Add controls at the top
                controls_widget = QWidget()
                controls_layout = QHBoxLayout(controls_widget)
                
                # Add peak naming button
                self.name_peaks_button = QPushButton("Assign Peak Names")
                self.name_peaks_button.clicked.connect(self.toggle_peak_naming_mode)
                controls_layout.addWidget(self.name_peaks_button)
                
                # Add show/hide amplitude labels checkbox
                amplitude_checkbox = QCheckBox("Show Amplitude Labels")
                amplitude_checkbox.setChecked(self.show_amplitude_labels)
                amplitude_checkbox.stateChanged.connect(self.toggle_amplitude_labels)
                controls_layout.addWidget(amplitude_checkbox)
                
                layout.addWidget(controls_widget)
                
                # Add the matplotlib canvas
                layout.addWidget(self.canvas)
                
                self.single_channel_window.setLayout(layout)
                self.single_channel_window.resize(1000, 600)
                self.single_channel_window.show()

            # Clear the figure
            self.fig.clear()
            ax = self.fig.add_subplot(111)

            # Get the first electrode's data
            electrode = next(iter(self.results.keys()))
            result = self.results[electrode]

            if result and result['average'][0] is not None:
                time_axis = result['time_axis']
                mean_waveform, lower_ci, upper_ci = result['average']
                data_type = "Filtered" if self.using_filtered_data else "Raw"

                # Plot average waveform and confidence interval
                ax.plot(time_axis, mean_waveform, 'b-', label=f'{data_type} Average')
                ax.fill_between(time_axis, lower_ci, upper_ci,
                                color='b', alpha=0.2, label='95% CI')

                # Plot peaks and their labels
                if electrode in self.average_peaks:
                    pos_count = neg_count = 1
                    for peak in self.average_peaks[electrode]:
                        color = 'red' if peak['type'] == 'positive' else 'green'
                        peak_label = f"{peak['type'].capitalize()} Peak"

                        if peak_label not in [l.get_label() for l in ax.get_lines()]:
                            ax.plot(peak['time'], peak['amplitude'], 'o', color=color,
                                    markersize=6, label=peak_label)
                        else:
                            ax.plot(peak['time'], peak['amplitude'], 'o', color=color,
                                    markersize=6)

                        # Add peak name labels
                        label_text = ''
                        if 'name' in peak:
                            label_text = peak['name']
                        elif show_amplitudes:
                            count = pos_count if peak['type'] == 'positive' else neg_count
                            label_text = f"{'P' if peak['type'] == 'positive' else 'N'}{count}"
                            if peak['type'] == 'positive':
                                pos_count += 1
                            else:
                                neg_count += 1

                        if label_text:
                            # Add value to label if show_amplitudes is True
                            if show_amplitudes:
                                label_text += f": {peak['amplitude']:.1f}µV"
                                
                            ax.annotate(
                                label_text,
                                (peak['time'], peak['amplitude']),
                                textcoords="offset points",
                                xytext=(0, 10 if peak['type'] == 'positive' else -15),
                                ha='center',
                                va='bottom' if peak['type'] == 'positive' else 'top',
                                fontsize=8,
                                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
                            )

                ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
                title = (f'{electrode} ({data_type})\n'
                        f'n={result["stats"]["n_waveforms"]}')
                ax.set_title(title)
                ax.set_xlabel('Time relative to peak (ms)')
                ax.set_ylabel('Amplitude (µV)')
                ax.grid(True)
                ax.legend()

            self.canvas.draw()

        else:
            # For global analysis using PlotCanvas
            self.display_plot_window()

    def toggle_peak_naming_mode(self):
        """Toggle peak naming mode and update button state"""
        self.peak_naming_mode = not self.peak_naming_mode
        
        if self.peak_naming_mode:
            self.name_peaks_button.setText("Exit Naming Mode")
            self.name_peaks_button.setStyleSheet("background-color: #ffcccc;")  # Light red to indicate active
            QMessageBox.information(self, "Peak Naming Mode", 
                                "Click on peaks to assign names. Click the button again to exit naming mode.")
            
            # Connect click event
            if self.analysis_type == 'global':
                self.global_plot_canvas.mpl_connect('button_press_event', self.on_plot_click)
            else:
                self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        else:
            self.name_peaks_button.setText("Assign Peak Names")
            self.name_peaks_button.setStyleSheet("")  # Reset style
            
            # Disconnect click event
            if self.analysis_type == 'global':
                self.global_plot_canvas.mpl_disconnect('button_press_event')
            else:
                self.canvas.mpl_disconnect('button_press_event')
    
        
    def set_data(self, electrode_data=None, filtered_data=None, peaks_info=None, windows=None):
        """Update data and parameters"""
        if electrode_data is not None:
            self.raw_data = electrode_data.copy()
            self.electrode_data = electrode_data.copy()
            self.electrode_config = ElectrodeConfig(electrode_data)
        if filtered_data is not None:
            self.filtered_data = filtered_data.copy()
        if peaks_info is not None:
            self.peaks_info = peaks_info
            self._determine_analysis_type()
        if windows is not None:
            self.windows = windows
        self.analysis_performed = False

    def set_data_type(self, use_filtered=False):
        """Switch between raw and filtered data"""
        if use_filtered:
            if not self.filtered_data:
                QMessageBox.warning(
                    self,
                    "Data Unavailable",
                    "Filtered data is not available. Using raw data instead."
                )
                use_filtered = False
            elif not any(isinstance(value, (list, np.ndarray)) and len(value) > 0 
                        for value in self.filtered_data.values()):
                QMessageBox.warning(
                    self,
                    "Data Unavailable",
                    "Filtered data is empty. Using raw data instead."
                )
                use_filtered = False
                            
        self.using_filtered_data = use_filtered
        self.electrode_data = self.filtered_data if use_filtered else self.raw_data
        print(f"Set data type to {'filtered' if use_filtered else 'raw'}")
        print(f"Electrode data contains {len(self.electrode_data) if self.electrode_data else 0} electrodes")
        self.analysis_performed = False
    
   
    def _determine_analysis_type(self):
        """Determine which type of analysis was performed based on data structure"""
        if not self.peaks_info:
            self.analysis_type = None
            return
            
        try:
            first_electrode = next(iter(self.peaks_info))
            peaks_data = self.peaks_info[first_electrode]
            print("Peaks data type:", type(peaks_data))
            print("Peaks data:", peaks_data)
            
            # Global analysis has (vertical_peaks, other_peaks) structure
            # Check if it's a tuple and first element is a list of peak tuples
            if (isinstance(peaks_data, tuple) and 
                len(peaks_data) == 2 and 
                isinstance(peaks_data[0], list) and
                all(isinstance(p, tuple) and len(p) == 2 for p in peaks_data[0])):
                self.analysis_type = 'global'
                print("Analysis type set to: global")
            else:
                self.analysis_type = 'single'
                print("Analysis type set to: single")
                
        except Exception as e:
            print(f"Error in _determine_analysis_type: {str(e)}")
            self.analysis_type = None
    
    

    
    def setup_ui(self):
        self.setWindowTitle("Waveform Analysis")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Use QVBoxLayout as the main layout for a single compact column
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)  # Reduce margins around the edges
        
        # Parameters adjustment
        # Pre-event window slider
        self.pre_event_slider = self.create_slider(
            "Pre-event window (ms):", 10, 200, self.pre_event_ms)
        self.pre_event_slider.findChild(QSlider).valueChanged.connect(
            lambda v: setattr(self, 'pre_event_ms', v))
        
        # Post-event window slider
        self.post_event_slider = self.create_slider(
            "Post-event window (ms):", 10, 200, self.post_event_ms)
        self.post_event_slider.findChild(QSlider).valueChanged.connect(
            lambda v: setattr(self, 'post_event_ms', v))
        
        # Add sliders for prominence thresholds
        self.pos_prominence_slider = self.create_slider(
            "Positive Peak Prominence:", 0, 100, 10)
        self.pos_prominence_slider.findChild(QSlider).valueChanged.connect(
            lambda v: setattr(self, 'positive_prominence', v))
        
        self.neg_prominence_slider = self.create_slider(
            "Negative Peak Prominence:", 0, 100, 10)
        self.neg_prominence_slider.findChild(QSlider).valueChanged.connect(
            lambda v: setattr(self, 'negative_prominence', v))
        
        # Add amplitude labels checkbox
        self.amplitude_labels_checkbox = QCheckBox("Show Amplitude Labels")
        self.amplitude_labels_checkbox.setChecked(self.show_amplitude_labels)
        self.amplitude_labels_checkbox.stateChanged.connect(self.toggle_amplitude_labels)
        
        # Interval selection button
        self.interval_button = QPushButton("Select Time Intervals")
        self.interval_button.clicked.connect(self.show_interval_selection)
        
        # Analysis buttons
        self.analyze_button = QPushButton("Compute Average Waveforms")
        self.analyze_button.clicked.connect(self.analyze_all_electrodes)
        
        self.find_avg_peaks_button = QPushButton("Find Peaks in Averages")
        self.find_avg_peaks_button.clicked.connect(self.find_peaks_in_averages)
        
        self.compute_amplitudes_button = QPushButton("Compute Amplitudes")
        self.compute_amplitudes_button.clicked.connect(self.compute_and_export_amplitudes)
        
        self.reset_button = QPushButton("Reset Analysis")
        self.reset_button.clicked.connect(self.reset_analysis)
        
        # Add section labels and widgets
        parameters_label = QLabel("Analysis Parameters")
        parameters_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(parameters_label)
        main_layout.addWidget(self.pre_event_slider)
        main_layout.addWidget(self.post_event_slider)
        main_layout.addWidget(self.pos_prominence_slider)
        main_layout.addWidget(self.neg_prominence_slider)
        main_layout.addWidget(self.amplitude_labels_checkbox)
        main_layout.addWidget(self.interval_button)
        
        # Add a separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)
        
        # Add control buttons
        controls_label = QLabel("Analysis Controls")
        controls_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(controls_label)
        main_layout.addWidget(self.analyze_button)
        main_layout.addWidget(self.find_avg_peaks_button)
        main_layout.addWidget(self.compute_amplitudes_button)
        main_layout.addWidget(self.reset_button)
        
        # Create matplotlib figure but don't add it to the layout
        # It will be used in the separate window
        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        
        # Set a fixed width for the window
        self.setFixedWidth(400)
        # Let the height adjust to content
        self.setMinimumHeight(100)

    def create_slider(self, label, min_value, max_value, default_value):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.addWidget(QLabel(label))
        
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(default_value)
        slider.setTickInterval((max_value - min_value) // 10)
        slider.setTickPosition(QSlider.TicksBelow)
        
        value_label = QLabel(str(default_value))
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        
        layout.addWidget(slider)
        layout.addWidget(value_label)
        return widget

    def show_interval_selection(self):
        if not self.electrode_data:
            QMessageBox.warning(
                self, 
                "No Data Available",
                "Please open Signal Analyzer and analyze signals first."
            )
            return

        max_time = max(len(signal) * 1000 // self.sampling_rate 
                      for signal in self.electrode_data.values())
        
        dialog = IntervalSelectionDialog(self, time_range=[0, max_time])
        if dialog.exec_() == QDialog.Accepted:
            self.intervals = dialog.get_intervals()

    def toggle_amplitude_labels(self, state):
        self.show_amplitude_labels = bool(state)
        self.update_plots(show_amplitudes=self.show_amplitude_labels)


    def extract_waveforms(self, signal, peak_times):
        """Extract waveform segments around each peak time"""
        pre_samples = int(self.pre_event_ms * self.sampling_rate / 1000)
        post_samples = int(self.post_event_ms * self.sampling_rate / 1000)
        
        # Convert peak times from ms to sample indices
        peak_indices = [int(t * self.sampling_rate / 1000) for t in peak_times]
        
        waveforms = []
        for peak_idx in peak_indices:
            if peak_idx >= pre_samples and peak_idx + post_samples <= len(signal):
                waveform = signal[peak_idx - pre_samples:peak_idx + post_samples]
                waveforms.append(waveform)
        
        return np.array(waveforms)
    
    def compute_average_waveform(self, waveforms):
        """Compute average waveform and confidence intervals"""
        if len(waveforms) == 0:
            return None, None, None
            
        mean_waveform = np.mean(waveforms, axis=0)
        sem = stats.sem(waveforms, axis=0)
        ci = sem * stats.t.ppf((1 + 0.95) / 2, len(waveforms) - 1)
        
        return mean_waveform, mean_waveform - ci, mean_waveform + ci
    
    
    
    def find_peaks_in_averages(self):
        """Find peaks in the averaged waveforms"""
        if not self.results:
            QMessageBox.warning(self, "Warning", "Please compute average waveforms first.")
            return
            
        self.average_peaks.clear()
        
        for electrode, result in self.results.items():
            if result.get('multiple_intervals'):
                # Handle multiple intervals
                interval_peaks = []
                for interval_result in result['interval_results']:
                    if interval_result['average'][0] is not None:
                        mean_waveform = interval_result['average'][0]
                        time_axis = interval_result['time_axis']
                        peaks = self.detect_peaks_in_average(mean_waveform, time_axis)
                        interval_peaks.append({
                            'interval': interval_result['interval'],
                            'peaks': peaks
                        })
                if interval_peaks:
                    self.average_peaks[electrode] = interval_peaks
            else:
                # Handle single interval
                if result['average'][0] is not None:
                    mean_waveform = result['average'][0]
                    time_axis = result['time_axis']
                    peaks = self.detect_peaks_in_average(mean_waveform, time_axis)
                    # Integrate detailed single-channel analysis
                    single_channel_peaks = []
                    pos_count = neg_count = 1
                    for peak in peaks:
                        peak_type = 'positive' if peak['amplitude'] > 0 else 'negative'
                        label = f"P{pos_count}" if peak_type == 'positive' else f"N{neg_count}"
                        single_channel_peaks.append({
                            'time': peak['time'],
                            'amplitude': peak['amplitude'],
                            'type': peak_type,
                            'label': label
                        })
                        if peak_type == 'positive':
                            pos_count += 1
                        else:
                            neg_count += 1
                    self.average_peaks[electrode] = single_channel_peaks
        
        self.update_plots()  # Update plots to show the detected peaks


    
    def _set_window_parameters(self):
        """Set window parameters based on available data"""
        self.pre_event_ms = 50  # Default values
        self.post_event_ms = 100
        
        if not self.windows or not self.peaks_info:
            return
            
        try:
            first_electrode = next(iter(self.windows.keys()))
            window = self.windows[first_electrode]
            peaks = self.peaks_info[first_electrode]
            
            if self.analysis_type == 'global':
                vertical_peaks, _ = peaks
                if vertical_peaks:
                    peak_times = [float(t) for t, _ in vertical_peaks]
                    if peak_times and window:
                        reference_peak = peak_times[0]
                        window_start = float(window[0][0])
                        window_end = float(window[0][1])
                        self.pre_event_ms = int(reference_peak - window_start)
                        self.post_event_ms = int(window_end - reference_peak)
            else:  # single channel analysis
                if peaks and window:
                    peak_times = [float(t) for t in peaks]
                    if peak_times:
                        reference_peak = peak_times[0]
                        window_start = float(window[0])
                        window_end = float(window[1])
                        self.pre_event_ms = int(reference_peak - window_start)
                        self.post_event_ms = int(window_end - reference_peak)
        except Exception as e:
            print(f"Error setting window parameters: {str(e)}")
            # Keep default values if there's an error

    
    
    def analyze_electrode(self, signal, peaks_info=None):
        """
        Analyze waveforms for a single electrode with support for both analysis types.
        """
        if peaks_info is None:
            return None

        results = {}

        # Extract peak times based on analysis type
        if self.analysis_type == 'global':
            vertical_peaks, _ = peaks_info
            all_peak_times = [t for t, _ in vertical_peaks]
            all_peak_amplitudes = [amp for _, amp in vertical_peaks]
        else:  # single channel analysis
            all_peak_times = [t for t in peaks_info]
            all_peak_amplitudes = [signal[int(t * self.sampling_rate / 1000)] for t in peaks_info]

            # Add detailed single-channel analysis logic
            pos_count = neg_count = 1
            peaks_detailed = []
            for t, amp in zip(all_peak_times, all_peak_amplitudes):
                peak_type = 'positive' if amp > 0 else 'negative'
                label = f"P{pos_count}" if peak_type == 'positive' else f"N{neg_count}"
                peaks_detailed.append({
                    'time': t,
                    'amplitude': amp,
                    'type': peak_type,
                    'label': label
                })

                if peak_type == 'positive':
                    pos_count += 1
                else:
                    neg_count += 1

            results['detailed_peaks'] = peaks_detailed

        if self.intervals and len(self.intervals) > 1:
            # Handle multiple intervals
            results['interval_results'] = []
            for start_ms, end_ms in self.intervals:
                interval_peaks = []
                interval_amplitudes = []
                for t, amp in zip(all_peak_times, all_peak_amplitudes):
                    if start_ms <= t <= end_ms:
                        interval_peaks.append(t)
                        interval_amplitudes.append(amp)

                if interval_peaks:
                    waveforms = self.extract_waveforms(signal, interval_peaks)
                    average_results = self.compute_average_waveform(waveforms)

                    interval_result = {
                        'interval': (start_ms, end_ms),
                        'waveforms': waveforms,
                        'average': average_results,
                        'time_axis': np.linspace(-self.pre_event_ms, self.post_event_ms,
                                                waveforms.shape[1] if len(waveforms) > 0 else 0),
                        'stats': {
                            'n_waveforms': len(waveforms),
                            'peak_amplitude_mean': np.mean(interval_amplitudes),
                            'peak_amplitude_std': np.std(interval_amplitudes),
                            'inter_peak_intervals': np.diff(interval_peaks)
                        }
                    }
                    results['interval_results'].append(interval_result)

            results['multiple_intervals'] = True
        else:
            # Single interval analysis
            waveforms = self.extract_waveforms(signal, all_peak_times)
            average_results = self.compute_average_waveform(waveforms)

            results['multiple_intervals'] = False
            results['waveforms'] = waveforms
            results['average'] = average_results
            results['time_axis'] = np.linspace(-self.pre_event_ms, self.post_event_ms,
                                            waveforms.shape[1] if len(waveforms) > 0 else 0)

            if len(waveforms) > 0:
                results['stats'] = {
                    'n_waveforms': len(waveforms),
                    'peak_amplitude_mean': np.mean(all_peak_amplitudes),
                    'peak_amplitude_std': np.std(all_peak_amplitudes),
                    'inter_peak_intervals': np.diff(all_peak_times)
                }

        return results

    
    def analyze_all_electrodes(self):
        """
        Analyze waveforms for all electrodes in the dataset.
        Supports both single channel and global analysis types.
        """
        if not self.electrode_data:
            QMessageBox.warning(self, "No Data", "Please load electrode data first.")
            return

        self.results.clear()
        
        # Iterate through all electrodes
        for electrode, signal in self.electrode_data.items():
            # Get peaks for this specific electrode
            peaks_info = (self.peaks_info.get(electrode) 
                        if electrode in self.peaks_info 
                        else None)
            
            # Skip if no peaks found
            if peaks_info is None:
                continue
            
            # Analyze the electrode
            result = self.analyze_electrode(signal, peaks_info)
            
            if result:
                self.results[electrode] = result
        
        # Check if any results were found
        if not self.results:
            QMessageBox.warning(self, "No Results", 
                                "No waveforms could be extracted. Check peak detection.")
            return
        
        # Mark analysis as performed
        self.analysis_performed = True
        
        # Update plots
        self.update_plots()
        
    def setup_peak_naming(self):
        # Add button to toggle peak naming mode
        self.name_peaks_button = QPushButton("Assign Peak Names")
        self.name_peaks_button.clicked.connect(self.toggle_peak_naming_mode)
        
        # Add to existing UI
        if hasattr(self, 'global_plot_window'):
            layout = self.global_plot_window.layout()
            layout.insertWidget(1, self.name_peaks_button)  # Insert before the plot
        
        self.peak_naming_mode = False
        self.peak_names = {}  # Store peak names for each electrode

    

    def compute_and_export_amplitudes(self):
        if not self.average_peaks:
            QMessageBox.warning(self, "Warning", "Please detect peaks first.")
            return
            
        data_for_export = []
        
        for electrode, result in self.results.items():
            if result is None:
                continue
                
            electrode_peaks = self.average_peaks.get(electrode, [])
            electrode_names = self.peak_names.get(electrode, {})
            
            if result.get('multiple_intervals'):
                # Multiple intervals case
                for i, interval_result in enumerate(result['interval_results']):
                    if electrode_peaks and isinstance(electrode_peaks, list) and len(electrode_peaks) > i:
                        interval_peak_data = electrode_peaks[i]
                        start_ms, end_ms = interval_result['interval']
                        
                        for peak in interval_peak_data['peaks']:
                            peak_name = electrode_names.get(peak['time'], 
                                                        f"{'P' if peak['type'] == 'positive' else 'N'}")
                            
                            data_for_export.append({
                                'Electrode': electrode,
                                'Interval': f"{start_ms}-{end_ms}ms",
                                'Peak_Name': peak_name,
                                'Peak_Type': "Positive" if peak['type'] == 'positive' else "Negative",
                                'Time_ms': round(peak['time'], 2),
                                'Amplitude_uV': round(peak['amplitude'], 2)
                            })
            else:
                # Single interval case
                for peak in electrode_peaks:
                    peak_name = electrode_names.get(peak['time'], 
                                                f"{'P' if peak['type'] == 'positive' else 'N'}")
                    
                    data_for_export.append({
                        'Electrode': electrode,
                        'Interval': 'Full Range',
                        'Peak_Name': peak_name,
                        'Peak_Type': "Positive" if peak['type'] == 'positive' else "Negative",
                        'Time_ms': round(peak['time'], 2),
                        'Amplitude_uV': round(peak['amplitude'], 2)
                    })
        
        # Create DataFrame and save to Excel
        if data_for_export:
            df = pd.DataFrame(data_for_export)
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Save Peak Amplitudes", "", "Excel Files (*.xlsx)")
            if file_name:
                if not file_name.endswith('.xlsx'):
                    file_name += '.xlsx'
                df.to_excel(file_name, index=False)
                QMessageBox.information(self, "Success", "Peak amplitudes saved successfully!")
        
        # Update plots with new labels
        self.update_plots()
        
    

    def on_plot_click(self, event):
            if not self.peak_naming_mode or event.inaxes is None:
                return
                
            # Find the closest peak to the click
            clicked_x, clicked_y = event.xdata, event.ydata
            closest_peak = None
            min_distance = float('inf')
            current_electrode = None
            
            for electrode, peaks in self.average_peaks.items():
                if isinstance(peaks, list):
                    for peak in peaks:
                        distance = ((peak['time'] - clicked_x) ** 2 + 
                                (peak['amplitude'] - clicked_y) ** 2) ** 0.5
                        if distance < min_distance and distance < 5:  # Threshold for clicking
                            min_distance = distance
                            closest_peak = peak
                            current_electrode = electrode
            
            if closest_peak is not None:
                # Show name selection dialog
                dialog = PeakNameDialog(self)
                if dialog.exec_() == QDialog.Accepted:
                    name = dialog.get_selected_name()
                    if name:
                        # Store the name in both dictionaries
                        if current_electrode not in self.peak_names:
                            self.peak_names[current_electrode] = {}
                        self.peak_names[current_electrode][closest_peak['time']] = name
                        
                        # Update the peak information directly
                        closest_peak['name'] = name
                        
                        # Update plots immediately to show the new label
                        self.update_plots()

    def detect_peaks_in_average(self, waveform, time_axis):
            """
            Detect peaks in an averaged waveform using user-defined prominences.
            Classifies peaks based on their actual amplitude relative to baseline.
            """
            pos_prominence = self.positive_prominence if hasattr(self, 'positive_prominence') else 1.0
            neg_prominence = self.negative_prominence if hasattr(self, 'negative_prominence') else 1.0

            # Find peaks in both directions
            pos_peaks, pos_properties = find_peaks(waveform, prominence=pos_prominence)
            neg_peaks, neg_properties = find_peaks(-waveform, prominence=neg_prominence)

            peaks = []
            
            # Process all peaks found in positive direction
            for peak_idx in pos_peaks:
                peak = {
                    'time': time_axis[peak_idx],
                    'amplitude': waveform[peak_idx],
                    'type': 'positive' if waveform[peak_idx] > 0 else 'negative'
                }
                # Add name if it exists in peak_names
                if hasattr(self, 'peak_names'):
                    for electrode, names in self.peak_names.items():
                        if peak['time'] in names:
                            peak['name'] = names[peak['time']]
                peaks.append(peak)
            
            # Process all peaks found in negative direction
            for peak_idx in neg_peaks:
                # Only add if this time point wasn't already added from pos_peaks
                if peak_idx not in pos_peaks:
                    peak = {
                        'time': time_axis[peak_idx],
                        'amplitude': waveform[peak_idx],
                        'type': 'negative'
                    }
                    # Add name if it exists in peak_names
                    if hasattr(self, 'peak_names'):
                        for electrode, names in self.peak_names.items():
                            if peak['time'] in names:
                                peak['name'] = names[peak['time']]
                    peaks.append(peak)

            # Sort peaks by time
            peaks.sort(key=lambda x: x['time'])
            return peaks

    def reset_analysis(self):
        """Reset all analysis results and peaks but preserve the plotting windows"""
        # Clear analysis data
        self.results.clear()
        self.intervals = None
        self.average_peaks.clear()
        self.analysis_performed = False
        
        if self.analysis_type != 'global':
            # Clear single-channel plot
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.set_xlabel('Time relative to peak (ms)')
            ax.set_ylabel('Amplitude (µV)')
            ax.grid(True)
            self.canvas.draw()
            self.canvas.flush_events()
        else:
            # Clear global plot if it exists
            if self.global_plot_window is not None:
                self.global_plot_canvas.clear_plot()
        
        # Show message to confirm reset
        QMessageBox.information(self, "Reset Complete", 
                            "Analysis has been reset. You can now perform a new analysis.")
        
        