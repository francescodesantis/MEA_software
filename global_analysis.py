from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QDialog, QListWidget, QDialogButtonBox, QMessageBox,
    QMainWindow, QDockWidget, QSlider, QCheckBox, QApplication,QRadioButton
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import scipy.signal 
from electrode_config import ElectrodeConfig
from matplotlib.gridspec import GridSpec
from plot8 import PlotCanvas
import pandas as pd


def apply_butterworth_filter(signal, sampling_rate, cutoff_low, cutoff_high, vertical_threshold):

    # Apply adaptive Butterworth filter
    nyquist_freq = sampling_rate / 2
    max_amplitude = np.max(np.abs(signal))
    if max_amplitude <= vertical_threshold:
            normalized_cutoff = cutoff_low / nyquist_freq
    else:
            normalized_cutoff = cutoff_high / nyquist_freq
    b, a = scipy.signal.butter(6, normalized_cutoff, btype='low', analog=False)
    filtered_signal = scipy.signal.filtfilt(b, a, signal)

    return filtered_signal

def apply_savgol_filter(signal, window_length, polyorder):
    return savgol_filter(signal, window_length, polyorder)

def apply_moving_average_filter(signal, window_size):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

def detect_peaks(signal, sampling_rate, pos_prominence=10, neg_prominence=10):
    peaks_pos, _ = find_peaks(signal, prominence=pos_prominence)
    peaks_neg, _ = find_peaks(-signal, prominence=neg_prominence)
    return peaks_pos, peaks_neg

"""def calculate_activation_windows(vertical_peaks, isi_percentage=80):
    if len(vertical_peaks) > 1:
        valid_times = np.array([v[0] for v in vertical_peaks])
        isi = np.diff(valid_times)
        isi_min = np.min(isi)
        isi_80_percent = (isi_percentage / 100) * isi_min

        shift_10_percent = 0.1 * isi_80_percent
        windows = [(peak - shift_10_percent, peak - shift_10_percent + isi_80_percent) for peak in valid_times]
        return windows, isi_min, isi_80_percent
    else:
        return [], np.nan, np.nan"""

def calculate_activation_windows(vertical_peaks, isi_percentage=80):
    """Calculate activation windows based only on vertical peaks"""
    if len(vertical_peaks) > 1:
        valid_times = np.array([v[0] for v in vertical_peaks])
        isi = np.diff(valid_times)
        isi_min = np.min(isi)
        isi_80_percent = (isi_percentage / 100) * isi_min

        shift_10_percent = 0.1 * isi_80_percent
        windows = [(peak - shift_10_percent, peak - shift_10_percent + isi_80_percent) for peak in valid_times]
        return windows, isi_min, isi_80_percent
    else:
        return [], np.nan, np.nan


class ControlPanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Control Panel")
        self.setup_ui()
    
    def setup_ui(self):
        # Main layout
        layout = QVBoxLayout(self)
        
        # Add vertical threshold input
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Vertical Threshold:"))
        self.vertical_threshold_input = QSpinBox()
        self.vertical_threshold_input.setRange(-1000, 1000)
        self.vertical_threshold_input.setValue(0)
        threshold_layout.addWidget(self.vertical_threshold_input)
        layout.addLayout(threshold_layout)
        
        # Create mode selection group
        mode_group = QWidget()
        mode_layout = QVBoxLayout(mode_group)
        mode_label = QLabel("Analysis Mode:")
        mode_label.setStyleSheet("font-weight: bold;")
        mode_layout.addWidget(mode_label)
        
        # Create radio buttons for modes
        self.no_filter_radio = QRadioButton("No Filter Mode (Raw Data Analysis)")
        self.auto_radio = QRadioButton("Automatic Filtering")
        self.custom_radio = QRadioButton("Custom Filtering")
        
        # Add tooltips
        self.no_filter_radio.setToolTip("Analyze raw data directly without any filtering")
        self.auto_radio.setToolTip("Apply all filters automatically with optimized parameters")
        self.custom_radio.setToolTip("Customize and apply filters individually")
        
        # Add radio buttons to mode group
        mode_layout.addWidget(self.no_filter_radio)
        mode_layout.addWidget(self.auto_radio)
        mode_layout.addWidget(self.custom_radio)
        
        # Set No Filter as default
        self.no_filter_radio.setChecked(True)
        layout.addWidget(mode_group)
        
        # No Filter Mode Message
        self.no_filter_message = QLabel(
            "No filtering applied. Analysis will be performed on raw data."
        )
        self.no_filter_message.setStyleSheet("""
            QLabel {
                color: #666;
                font-style: italic;
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 4px;
            }
        """)
        self.no_filter_message.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.no_filter_message)
        
        # Automatic Filtering Section
        self.auto_controls = QWidget()
        auto_layout = QVBoxLayout(self.auto_controls)
        
        self.auto_filter_button = QPushButton("Apply Automatic Filtering")
        self.auto_filter_button.setStyleSheet("""
            QPushButton { 
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        auto_layout.addWidget(self.auto_filter_button)
        
        
        #auto_info = QLabel("Applies optimal filters in sequence:\nSavitzky-Golay → Butterworth → Moving Average")
        #auto_info.setStyleSheet("color: #666; font-style: italic;")
        #auto_info.setAlignment(Qt.AlignCenter)
        #auto_layout.addWidget(auto_info)
        layout.addWidget(self.auto_controls)
        
        # Custom Filtering Section
        self.custom_controls = QWidget()
        custom_layout = QVBoxLayout(self.custom_controls)

        self.apply_all_custom_button = QPushButton("Apply All Filters")
        self.apply_all_custom_button.setStyleSheet("""
            QPushButton { 
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        custom_layout.addWidget(self.apply_all_custom_button)

        # Add an info label
        #info_label = QLabel("Applies filters in sequence: Savitzky-Golay → Butterworth → Moving Average")
        #info_label.setStyleSheet("color: #666; font-style: italic;")
        #info_label.setAlignment(Qt.AlignCenter)
        #custom_layout.addWidget(info_label)

        # Add some spacing before the individual filters
        custom_layout.addSpacing(10)
                
        # Add filter parameter sliders
        parameters_layout = QVBoxLayout()
        
        # Group 1: Savitzky-Golay filter
        savgol_group = QVBoxLayout()
        self.savgol_button = QPushButton("Apply Savitzky-Golay Filter")
        self.savgol_button.setCheckable(True)
        self.savgol_button.setStyleSheet("""
            QPushButton { background-color: #f0f0f0; }
            QPushButton:checked { background-color: #90EE90; }
        """)
        savgol_group.addWidget(self.savgol_button)
        
        self.savgol_window_slider = self.create_slider("Window:", 3, 101, 21)
        self.savgol_order_slider = self.create_slider("Order:", 1, 5, 3)
        savgol_group.addWidget(self.savgol_window_slider)
        savgol_group.addWidget(self.savgol_order_slider)
        custom_layout.addLayout(savgol_group)
        
        # Group 2: Butterworth filter
        butterworth_group = QVBoxLayout()
        self.butter_button = QPushButton("Apply Butterworth Filter")
        self.butter_button.setCheckable(True)
        self.butter_button.setStyleSheet("""
            QPushButton { background-color: #f0f0f0; }
            QPushButton:checked { background-color: #90EE90; }
        """)
        butterworth_group.addWidget(self.butter_button)
        
        self.butter_low_slider = self.create_slider("Low Cutoff (Hz):", 1, 500, 100)
        self.butter_high_slider = self.create_slider("High Cutoff (Hz):", 500, 5000, 2000)
        butterworth_group.addWidget(self.butter_low_slider)
        butterworth_group.addWidget(self.butter_high_slider)
        custom_layout.addLayout(butterworth_group)
        
        # Group 3: Moving average filter
        moving_avg_group = QVBoxLayout()
        self.moving_avg_button = QPushButton("Apply Moving Average")
        self.moving_avg_button.setCheckable(True)
        self.moving_avg_button.setStyleSheet("""
            QPushButton { background-color: #f0f0f0; }
            QPushButton:checked { background-color: #90EE90; }
        """)
        moving_avg_group.addWidget(self.moving_avg_button)
        
        self.mov_avg_window_slider = self.create_slider("Window:", 3, 101, 15)
        moving_avg_group.addWidget(self.mov_avg_window_slider)
        custom_layout.addLayout(moving_avg_group)
        
        layout.addWidget(self.custom_controls)
        
        # Peak Detection Parameters
        peak_group = QVBoxLayout()
        peak_label = QLabel("Peak Detection Parameters")
        peak_label.setStyleSheet("font-weight: bold;")
        peak_group.addWidget(peak_label)
        
        self.pos_prominence_slider = self.create_slider("Positive Peak Prominence:", 1, 100, 10)
        self.neg_prominence_slider = self.create_slider("Negative Peak Prominence:", 1, 100, 10)
        peak_group.addWidget(self.pos_prominence_slider)
        peak_group.addWidget(self.neg_prominence_slider)
        layout.addLayout(peak_group)
        
        # Analysis buttons
        buttons_layout = QVBoxLayout()
        
        self.peaks_button = QPushButton("Find Peaks")
        buttons_layout.addWidget(self.peaks_button)
        
        self.windows_button = QPushButton("Compute Activation Windows")
        buttons_layout.addWidget(self.windows_button)
        
        self.reset_button = QPushButton("Reset")
        buttons_layout.addWidget(self.reset_button)
        
        layout.addLayout(buttons_layout)
        
        # Connect radio buttons to update mode
        self.no_filter_radio.toggled.connect(self.update_mode)
        self.auto_radio.toggled.connect(self.update_mode)
        self.custom_radio.toggled.connect(self.update_mode)
        
        # Initialize mode visibility
        self.update_mode()
    
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
        widget.slider = slider
        return widget
    
    def update_mode(self):
        # Update visibility based on selected mode
        self.no_filter_message.setVisible(self.no_filter_radio.isChecked())
        self.auto_controls.setVisible(self.auto_radio.isChecked())
        self.custom_controls.setVisible(self.custom_radio.isChecked())

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QDialog, QListWidget, QDialogButtonBox, QMessageBox,
    QMainWindow, QDockWidget, QSlider, QCheckBox, QSpinBox,
    QGridLayout
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import scipy.signal 
import math
import csv
from PyQt5.QtWidgets import QFileDialog

class GlobalAnalyzer(QMainWindow):
    def __init__(self, electrode_data, selected_electrodes, sampling_rate=10000):
        super().__init__()
        self.electrode_data = electrode_data
        self.selected_electrodes = selected_electrodes
        self.sampling_rate = sampling_rate
        self.electrode_config = ElectrodeConfig(self.electrode_data)
        
        # Initialize data
        self.initialize_data()
        
        # Initialize storage for analysis
        self.filtered_signals = {electrode: None for electrode in selected_electrodes}
        self.peaks = {electrode: None for electrode in selected_electrodes}
        self.windows = {electrode: None for electrode in selected_electrodes}
        self.active_filters = {
            'butterworth': False,
            'savgol': False,
            'moving_average': False
        }
        
        self.plot_canvas = None
        self.setWindowTitle("Signal Analysis")
        self.setMinimumSize(1200, 800)
        
        self.setup_ui()
        
        # Create and show control panel in separate window
        self.control_panel_window = QMainWindow()
        self.control_panel_window.setWindowTitle("Control Panel")
        self.control_panel = ControlPanel()
        self.control_panel_window.setCentralWidget(self.control_panel)
        
        # Connect control panel signals
        self.connect_control_panel()
        
        # Show only control panel initially
        self.control_panel_window.show()
        self.hide()

    def connect_control_panel(self):
        """Connect control panel signals to slots with mode handling"""
        # Connect mode radio buttons
        self.control_panel.no_filter_radio.toggled.connect(self.handle_mode_change)
        self.control_panel.auto_radio.toggled.connect(self.handle_mode_change)
        self.control_panel.custom_radio.toggled.connect(self.handle_mode_change)
        
        # Connect auto filter button
        self.control_panel.auto_filter_button.clicked.connect(self.apply_automatic_mode)
        
        # Connect custom filter buttons
        self.control_panel.apply_all_custom_button.clicked.connect(self.apply_all_custom_filters)
        self.control_panel.savgol_button.toggled.connect(
            lambda checked: self.handle_filter_change('savgol', checked))
        self.control_panel.butter_button.toggled.connect(
            lambda checked: self.handle_filter_change('butterworth', checked))
        self.control_panel.moving_avg_button.toggled.connect(
            lambda checked: self.handle_filter_change('moving_average', checked))
        
        # Connect analysis buttons
        self.control_panel.peaks_button.clicked.connect(self.find_peaks)
        self.control_panel.windows_button.clicked.connect(self.compute_windows)
        self.control_panel.reset_button.clicked.connect(self.reset_plots)
        
        # Connect threshold input
        self.control_panel.vertical_threshold_input.valueChanged.connect(
            self.update_vertical_threshold)

    def handle_mode_change(self):
        """Handle changes in filtering mode"""
        # Reset any active filters when changing modes
        self.reset_plots()
        
        if self.control_panel.no_filter_radio.isChecked():
            self.handle_no_filter_mode()
        elif self.control_panel.auto_radio.isChecked():
            self.handle_auto_mode()
        elif self.control_panel.custom_radio.isChecked():
            self.handle_custom_mode()

    def handle_no_filter_mode(self):
        """Handle no filter mode selection"""
        self.filtered_signals = {electrode: None for electrode in self.selected_electrodes}
        if self.plot_canvas:
            time = np.arange(len(self.electrode_data[self.selected_electrodes[0]])) / self.sampling_rate * 1000
            plot_data = pd.DataFrame({
                electrode: self.electrode_data[electrode] 
                for electrode in self.selected_electrodes
            })
            self.plot_canvas.plot(time, plot_data, self.electrode_config)
            self.show()

    def handle_auto_mode(self):
        """Handle automatic mode selection"""
        # Clear any existing custom filters
        self.active_filters = {filter_name: False for filter_name in self.active_filters}
        
        # Enable auto filter controls
        self.control_panel.auto_filter_button.setEnabled(True)

    def handle_custom_mode(self):
        """Handle custom mode selection"""
        # Enable all custom filter controls
        self.control_panel.savgol_button.setEnabled(True)
        self.control_panel.butter_button.setEnabled(True)
        self.control_panel.moving_avg_button.setEnabled(True)

    def apply_automatic_mode(self):
        """Apply automatic filtering to all signals"""
        try:
            for electrode in self.selected_electrodes:
                self.filtered_signals[electrode] = self.apply_automatic_filtering(
                    self.electrode_data[electrode]
                )
            
            if not self.plot_canvas:
                self.initialize_plot_canvas()
            
            time = np.arange(len(self.electrode_data[self.selected_electrodes[0]])) / self.sampling_rate * 1000
            plot_data = pd.DataFrame({
                electrode: self.filtered_signals[electrode] 
                for electrode in self.selected_electrodes
            })
            self.plot_canvas.plot(time, plot_data, self.electrode_config)
            self.show()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply automatic filtering: {str(e)}")
    
    def apply_all_custom_filters(self):
        """Apply all filters in sequence with user-selected parameters"""
        try:
            for electrode in self.selected_electrodes:
                signal = self.electrode_data[electrode].copy()
                
                # 1. Apply Savitzky-Golay
                window_length = self.control_panel.savgol_window_slider.slider.value()
                if window_length % 2 == 0:
                    window_length += 1
                polyorder = self.control_panel.savgol_order_slider.slider.value()
                signal = apply_savgol_filter(signal, window_length, polyorder)
                
                # 2. Apply Butterworth
                cutoff_low = self.control_panel.butter_low_slider.slider.value()
                cutoff_high = self.control_panel.butter_high_slider.slider.value()
                vertical_threshold = self.control_panel.vertical_threshold_input.value()
                signal = apply_butterworth_filter(
                    signal,
                    self.sampling_rate,
                    cutoff_low,
                    cutoff_high,
                    vertical_threshold
                )
                
                # 3. Apply Moving Average
                window_size = self.control_panel.mov_avg_window_slider.slider.value()
                signal = apply_moving_average_filter(signal, window_size)
                
                # Store the filtered signal
                self.filtered_signals[electrode] = signal
            
            # Update plot
            if not self.plot_canvas:
                self.initialize_plot_canvas()
            
            time = np.arange(len(self.electrode_data[self.selected_electrodes[0]])) / self.sampling_rate * 1000
            plot_data = pd.DataFrame({
                electrode: self.filtered_signals[electrode] 
                for electrode in self.selected_electrodes
            })
            self.plot_canvas.plot(time, plot_data, self.electrode_config)
            self.show()
            
            # Auto-check all filter buttons to show they've been applied
            self.control_panel.savgol_button.setChecked(True)
            self.control_panel.butter_button.setChecked(True)
            self.control_panel.moving_avg_button.setChecked(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply all filters: {str(e)}")

    def handle_filter_change(self, filter_name, checked):
        """Handle filter button toggle with mode checking"""
        if self.control_panel.custom_radio.isChecked():
            self.active_filters[filter_name] = checked
            self.update_filtered_signals()
        else:
            # Uncheck the button if we're not in custom mode
            if filter_name == 'butterworth':
                self.control_panel.butter_button.setChecked(False)
            elif filter_name == 'savgol':
                self.control_panel.savgol_button.setChecked(False)
            elif filter_name == 'moving_average':
                self.control_panel.moving_avg_button.setChecked(False)

    def update_filtered_signals(self):
        """Update filtered signals based on current mode and active filters"""
        if self.control_panel.no_filter_radio.isChecked():
            self.filtered_signals = {electrode: None for electrode in self.selected_electrodes}
        elif self.control_panel.auto_radio.isChecked():
            # Don't update here - wait for auto filter button click
            return
        elif self.control_panel.custom_radio.isChecked() and any(self.active_filters.values()):
            for electrode in self.selected_electrodes:
                self.filtered_signals[electrode] = self.apply_all_active_filters(
                    self.electrode_data[electrode]
                )
        
        if not self.plot_canvas:
            self.initialize_plot_canvas()
        
        time = np.arange(len(self.electrode_data[self.selected_electrodes[0]])) / self.sampling_rate * 1000
        plot_data = pd.DataFrame({
            electrode: self.filtered_signals[electrode] if self.filtered_signals[electrode] is not None 
            else self.electrode_data[electrode]
            for electrode in self.selected_electrodes
        })
        self.plot_canvas.plot(time, plot_data, self.electrode_config)
        self.show()

    def reset_plots(self):
        """Reset all analysis and plots with mode handling"""
        self.filtered_signals = {electrode: None for electrode in self.selected_electrodes}
        self.peaks = {electrode: None for electrode in self.selected_electrodes}
        self.windows = {electrode: None for electrode in self.selected_electrodes}
        self.active_filters = {filter_name: False for filter_name in self.active_filters}
        
        # Reset control panel buttons
        self.control_panel.butter_button.setChecked(False)
        self.control_panel.savgol_button.setChecked(False)
        self.control_panel.moving_avg_button.setChecked(False)
        
        # Reset plot canvas
        if self.plot_canvas:
            self.plot_canvas.vertical_amplitudes_info = {}
            self.plot_canvas.windows = {}
            
            time = np.arange(len(self.electrode_data[self.selected_electrodes[0]])) / self.sampling_rate * 1000
            plot_data = pd.DataFrame({
                electrode: self.electrode_data[electrode] 
                for electrode in self.selected_electrodes
            })
            self.plot_canvas.plot(time, plot_data, self.electrode_config)

    
    def initialize_plot_canvas(self):
        # Clear existing plot canvas if it exists
        if self.plot_canvas:
            self.plot_layout.removeWidget(self.plot_canvas)
            self.plot_canvas.deleteLater()
        
        # Create new plot canvas with minimum size
        self.plot_canvas = PlotCanvas(self)
        self.plot_canvas.setMinimumSize(900, 500)  # Minimum size for canvas
        self.plot_layout.addWidget(self.plot_canvas)
        
        # Initialize the plot grid
        time = np.arange(len(self.electrode_data[self.selected_electrodes[0]])) / self.sampling_rate * 1000
        data = pd.DataFrame({
            electrode: self.filtered_signals[electrode] if self.filtered_signals[electrode] is not None 
            else self.electrode_data[electrode] 
            for electrode in self.selected_electrodes
        })
        self.plot_canvas.plot(time, data, self.electrode_config)

    def apply_all_active_filters(self, signal):
        """Apply filters in consistent order: Savitzky-Golay -> Butterworth -> Moving Average"""
        filtered_signal = signal.copy()
        vertical_threshold = self.control_panel.vertical_threshold_input.value()
        
        # 1. First apply Savitzky-Golay if active
        if self.active_filters['savgol']:
            window_length = self.control_panel.savgol_window_slider.slider.value()
            if window_length % 2 == 0:
                window_length += 1
            polyorder = self.control_panel.savgol_order_slider.slider.value()
            filtered_signal = apply_savgol_filter(filtered_signal, window_length, polyorder)
        
        # 2. Then apply Butterworth if active
        if self.active_filters['butterworth']:
            cutoff_low = self.control_panel.butter_low_slider.slider.value()
            cutoff_high = self.control_panel.butter_high_slider.slider.value()
            filtered_signal = apply_butterworth_filter(
                filtered_signal, 
                self.sampling_rate, 
                cutoff_low, 
                cutoff_high, 
                vertical_threshold
            )
        
        # 3. Finally apply Moving Average if active
        if self.active_filters['moving_average']:
            window_size = self.control_panel.mov_avg_window_slider.slider.value()
            filtered_signal = apply_moving_average_filter(filtered_signal, window_size)
        
        return filtered_signal

    def apply_automatic_filtering(self, signal):
        """Apply automatic filtering with optimized parameters"""
        filtered_signal = signal.copy()
        
        # 1. Apply Savitzky-Golay with optimized parameters
        window_length = 21  # Optimized for typical signal characteristics
        if window_length % 2 == 0:
            window_length += 1
        polyorder = 3
        filtered_signal = apply_savgol_filter(filtered_signal, window_length, polyorder)
        
        # 2. Apply Butterworth with adaptive parameters
        cutoff_low = 100  # Optimized low cutoff
        cutoff_high = 2000  # Optimized high cutoff
        vertical_threshold = self.control_panel.vertical_threshold_input.value()
        filtered_signal = apply_butterworth_filter(
            filtered_signal, 
            self.sampling_rate, 
            cutoff_low, 
            cutoff_high, 
            vertical_threshold
        )
        
        # 3. Apply Moving Average with optimized window
        window_size = 15  # Optimized window size
        filtered_signal = apply_moving_average_filter(filtered_signal, window_size)
        
        return filtered_signal

    def find_peaks(self):
        #Find and classify peaks using consistent logic with SignalAnalyzer
        if not self.plot_canvas:
            return

        try:
            vertical_threshold = self.control_panel.vertical_threshold_input.value()
            pos_prominence = self.control_panel.pos_prominence_slider.slider.value()
            neg_prominence = self.control_panel.neg_prominence_slider.slider.value()

            self.plot_canvas.vertical_threshold = vertical_threshold

            for electrode in self.selected_electrodes:
                # Use filtered signal if available, maintaining filter order
                signal = (self.filtered_signals[electrode] 
                         if self.filtered_signals.get(electrode) is not None 
                         else self.electrode_data[electrode])

                # Find both positive and negative peaks
                peaks_pos, _ = find_peaks(signal, prominence=pos_prominence)
                peaks_neg, _ = find_peaks(-signal, prominence=neg_prominence)

                # Combine and sort all peaks
                final_peaks = np.sort(np.concatenate((peaks_pos, peaks_neg)))

                # Classify peaks based on threshold
                vertical_peaks = []
                other_peaks = []

                for peak_idx in final_peaks:
                    peak_time = peak_idx / self.sampling_rate * 1000
                    peak_amplitude = signal[peak_idx]

                    if vertical_threshold > 0:
                        is_vertical = peak_amplitude >= vertical_threshold
                    else:
                        is_vertical = peak_amplitude <= vertical_threshold

                    if is_vertical:
                        vertical_peaks.append((peak_time, peak_amplitude))
                    else:
                        other_peaks.append((peak_time, peak_amplitude))

                # Store peaks for this electrode
                self.peaks[electrode] = (vertical_peaks, other_peaks)
                self.plot_canvas.vertical_amplitudes_info[electrode] = vertical_peaks + other_peaks

            # Update plot with new peaks
            time = np.arange(len(self.electrode_data[self.selected_electrodes[0]])) / self.sampling_rate * 1000
            plot_data = pd.DataFrame({
                electrode: (self.filtered_signals[electrode] 
                        if self.filtered_signals.get(electrode) is not None 
                        else self.electrode_data[electrode])
                for electrode in self.selected_electrodes
            })

            self.plot_canvas.plot(time, plot_data, self.electrode_config)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to find peaks: {str(e)}")
    

    def compute_windows(self):
        """Compute activation windows based on vertical peaks only"""
        try:
            vertical_threshold = self.control_panel.vertical_threshold_input.value()
            
            for electrode in self.selected_electrodes:
                if electrode in self.peaks:
                    vertical_peaks, _ = self.peaks[electrode]
                    windows, isi_min, isi_80_percent = calculate_activation_windows(vertical_peaks)
                    self.windows[electrode] = windows
            
            # Update plot with computed windows
            time = np.arange(len(self.electrode_data[self.selected_electrodes[0]])) / self.sampling_rate * 1000
            plot_data = pd.DataFrame({
                electrode: (self.filtered_signals[electrode] 
                        if self.filtered_signals.get(electrode) is not None 
                        else self.electrode_data[electrode])
                for electrode in self.selected_electrodes
            })
            
            self.plot_canvas.windows = self.windows
            self.plot_canvas.plot(time, plot_data, self.electrode_config)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to compute windows: {str(e)}")

    
    def reset_plots(self):
        """Reset all analysis and plots"""
        self.filtered_signals = {electrode: None for electrode in self.selected_electrodes}
        self.peaks = {electrode: None for electrode in self.selected_electrodes}
        self.windows = {electrode: None for electrode in self.selected_electrodes}
        self.active_filters = {filter_name: False for filter_name in self.active_filters}
        
        # Reset control panel buttons
        self.control_panel.butter_button.setChecked(False)
        self.control_panel.savgol_button.setChecked(False)
        self.control_panel.moving_avg_button.setChecked(False)
        
        # Reset plot canvas data
        if self.plot_canvas:
            self.plot_canvas.vertical_amplitudes_info = {}
            self.plot_canvas.windows = {}
            
            # Update plot with original data
            time = np.arange(len(self.electrode_data[self.selected_electrodes[0]])) / self.sampling_rate * 1000
            plot_data = pd.DataFrame({
                electrode: self.electrode_data[electrode] 
                for electrode in self.selected_electrodes
            })
            self.plot_canvas.plot(time, plot_data, self.electrode_config)

    
    def update_vertical_threshold(self, value):
        #Update vertical threshold in plot canvas
        if self.plot_canvas:
            self.plot_canvas.update_vertical_threshold(value)
            self.find_peaks()  # Refresh peaks with new threshold

    
    
            
    def initialize_data(self):
        """Initialize DataFrame with all signals to improve performance"""
        # Create time array once
        self.time = np.arange(len(self.electrode_data[self.selected_electrodes[0]])) / self.sampling_rate * 1000
        
        # Create DataFrame with all signals at once
        data_dict = {}
        for electrode in self.selected_electrodes:
            data_dict[electrode] = self.electrode_data[electrode]
        
        self.plot_data = pd.DataFrame(data_dict)
        
        # Create filtered data DataFrame
        self.filtered_data = pd.DataFrame(index=self.plot_data.index)


    def setup_ui(self):
        """Setup main window UI"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Create plot widget
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)
        main_layout.addWidget(self.plot_widget)

    
    def toggle_filter(self, filter_name, checked):
        """Toggle filter and update plots"""
        self.active_filters[filter_name] = checked
        self.update_filtered_signals()

    
#in order to share peaks information with other classes
    def get_peaks(self):
        """Public method to access the peaks data."""
        if not hasattr(self, 'peaks') or self.peaks is None:
            raise ValueError("Peaks have not been computed yet.")
            
        # Convert peaks to format needed for heatmap
        processed_peaks = {}
        for electrode, peak_data in self.peaks.items():
            if peak_data is not None:
                vertical_peaks, other_peaks = peak_data
                # Convert time values back to sample indices
                indices = []
                for peak_time, _ in vertical_peaks + other_peaks:
                    idx = int(peak_time * self.sampling_rate / 1000)
                    indices.append(idx)
                processed_peaks[electrode] = np.array(sorted(indices))
        
        return processed_peaks
