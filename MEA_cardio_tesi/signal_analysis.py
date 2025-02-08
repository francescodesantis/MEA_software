from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QDialog, QListWidget, QDialogButtonBox, QMessageBox,
    QMainWindow, QDockWidget, QSlider, QCheckBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import scipy.signal 
from PyQt5.QtWidgets import QRadioButton

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

def calculate_activation_windows(vertical_peaks, isi_percentage=80):
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





from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QDialog, QListWidget, QDialogButtonBox, QMessageBox,
    QMainWindow, QDockWidget, QSlider, QCheckBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QSplitter, QWidget, QLabel

class SignalWindow(QMainWindow):
    def __init__(self, electrode, signal_data, sampling_rate, filtered_signal=None, peaks=None, windows=None):
        super().__init__()
        self.electrode = electrode
        self.signal_data = signal_data
        self.sampling_rate = sampling_rate
        self.filtered_signal = filtered_signal
        self.peaks = peaks
        self.windows = windows
        self.heart_rate = None

        self.horizontal_line_visible = True
        self.raw_signal_visible = True  # New flag for raw signal visibility
        self.horizontal_line_position = np.mean(signal_data)
        self.final_threshold_set = False
        self.dragging_line = False
        self.drag_sensitivity = 0.2  # Increased sensitivity further
        
        # Flag to track if threshold is confirmed
        self.threshold_confirmed = False

        self.setWindowTitle(f"{electrode} Signal")
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create matplotlib figure with navigation toolbar
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        
        main_layout.addWidget(self.nav_toolbar)
        main_layout.addWidget(self.canvas)
        
        # Create a horizontal layout for controls
        controls_layout = QHBoxLayout()

        # Add checkbox for raw signal visibility
        self.raw_signal_checkbox = QCheckBox("Show/Hide Raw Signal")
        self.raw_signal_checkbox.setChecked(self.raw_signal_visible)
        self.raw_signal_checkbox.stateChanged.connect(self.toggle_raw_signal)
        controls_layout.addWidget(self.raw_signal_checkbox)
        
        
        # Add checkbox for horizontal line visibility
        self.line_checkbox = QCheckBox("Show/Hide Threshold Line")
        self.line_checkbox.setChecked(self.horizontal_line_visible)
        self.line_checkbox.stateChanged.connect(self.toggle_horizontal_line)
        controls_layout.addWidget(self.line_checkbox)
        
        # Add confirm threshold button
        self.confirm_threshold_btn = QPushButton("Confirm Threshold")
        self.confirm_threshold_btn.clicked.connect(self.confirm_threshold)
        self.confirm_threshold_btn.setStyleSheet("""
            QPushButton { background-color: #f0f0f0; }
            QPushButton:checked { background-color: #90EE90; }
        """)
        controls_layout.addWidget(self.confirm_threshold_btn)
        
        # Add reset threshold button
        self.reset_threshold_btn = QPushButton("Reset Threshold")
        self.reset_threshold_btn.clicked.connect(self.reset_threshold)
        controls_layout.addWidget(self.reset_threshold_btn)
        
        main_layout.addLayout(controls_layout)
        
        # Add threshold value label
        self.threshold_label = QLabel("Current threshold: {:.2f}".format(self.horizontal_line_position))
        self.threshold_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.threshold_label)

        # Plot initial data
        self.plot_signal()

        # Connect event handlers for interactive line dragging
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_drag)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)

    
    def confirm_threshold(self):
        """Confirm the threshold and enable peak detection"""
        self.threshold_confirmed = True
        self.confirm_threshold_btn.setEnabled(False)
        self.confirm_threshold_btn.setStyleSheet("background-color: #90EE90;")  # Set to green when confirmed
        self.threshold_label.setText(f"Threshold confirmed at {self.horizontal_line_position:.2f}")
        
        # Emit a signal or call a method to trigger peak detection
        if hasattr(self, 'parent') and hasattr(self.parent, 'find_peaks'):
            self.parent.find_peaks()

    def reset_threshold(self):
        """Reset the threshold to its initial state"""
        self.threshold_confirmed = False
        self.confirm_threshold_btn.setEnabled(True)
        self.confirm_threshold_btn.setStyleSheet("background-color: #f0f0f0;")
        self.horizontal_line_position = np.mean(self.signal_data)
        self.threshold_label.setText("Current threshold: {:.2f}".format(self.horizontal_line_position))
        self.peaks = None  # Clear existing peaks
        self.plot_signal()
    
    def toggle_raw_signal(self, state):
        """Toggle the visibility of the raw signal"""
        self.raw_signal_visible = state == Qt.Checked
        self.plot_signal()

    def plot_signal(self):
        self.ax.clear()
        time = np.arange(len(self.signal_data)) / self.sampling_rate * 1000
        
        # Plot raw signal
        #self.ax.plot(time, self.signal_data, label='Raw Signal', color='blue', alpha=0.5, linewidth=1)

        # Plot raw signal only if visible
        if self.raw_signal_visible:
            self.ax.plot(time, self.signal_data, label='Raw Signal', color='blue', alpha=0.5, linewidth=1)
        
        # Plot filtered signal if available
        if self.filtered_signal is not None:
            self.ax.plot(time, self.filtered_signal, label='Filtered Signal', color='orange', linewidth=1.5)
        
        # Plot peaks if available
        plotted_labels = []
        if self.peaks is not None:
            vertical_peaks, other_peaks = self.peaks
            
            # Plot vertical peaks in red
            for peak_time, amplitude in vertical_peaks:
                self.ax.scatter(peak_time, amplitude, color='red', 
                              label='Vertical Peak' if 'Vertical Peak' not in plotted_labels else "",
                              zorder=5)
                if 'Vertical Peak' not in plotted_labels:
                    plotted_labels.append('Vertical Peak')
            
            # Plot other peaks in green
            for peak_time, amplitude in other_peaks:
                self.ax.scatter(peak_time, amplitude, color='green',
                              label='Other Peak' if 'Other Peak' not in plotted_labels else "",
                              zorder=5)
                if 'Other Peak' not in plotted_labels:
                    plotted_labels.append('Other Peak')
        
        # Plot activation windows if available
        if self.windows is not None:
            for start, end in self.windows:
                self.ax.axvspan(start, end, color='yellow', alpha=0.3,
                              label='Activation Window' if 'Activation Window' not in plotted_labels else "")
                if 'Activation Window' not in plotted_labels:
                    plotted_labels.append('Activation Window')
        
        # Plot the horizontal line if visible
        if self.horizontal_line_visible:
            self.horizontal_line = self.ax.axhline(
                self.horizontal_line_position, color='red', linestyle='--', linewidth=1.5, label='Threshold Line'
            )
        
        self.ax.set_xlabel('Time (ms)')
        self.ax.set_ylabel('Amplitude (uV)')
        self.ax.set_title(f'{self.electrode} Signal Analysis')
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()

        # Calculate and display heart rate if vertical peaks exist
        if self.peaks is not None:
            vertical_peaks, _ = self.peaks
            if len(vertical_peaks) > 1:
                # Sort peaks by time
                sorted_peaks = sorted(vertical_peaks, key=lambda x: x[0])
                
                # Calculate RR intervals (in seconds)
                rr_intervals = [
                    (sorted_peaks[i+1][0] - sorted_peaks[i][0]) / 1000  # Convert ms to seconds
                    for i in range(len(sorted_peaks) - 1)
                ]
                
                # Calculate average RR interval and heart rate
                avg_rr_interval = np.mean(rr_intervals)
                self.heart_rate = 60 / avg_rr_interval if avg_rr_interval > 0 else None
                
                # Add heart rate to the plot
                if self.heart_rate is not None:
                    heart_rate_text = f'Heart Rate: {self.heart_rate:.1f} bpm'
                    self.ax.text(0.02, 0.98, heart_rate_text, 
                                 transform=self.ax.transAxes, 
                                 verticalalignment='top', 
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        self.canvas.draw()
        
    def on_mouse_press(self, event):
        """Detect if the user clicked near the horizontal line with increased sensitivity"""
        if event.button == 1 and self.horizontal_line_visible and event.ydata is not None and event.inaxes:
            # Calculate the sensitivity zone based on the data range
            data_range = np.max(self.signal_data) - np.min(self.signal_data)
            sensitivity = data_range * self.drag_sensitivity
            
            if abs(event.ydata - self.horizontal_line_position) < sensitivity:
                self.dragging_line = True
                # Update the cursor style to indicate dragging
                self.canvas.setCursor(Qt.SizeVerCursor)
                # Update the line position immediately
                self.horizontal_line_position = event.ydata
                self.threshold_label.setText("Current threshold: {:.2f}".format(self.horizontal_line_position))
                self.plot_signal()

    def on_mouse_drag(self, event):
        """Update the position of the line while dragging"""
        if self.dragging_line and event.ydata is not None and event.inaxes:
            self.horizontal_line_position = event.ydata
            self.threshold_label.setText("Current threshold: {:.2f}".format(self.horizontal_line_position))
            self.plot_signal()

    def on_mouse_release(self, event):
        """Stop dragging the line when the mouse button is released"""
        if self.dragging_line:
            self.dragging_line = False
            # Reset cursor style
            self.canvas.setCursor(Qt.ArrowCursor)
            # Enable the confirm button if it was disabled
            self.confirm_threshold_btn.setEnabled(True)
            self.confirm_threshold_btn.setStyleSheet("background-color: #f0f0f0;")
            self.threshold_confirmed = False

    def toggle_horizontal_line(self, state):
        self.horizontal_line_visible = state == Qt.Checked
        self.plot_signal()
    
    def update_peaks(self):
        """Recalculate peaks based on the current threshold and signal"""
        if self.filtered_signal is not None:
            signal_to_use = self.filtered_signal
        else:
            signal_to_use = self.signal_data
        
        try:
            # Find peaks based on the vertical threshold
            peaks_pos, _ = find_peaks(signal_to_use, prominence=10)  # Default prominence
            peaks_neg, _ = find_peaks(-signal_to_use, prominence=10)
            
            # Combine and sort all peaks
            final_peaks = np.concatenate((peaks_pos, peaks_neg))
            final_peaks = np.sort(final_peaks)
            
            # Classify peaks
            vertical_peaks = []
            other_peaks = []
            
            for peak_idx in final_peaks:
                peak_time = peak_idx / self.sampling_rate * 1000
                peak_amplitude = signal_to_use[peak_idx]
                
                if peak_amplitude >= self.horizontal_line_position:
                    vertical_peaks.append((peak_time, peak_amplitude))
                else:
                    other_peaks.append((peak_time, peak_amplitude))
            
            self.peaks = (vertical_peaks, other_peaks)
            self.plot_signal()
        except Exception as e:
            print(f"Error updating peaks: {str(e)}")

    def update_plot(self, filtered_signal=None, peaks=None, windows=None):
        self.filtered_signal = filtered_signal
        self.peaks = peaks
        self.windows = windows
        self.plot_signal()


# Update the SignalAnalyzer class to handle the confirmed threshold
class SignalAnalyzer(QMainWindow):
    def __init__(self, electrode_data, selected_electrodes, sampling_rate=10000):
        super().__init__()
        self.electrode_data = electrode_data
        self.selected_electrodes = selected_electrodes
        self.sampling_rate = sampling_rate
        
        # Dictionary to store filtered signals for each electrode
        self.filtered_signals = {electrode: None for electrode in selected_electrodes}
        self.peaks = {electrode: None for electrode in selected_electrodes}
        self.windows = {electrode: None for electrode in selected_electrodes}
        
        # Default values for automatic filtering
        self.auto_filter_params = {
            'savgol': {
                'window_length': 21,
                'polyorder': 3
            },
            'butterworth': {
                'cutoff_low': 100,
                'cutoff_high': 2000
            },
            'moving_average': {
                'window_size': 15
            }
        }
        
        # Track active filters
        self.active_filters = {
            'butterworth': False,
            'savgol': False,
            'moving_average': False
        }
        
        # Dictionary to store signal windows
        self.signal_windows = {}
        
        self.setWindowTitle("Signal Analysis")
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        # Add filtering mode selection
        mode_group = QWidget()
        mode_layout = QVBoxLayout(mode_group)
        mode_label = QLabel("Analysis Mode:")
        mode_label.setStyleSheet("font-weight: bold;")
        mode_layout.addWidget(mode_label)
        
        # Create radio buttons for all modes
        self.no_filter_radio = QRadioButton("No Filter Mode (Raw Data Analysis)")
        self.auto_radio = QRadioButton("Automatic Filtering")
        self.custom_radio = QRadioButton("Custom Filtering")
        
        # Add tooltips for each mode
        self.no_filter_radio.setToolTip("Analyze raw data directly without any filtering")
        self.auto_radio.setToolTip("Apply all filters automatically with optimized parameters")
        self.custom_radio.setToolTip("Customize and apply filters individually")
        
        # Add radio buttons to mode group
        mode_layout.addWidget(self.no_filter_radio)
        mode_layout.addWidget(self.auto_radio)
        mode_layout.addWidget(self.custom_radio)
        
        # Set No Filter as default
        self.no_filter_radio.setChecked(True)
        
        control_layout.addWidget(mode_group)
        
        # Create widgets for different modes
        
        # 1. No Filter Mode Message
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
        control_layout.addWidget(self.no_filter_message)
        
        # 2. Automatic Filtering Section
        self.auto_controls = QWidget()
        auto_layout = QVBoxLayout(self.auto_controls)
        
        self.auto_filter_button = QPushButton("Apply Automatic Filtering")
        self.auto_filter_button.clicked.connect(self.apply_automatic_filtering)
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
        
        control_layout.addWidget(self.auto_controls)
        
        # 3. Custom Filtering Section
        self.custom_controls = QWidget()
        custom_layout = QVBoxLayout(self.custom_controls)
        
        # Add Apply All Filters button
        self.apply_all_custom_button = QPushButton("Apply All Filters")
        self.apply_all_custom_button.clicked.connect(self.apply_all_custom_filters)
        self.apply_all_custom_button.setStyleSheet("""
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
        custom_layout.addWidget(self.apply_all_custom_button)
        
        # Group 1: Savitzky-Golay filter and parameters
        savgol_group = QVBoxLayout()
        self.savgol_button = QPushButton("Apply Savitzky-Golay Filter")
        self.savgol_button.setCheckable(True)
        self.savgol_button.clicked.connect(self.toggle_savgol)
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
        
        # Add spacing
        custom_layout.addSpacing(10)
        
        # Group 2: Butterworth filter and parameters
        butterworth_group = QVBoxLayout()
        self.butter_button = QPushButton("Apply Butterworth Filter")
        self.butter_button.setCheckable(True)
        self.butter_button.clicked.connect(self.toggle_butterworth)
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
        
        # Add spacing
        custom_layout.addSpacing(10)
        
        # Group 3: Moving average filter and parameter
        moving_avg_group = QVBoxLayout()
        self.moving_avg_button = QPushButton("Apply Moving Average")
        self.moving_avg_button.setCheckable(True)
        self.moving_avg_button.clicked.connect(self.toggle_moving_average)
        self.moving_avg_button.setStyleSheet("""
            QPushButton { background-color: #f0f0f0; }
            QPushButton:checked { background-color: #90EE90; }
        """)
        moving_avg_group.addWidget(self.moving_avg_button)
        
        self.mov_avg_window_slider = self.create_slider("Window:", 3, 101, 15)
        moving_avg_group.addWidget(self.mov_avg_window_slider)
        custom_layout.addLayout(moving_avg_group)
        
        
        control_layout.addWidget(self.custom_controls)
        
        # Peak Detection Parameters (always visible)
        peak_group = QVBoxLayout()
        peak_label = QLabel("Peak Detection Parameters")
        peak_label.setStyleSheet("font-weight: bold;")
        peak_group.addWidget(peak_label)
        
        self.pos_prominence_slider = self.create_slider("Positive Peak Prominence:", 1, 100, 10)
        self.neg_prominence_slider = self.create_slider("Negative Peak Prominence:", 1, 100, 10)
        peak_group.addWidget(self.pos_prominence_slider)
        peak_group.addWidget(self.neg_prominence_slider)
        control_layout.addLayout(peak_group)
        
        # Analysis buttons (always visible)
        buttons_layout = QVBoxLayout()
        
        self.peaks_button = QPushButton("Find Peaks")
        self.peaks_button.clicked.connect(self.find_peaks)
        buttons_layout.addWidget(self.peaks_button)
        
        self.windows_button = QPushButton("Compute Activation Windows")
        self.windows_button.clicked.connect(self.compute_windows)
        buttons_layout.addWidget(self.windows_button)
        
        self.reset_button = QPushButton("Reset Plots")
        self.reset_button.clicked.connect(self.reset_plots)
        buttons_layout.addWidget(self.reset_button)
        
        control_layout.addLayout(buttons_layout)
        main_layout.addWidget(control_panel)
        
        # Connect radio buttons to control visibility
        self.no_filter_radio.toggled.connect(self.update_mode)
        self.auto_radio.toggled.connect(self.update_mode)
        self.custom_radio.toggled.connect(self.update_mode)
        
        # Initialize visibility
        self.update_mode()
        
        # Create signal windows
        self.create_signal_windows()

    def update_mode(self):
        """Update the visibility of controls based on selected mode"""
        # Hide all mode-specific controls first
        self.no_filter_message.setVisible(False)
        self.auto_controls.setVisible(False)
        self.custom_controls.setVisible(False)
        
        # Show controls based on selected mode
        if self.no_filter_radio.isChecked():
            self.no_filter_message.setVisible(True)
            # Reset any active filters
            self.reset_filters()
        elif self.auto_radio.isChecked():
            self.auto_controls.setVisible(True)
            # Reset any active filters
            self.reset_filters()
        else:  # Custom mode
            self.custom_controls.setVisible(True)
            # Reset any active filters
            self.reset_filters()

    def find_peaks(self):
        """Modified to use raw or filtered signal based on mode"""
        try:
            pos_prominence = self.pos_prominence_slider.findChild(QSlider).value()
            neg_prominence = self.neg_prominence_slider.findChild(QSlider).value()
            
            for electrode in self.selected_electrodes:
                window = self.signal_windows[electrode]
                
                # Only process peaks if threshold is confirmed
                if not window.threshold_confirmed:
                    continue
                
                # Choose signal based on mode
                if self.no_filter_radio.isChecked():
                    signal = self.electrode_data[electrode]
                else:
                    # Use filtered signal if available, otherwise use raw signal
                    signal = self.filtered_signals[electrode] if self.filtered_signals[electrode] is not None else self.electrode_data[electrode]
                
                vertical_threshold = window.horizontal_line_position
                
                peaks_pos, _ = find_peaks(signal, prominence=pos_prominence)
                peaks_neg, _ = find_peaks(-signal, prominence=neg_prominence)
                
                final_peaks = np.concatenate((peaks_pos, peaks_neg))
                final_peaks = np.sort(final_peaks)
                
                vertical_peaks = []
                other_peaks = []
                
                for peak_idx in final_peaks:
                    peak_time = peak_idx / self.sampling_rate * 1000
                    peak_amplitude = signal[peak_idx]
                    
                    if peak_amplitude >= vertical_threshold:
                        vertical_peaks.append((peak_time, peak_amplitude))
                    else:
                        other_peaks.append((peak_time, peak_amplitude))
                
                self.peaks[electrode] = (vertical_peaks, other_peaks)
            
            self.update_all_windows()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to find peaks: {str(e)}")

    def apply_all_custom_filters(self):
        """Apply all filters in sequence using the current custom parameters"""
        try:
            for electrode in self.selected_electrodes:
                filtered_signal = self.electrode_data[electrode].copy()
                vertical_threshold = self.signal_windows[electrode].horizontal_line_position
                
                # 1. Apply Savitzky-Golay
                window_length = self.savgol_window_slider.findChild(QSlider).value()
                if window_length % 2 == 0:
                    window_length += 1
                polyorder = self.savgol_order_slider.findChild(QSlider).value()
                filtered_signal = apply_savgol_filter(filtered_signal, window_length, polyorder)
                
                # 2. Apply Butterworth
                cutoff_low = self.butter_low_slider.findChild(QSlider).value()
                cutoff_high = self.butter_high_slider.findChild(QSlider).value()
                filtered_signal = apply_butterworth_filter(
                    filtered_signal, 
                    self.sampling_rate,
                    cutoff_low,
                    cutoff_high,
                    vertical_threshold
                )
                
                # 3. Apply Moving Average
                window_size = self.mov_avg_window_slider.findChild(QSlider).value()
                filtered_signal = apply_moving_average_filter(filtered_signal, window_size)
                
                # Store the filtered signal
                self.filtered_signals[electrode] = filtered_signal
                
                # Set all filter buttons to checked state
                self.savgol_button.setChecked(True)
                self.butter_button.setChecked(True)
                self.moving_avg_button.setChecked(True)
                
                # Update active filters state
                self.active_filters['savgol'] = True
                self.active_filters['butterworth'] = True
                self.active_filters['moving_average'] = True
            
            self.update_all_windows()
            QMessageBox.information(self, "Success", "All filters applied successfully in sequence!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply filters: {str(e)}")

    def update_filter_mode(self):
        """Update the visibility of controls based on selected mode"""
        is_custom = self.custom_radio.isChecked()
        self.custom_controls.setVisible(is_custom)
        self.auto_filter_button.setVisible(not is_custom)
        
        # Reset filters when switching modes
        self.reset_filters()

    def apply_automatic_filtering(self):
        """Apply all filters in sequence with default parameters"""
        try:
            for electrode in self.selected_electrodes:
                filtered_signal = self.electrode_data[electrode].copy()
                
                # 1. Apply Savitzky-Golay
                filtered_signal = apply_savgol_filter(
                    filtered_signal,
                    self.auto_filter_params['savgol']['window_length'],
                    self.auto_filter_params['savgol']['polyorder']
                )
                
                # 2. Apply Butterworth
                vertical_threshold = self.signal_windows[electrode].horizontal_line_position
                filtered_signal = apply_butterworth_filter(
                    filtered_signal,
                    self.sampling_rate,
                    self.auto_filter_params['butterworth']['cutoff_low'],
                    self.auto_filter_params['butterworth']['cutoff_high'],
                    vertical_threshold
                )
                
                # 3. Apply Moving Average
                filtered_signal = apply_moving_average_filter(
                    filtered_signal,
                    self.auto_filter_params['moving_average']['window_size']
                )
                
                self.filtered_signals[electrode] = filtered_signal
            
            self.update_all_windows()
            QMessageBox.information(self, "Success", "Automatic filtering applied successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply automatic filtering: {str(e)}")

    def reset_filters(self):
        """Reset all filters and their states"""
        self.active_filters = {filter_name: False for filter_name in self.active_filters}
        self.filtered_signals = {electrode: None for electrode in self.selected_electrodes}
        
        # Reset button states
        self.butter_button.setChecked(False)
        self.savgol_button.setChecked(False)
        self.moving_avg_button.setChecked(False)
        
        self.update_all_windows()
    
    
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
    

    def create_signal_windows(self):
        for electrode in self.selected_electrodes:
            window = SignalWindow(
                electrode,
                self.electrode_data[electrode],
                self.sampling_rate
            )
            self.signal_windows[electrode] = window
            window.show()
    
    def update_all_windows(self):
        for electrode, window in self.signal_windows.items():
            window.update_plot(
                filtered_signal=self.filtered_signals[electrode],
                peaks=self.peaks[electrode],
                windows=self.windows[electrode]
            )

    
    def apply_all_active_filters(self, signal, electrode):
        filtered_signal = signal.copy()
        
        # Get the threshold value from the corresponding signal window
        vertical_threshold = self.signal_windows[electrode].horizontal_line_position
        
        # 1. First apply Savitzky-Golay if active
        if self.active_filters['savgol']:
            window_length = self.savgol_window_slider.findChild(QSlider).value()
            if window_length % 2 == 0:
                window_length += 1
            polyorder = self.savgol_order_slider.findChild(QSlider).value()
            filtered_signal = apply_savgol_filter(filtered_signal, window_length, polyorder)
        
        # 2. Then apply Butterworth if active
        if self.active_filters['butterworth']:
            cutoff_low = self.butter_low_slider.findChild(QSlider).value()
            cutoff_high = self.butter_high_slider.findChild(QSlider).value()
            filtered_signal = apply_butterworth_filter(filtered_signal, self.sampling_rate, 
                                                    cutoff_low, cutoff_high, vertical_threshold)
        
        # 3. Finally apply Moving Average if active
        if self.active_filters['moving_average']:
            window_size = self.mov_avg_window_slider.findChild(QSlider).value()
            filtered_signal = apply_moving_average_filter(filtered_signal, window_size)
        
        return filtered_signal

    # Update the update_filtered_signals method to pass the electrode parameter
    def update_filtered_signals(self):
        for electrode in self.selected_electrodes:
            if any(self.active_filters.values()):
                self.filtered_signals[electrode] = self.apply_all_active_filters(
                    self.electrode_data[electrode], 
                    electrode
                )
            else:
                self.filtered_signals[electrode] = None
        self.update_all_windows()

   
    def compute_windows(self):
        try:
            for electrode in self.selected_electrodes:
                if self.peaks[electrode] is None:
                    continue
                    
                vertical_peaks, _ = self.peaks[electrode]  # Use only vertical peaks for windows
                windows, _, _ = calculate_activation_windows(vertical_peaks)
                self.windows[electrode] = windows
            
            self.update_all_windows()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to compute activation windows: {str(e)}")
    

    
    def toggle_butterworth(self, checked):
        self.active_filters['butterworth'] = checked
        self.update_filtered_signals()
    
    def toggle_savgol(self, checked):
        self.active_filters['savgol'] = checked
        self.update_filtered_signals()
    
    def toggle_moving_average(self, checked):
        self.active_filters['moving_average'] = checked
        self.update_filtered_signals()
    
    def reset_plots(self):
        self.filtered_signals = {electrode: None for electrode in self.selected_electrodes}
        self.peaks = {electrode: None for electrode in self.selected_electrodes}
        self.windows = {electrode: None for electrode in self.selected_electrodes}
        self.active_filters = {filter_name: False for filter_name in self.active_filters}
        
        # Reset button states
        self.butter_button.setChecked(False)
        self.savgol_button.setChecked(False)
        self.moving_avg_button.setChecked(False)
        
        self.update_all_windows()
    
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
