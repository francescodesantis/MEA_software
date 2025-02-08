import sys
import numpy as np
import pandas as pd
from threading import Thread
import neurokit2 as nk
from scipy.signal import resample
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QWidget,
    QMessageBox, QLabel, QMenuBar, QMenu, QDialog, QLineEdit,
    QPushButton, QTableView, QHeaderView, QTableWidget, QTableWidgetItem,
    QComboBox, QListWidget, QDialogButtonBox, QGroupBox, QCheckBox,
    QInputDialog, QHBoxLayout, QListWidgetItem
)
from PyQt5.QtWidgets import QRadioButton, QDialog, QVBoxLayout, QListWidget, QLabel, QDialogButtonBox, QMessageBox
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QSlider, QCheckBox, QComboBox, 
                             QListWidget, QDialog, QDialogButtonBox, QMessageBox)
from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QToolBar, QLabel, 
    QSpinBox, QPushButton, QProgressDialog, QFileDialog
)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QAbstractTableModel
from ipywidgets import interact, Dropdown, FileUpload, VBox
from file_import import DataLoadingThread
from plot import PlotCanvas
from electrode_config import ElectrodeConfig
from signal_analysis import SignalAnalyzer
from signal_averager import WaveformAnalyzer, IntervalSelectionDialog
from global_analysis import GlobalAnalyzer
from exclude_electrodes import ManuallyExcludeElectrodes
from propagation import PropagationHeatmap
from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QToolBar, QLabel, 
    QSpinBox, QPushButton, QProgressDialog, QFileDialog
)
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtChart import (
    QChart, QChartView, QScatterSeries, QLineSeries,
    QValueAxis
)
from PyQt5.QtGui import QColor, QGradient, QLinearGradient, QPainter
import traceback
import os
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Class used to display loading window while data is being imported
class LoadingDialog(QDialog):
    def __init__(self, message="Loading, please wait..."):
        super().__init__()
        self.setWindowTitle("Loading")
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.setModal(True)

        layout = QVBoxLayout()
        label = QLabel(message)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        self.setLayout(layout)
        self.setFixedSize(200, 150)

# Main window class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Cardiomyocytes software v0.1')
        self.setGeometry(100, 100, 1200, 800)

        # Data variables
        self.time = None
        self.electrode_data = None
        self.sorted_electrode_headers = None
        self.selected_headers = {}
        self.original_data = None
        self.original_time = None
        self.peaks = {}
        self.sampleFreq = None
        self.avg_rates = None
        self.p_peaks = {}
        self.t_peaks = {}
        self.times = {}  # Dizionario per memorizzare i tempi dei picchi
        self.min_peak_height = None
        self.is_filtered = False
        # Menu Setup
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)

        # Import file and exit options
        file_menu = QMenu("File", self)
        self.menu_bar.addMenu(file_menu)
        open_action = file_menu.addAction("Open File")
        open_action.triggered.connect(self.open_file)
        plot_action = file_menu.addAction("Plot Original Data")  # Plot File
        plot_action.triggered.connect(self.plot_original_data)
        file_menu.addAction("&Exit", self.close)
        exclude_electrodes_action = file_menu.addAction("Exclude Electrodes")  
        exclude_electrodes_action.triggered.connect(self.man_exclude_electrodes)

        # Menu Signal Analysis
        signal_analysis_menu = QMenu("Signal Analysis", self)
        self.menu_bar.addMenu(signal_analysis_menu)

        # Add action to open Electrode Selection
        signal_analysis_menu.addAction("Select Electrodes", self.show_electrodes_selection)

        # Add action to open Global Analysis
        signal_analysis_menu.addAction("Global Analysis", self.show_global_analysis)


        # Add action to open Single Channel Analysis
        signal_analysis_menu.addAction("Single Channel Analysis", self.show_signal_analysis)

        
        # Data comparison menu 
        data_comparison_menu=QMenu("Data Comparison", self)
        self.menu_bar.addMenu(data_comparison_menu)
        
        
        #Averager menu
        signal_average_menu=QMenu("Averaging", self)
        self.menu_bar.addMenu(signal_average_menu)

        #Add a new action for averaged signal plotting
        averaged_signal_action=signal_average_menu.addAction("Average signal")
        averaged_signal_action.triggered.connect(self.show_waveform_analysis)

        # Menu Propagation analysis
        propagation_analysis_menu = QMenu("Propagation analysis", self)
        self.menu_bar.addMenu(propagation_analysis_menu)

        #Add a new action for heat map 
        heat_map_action=propagation_analysis_menu.addAction("Heat Map")
        heat_map_action.triggered.connect(self.show_heat_map)

        # Handle plotting
        self.plot_canvas = PlotCanvas(self, width=12, height=8)

        # Organize plot layout
       
        layout = QVBoxLayout()
        self.results_table = QTableWidget()
        self.fileName = QLabel("No file imported")
        self.fileName.setFixedWidth(300)
        self.fileName.setWordWrap(True)
        layout.addWidget(self.fileName)
        layout.addWidget(self.plot_canvas)
        layout.addWidget(self.plot_canvas.toolbar)
        layout.addWidget(self.results_table)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Customize style
        self.menu_bar.setStyleSheet("""
            QMenuBar {
                background-color: darkred;
                color: #ffffff;
            }
            QMenuBar::item:selected {
                background-color: #ccbcc8;
                color: #000000;
            }
        """)
        self.setStyleSheet("background-color: lightpink; color: black;")

    #FILE BUTTON

    def open_file(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Open Data File", "", "Text Files (*.txt);; All files (*.*)") # Allow choice of file type
        #_ = we only care about file path and not filter, so we use _ to ignore it
        # here self is used to ensure that dialog is child of main window, it refers to instance of the class
        # "" is initial directory, here empty to start in default directory

        # If file is selected, loading window shows up
        if file_path:
            loading_dialog = LoadingDialog("Loading, please wait...")
            loading_dialog.show() # Execution of dialog window, which blocks main window interactions
            QApplication.processEvents()

            # Load data in background
            data_loader = DataLoadingThread(file_path)
            data_loader.data_loaded.connect(self.on_data_loaded) # Triggers next function after data loading is complete
            data_loader.start()
            data_loader.wait() # Wait for thread to finish before proceeding
            file_name = os.path.basename(file_path)
            self.fileName.setText(f"Imported file: {file_name}")
            loading_dialog.close()


    # After data is loaded allow some functions:
    def on_data_loaded(self, time, electrode_data, sorted_electrode_headers):
        self.original_data = electrode_data.copy() # Store original data for future reference
        self.original_time = time.copy()
        self.sorted_electrode_headers = sorted_electrode_headers.copy()

        self.time = time
        self.electrode_data = electrode_data
        self.sorted_electrode_headers = sorted_electrode_headers
        self.selected_headers = self.sorted_electrode_headers # Initially have all headers
        
        # Populate the Electrode column in the table
        self.results_table.setColumnCount(1)  # Start with just the Electrode column
        self.results_table.setHorizontalHeaderItem(0, QTableWidgetItem("Electrode"))
        self.results_table.setRowCount(len(self.sorted_electrode_headers))
        for row, electrode in enumerate(self.sorted_electrode_headers):
            self.results_table.setItem(row, 0, QTableWidgetItem(str(electrode)))

        if self.electrode_data is not None:
            #self.man_exclude_action.setDisabled(False) # Enable exclusion of electrodes
            #self.auto_exclude_action.setDisabled(False) # Enable frequency-based exclusion

            # Plot all electrodes by default
            self.selected_headers = sorted_electrode_headers  
            
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Data imported successfully!")
            msg.setWindowTitle("Success")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        else:
            print('Failed to load electrode data')

    def close_file(self): # Close GUI Message Box when selecting Exit in File menu
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText("Are you sure you want to exit?")
        msg.setWindowTitle("Exit")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        answer = msg.exec_()
        if answer == QMessageBox.Yes:
            self.close()
    
    def closeEvent(self, event): # Close GUI Message Box when clicking the X button
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText("Are you sure you want to exit?")
        msg.setWindowTitle("Exit")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        answer = msg.exec_()
        if answer == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    # PLOT DATA

    def plot_original_data(self):
        if self.original_data is None:
            print("no data")
            return
        self.on_data_loaded(self.original_time, self.original_data, self.sorted_electrode_headers)
        self.plot_data()

    
    def plot_data(self):
        if self.electrode_data is None:
            print("no data")
            return
        if self.time is not None and self.electrode_data is not None:
            loading_dialog = LoadingDialog("Loading, please wait...")
            loading_dialog.show()  # Execution of dialog window, which blocks main window interactions
            QApplication.processEvents()

            if not self.selected_headers:
                self.selected_headers = self.sorted_electrode_headers  # Default to plot all electrodes if none selected
            
            # First plot the selected electrodes data
            selected_data = self.electrode_data[self.selected_headers]  # Take data from only selected electrodes
            electrode_config = ElectrodeConfig(selected_data)  # Get electrode configuration
            self.plot_canvas.plot(self.time, selected_data, electrode_config=electrode_config)
            
            # Now, if peaks have been calculated, update the plot with those peaks
            if hasattr(self.plot_canvas, 'peaks') and self.plot_canvas.peaks:
                self.plot_canvas.update_peaks(self.plot_canvas.peaks)  # This will update the plot with the detected peaks
            
            loading_dialog.close()

        else:
            print("Data not loaded yet.")
    
    def update_plot(self): # Update plot based on selected electrodes.
        print("Updating plot with headers:", self.selected_headers)
        self.plot_data()

    
    #EXCLUDE ELECTRODES
    def man_exclude_electrodes(self):
        if self.electrode_data is None:
            
            return
        dialog = ManuallyExcludeElectrodes(self.sorted_electrode_headers, self.selected_headers, self)
        dialog.electrodes_remaining.connect(self.apply_exclusion)
        if dialog.exec_() == QDialog.Accepted:
            pass


    # Apply exclusion of electrodes
    def apply_exclusion(self, selected_headers):
        try:
            self.selected_headers = sorted(selected_headers)
            # Update electrode data based on selected headers. Analysis will be performed on this data only
            self.electrode_data = self.electrode_data[self.selected_headers]
            
            # Get all existing rows and map them to electrodes
            electrode_to_row = {self.results_table.item(row, 0).text(): row 
                        for row in range(self.results_table.rowCount())
                        if self.results_table.item(row, 0) is not None}
            
            # Collect data for electrodes to keep
            remaining_data = {
                electrode: [
                    self.results_table.item(row, col).text() if self.results_table.item(row, col) else ""
                    for col in range(self.results_table.columnCount())
                ]
                for electrode, row in electrode_to_row.items()
                if electrode in self.selected_headers
            }

            # Clear the entire table and reset the number of rows
            self.results_table.setRowCount(0)
            self.results_table.setRowCount(len(self.selected_headers))

            # Repopulate the table with the remaining data
            for new_row, electrode in enumerate(self.selected_headers):
                # Update the Electrode column
                self.results_table.setItem(new_row, 0, QTableWidgetItem(electrode))

                # Update the analysis values
                if electrode in remaining_data:
                    for col, value in enumerate(remaining_data[electrode][1:], start=1):  # Skip Electrode column
                        self.results_table.setItem(new_row, col, QTableWidgetItem(value))

            self.update_plot()  # Update plot based on selected electrodes
        except KeyError as e:
            # Display error message to user
            error_message = f"You can't retrieve the excluded electrodes. Re-plot all the original data."
            
            # Create and display a message box
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("Electrode Not Found")
            msg_box.setText("Error: Some selected electrodes are missing.")
            msg_box.setInformativeText(error_message)
            msg_box.exec_()


    #SIGNAL ANALYSIS

    #ELECTRODES SELECTION
    def show_electrodes_selection(self):
        """
        Displays a dialog to select electrodes and stores the selected electrodes.
        Includes a 'Select All' checkbox for bulk selection.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Electrodes")
        
        # Create Select All checkbox
        select_all_checkbox = QCheckBox("Select All Electrodes")
        
        electrode_list = QListWidget(dialog)
        electrode_list.setSelectionMode(QListWidget.MultiSelection)
        for electrode in self.electrode_data.columns:
            electrode_list.addItem(electrode)
            
        # Connect checkbox to selection function
        def toggle_all_selection(state):
            for i in range(electrode_list.count()):
                item = electrode_list.item(i)
                item.setSelected(state == Qt.Checked)
                
        select_all_checkbox.stateChanged.connect(toggle_all_selection)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal,
            dialog
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select Electrodes for Signal Analysis"))
        layout.addWidget(select_all_checkbox)  # Add checkbox to layout
        layout.addWidget(electrode_list)
        layout.addWidget(buttons)
        
        if dialog.exec_() == QDialog.Accepted:
            selected_electrodes = [item.text() for item in electrode_list.selectedItems()]
            if not selected_electrodes:
                QMessageBox.warning(self, "No Selection", "No electrodes selected!")
            else:
                self.selected_electrodes = selected_electrodes
        else:
            self.selected_electrodes = None

  
   
    #GLOBAL ANALYSIS
    def show_global_analysis(self):
       
        # Check if electrodes have been selected
        if not hasattr(self, 'selected_electrodes') or not self.selected_electrodes:
            QMessageBox.warning(self, "No Electrodes", "Please select electrodes first!")
            return
        
        try:
            # Create the global analyzer window for the stored selected electrodes
            self.global_analyzer = GlobalAnalyzer(
                electrode_data=self.electrode_data,
                selected_electrodes=self.selected_electrodes,
                sampling_rate=10000
            )
            self.global_analyzer.show()
        

            #in order to use peaks information for other purposes
            peaks = self.global_analyzer.get_peaks()
            print("Detected Peaks:", peaks)  # For debugging or further processing

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to initialize global analysis: {str(e)}"
            )

        
    #SINGLE CHANNEL ANALYSIS
    def show_signal_analysis(self):
        """
        Handles signal analysis using previously selected electrodes.
        """
        if not hasattr(self, 'selected_electrodes') or not self.selected_electrodes:
            QMessageBox.warning(self, "No Electrodes", "Please select electrodes first!")
            return
        
        # Create single analyzer window for the stored selected electrodes
        self.signal_analyzer = SignalAnalyzer(
            self.electrode_data,
            self.selected_electrodes,
            sampling_rate=10000
        )
        self.signal_analyzer.show()
        
    #AVERAGER
 
    def show_waveform_analysis(self):
        """Launch waveform analysis with interval selection support"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Waveform Analysis Setup")
        
        layout = QVBoxLayout(dialog)
        
        # Check if any analysis results are available
        has_signal_analyzer = hasattr(self, 'signal_analyzer')
        has_global_analyzer = hasattr(self, 'global_analyzer')
        
        if not has_signal_analyzer and not has_global_analyzer:
            QMessageBox.warning(
                self, 
                "Analysis Required",
                "Neither Signal Analyzer nor Global Analyzer results are available. "
                "Please perform either single channel or global analysis before computing the averaging."
            )
            return
        
        # Analysis type selection
        analysis_type_group = QWidget()
        analysis_type_layout = QHBoxLayout(analysis_type_group)
        
        analysis_label = QLabel("Select Analysis Type:")
        analysis_type_layout.addWidget(analysis_label)
        
        signal_radio = QRadioButton("Single Channel Analysis")
        signal_radio.setEnabled(has_signal_analyzer)
        global_radio = QRadioButton("Global Analysis")
        global_radio.setEnabled(has_global_analyzer)
        
        # Set default selection logic
        if has_signal_analyzer and has_global_analyzer:
            # If both available, default to single channel
            signal_radio.setChecked(True)
        elif has_signal_analyzer:
            signal_radio.setChecked(True)
        elif has_global_analyzer:
            global_radio.setChecked(True)
        
        analysis_type_layout.addWidget(signal_radio)
        analysis_type_layout.addWidget(global_radio)
        layout.addWidget(analysis_type_group)
        
        # Data type selection
        data_type_group = QWidget()
        data_type_layout = QHBoxLayout(data_type_group)
        
        data_label = QLabel("Select Data Type:")
        data_type_layout.addWidget(data_label)
        
        raw_radio = QRadioButton("Raw Data")
        filtered_radio = QRadioButton("Filtered Data")
        raw_radio.setChecked(True)  # Default to raw data
        
        data_type_layout.addWidget(raw_radio)
        data_type_layout.addWidget(filtered_radio)
        layout.addWidget(data_type_group)
        
        # Electrode selection (using previously selected electrodes)
        if not hasattr(self, 'selected_electrodes') or not self.selected_electrodes:
            QMessageBox.warning(self, "No Electrodes", "No electrodes have been selected. Please select electrodes first.")
            return
        
        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, dialog
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec_() == QDialog.Accepted:
            # Determine which analyzer to use
            using_signal_analyzer = signal_radio.isChecked()
            analyzer = self.signal_analyzer if using_signal_analyzer else self.global_analyzer
            self.selected_analysis_type = 'single' if signal_radio.isChecked() else 'global'
            analyzer = self.signal_analyzer if signal_radio.isChecked() else self.global_analyzer
    
            use_filtered_data = filtered_radio.isChecked()
            
            # Prepare raw data dictionary
            raw_data = {
                electrode: self.electrode_data[electrode].to_numpy()
                for electrode in self.selected_electrodes
            }
            
            # Prepare filtered data dictionary
            filtered_data = {}
            #filtered_attr = 'filtered_signals' if using_signal_analyzer else 'filtered_signal'
            filtered_attr = 'filtered_signals'

            if use_filtered_data and hasattr(analyzer, filtered_attr):
                filtered_source = getattr(analyzer, filtered_attr)
                for electrode in self.selected_electrodes:
                    if electrode in filtered_source:
                        filtered_data[electrode] = filtered_source[electrode]
                
                # Revert to raw if no filtered data found
                if not filtered_data:
                    use_filtered_data = False
            
            # Get peaks and windows
            try:
                peaks_info = {e: analyzer.peaks[e] for e in self.selected_electrodes}
                windows = {e: analyzer.windows[e] for e in self.selected_electrodes}
            except KeyError as e:
                QMessageBox.warning(
                    self,
                    "Missing Analysis Data",
                    f"Missing analysis data for electrode {str(e)}. "
                    "Please ensure analysis is complete for all selected electrodes."
                )
                return
            
            # Create WaveformAnalyzer instance
            self.waveform_analyzer = WaveformAnalyzer(
                electrode_data=raw_data,
                filtered_data=filtered_data if use_filtered_data else None,
                peaks_info=peaks_info,
                windows=windows
            )

            """self.waveform_analyzer = WaveformAnalyzer(
            electrode_data=raw_data,
            filtered_data=filtered_data if use_filtered_data else None,
            peaks_info=peaks_info,
            windows=windows,
            force_analysis_type='single'  # Pass the analysis type directly
            )"""

            
            # Set data type
            self.waveform_analyzer.set_data_type(use_filtered_data)
            self.waveform_analyzer.show()
            
    
    
    #HEAT MAP
    def show_heat_map(self):
        # Initialize progress dialog as class member
        self.progress = QProgressDialog("Generating heatmap...", "Cancel", 0, 100, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.show()
        QApplication.processEvents()

        try:
            # Check if we have a signal analyzer or global analyzer
            using_signal_analyzer = hasattr(self, 'signal_analyzer')
            using_global_analyzer = hasattr(self, 'global_analyzer')

            if not using_signal_analyzer and not using_global_analyzer:
                QMessageBox.warning(self, "Warning", "No analyzer found. Please run analysis first.")
                return

            # Get peaks using the appropriate getter
            try:
                if using_signal_analyzer:
                    peaks_data = self.signal_analyzer.get_peaks()
                    print("Retrieved peaks from signal analyzer")
                else:
                    peaks_data = self.global_analyzer.get_peaks()
                    print("Retrieved peaks from global analyzer")

                print(f"\nDEBUG: Retrieved peaks data structure:")
                print(f"Number of electrodes with peaks: {len(peaks_data)}")
                for electrode, peaks in peaks_data.items():
                    print(f"Electrode {electrode}: {len(peaks)} peaks")
                    if len(peaks) > 0:
                        print(f"First few peaks: {peaks[:5]}")

            except ValueError as e:
                QMessageBox.warning(self, "Warning", str(e))
                return

            # Verify we have enough valid electrodes
            valid_electrodes = [e for e, p in peaks_data.items() if len(p) > 0]
            if len(valid_electrodes) < 2:
                QMessageBox.warning(
                    self, 
                    "Warning", 
                    f"Found peaks in {len(valid_electrodes)} electrodes. Need at least 2 electrodes with peaks.\n"
                    "Try adjusting peak detection parameters (prominence, threshold)."
                )
                return

            # Create the heatmap window
            self.heatmap_window = QMainWindow()
            self.heatmap_window.setWindowTitle("Signal Propagation Heatmap")
            self.heatmap_window.resize(800, 600)

            # Create central widget and layout
            central_widget = QWidget()
            self.heatmap_window.setCentralWidget(central_widget)
            main_layout = QVBoxLayout(central_widget)

            # Create toolbar for controls
            toolbar_widget = QWidget()
            toolbar_layout = QHBoxLayout(toolbar_widget)
            main_layout.addWidget(toolbar_widget)

            # Add control widgets
            window_size_label = QLabel("Window Size (ms):")
            toolbar_layout.addWidget(window_size_label)

            window_size_input = QSpinBox()
            window_size_input.setRange(1, 1000)
            window_size_input.setValue(200)  # Default 200ms window
            toolbar_layout.addWidget(window_size_input)

            percentage_label = QLabel("Min Peak % Required:")
            toolbar_layout.addWidget(percentage_label)

            percentage_input = QSpinBox()
            percentage_input.setRange(1, 100)
            percentage_input.setValue(80)  # Default 80%
            toolbar_layout.addWidget(percentage_input)

            # Create PlotCanvas instance
            self.plot_canvas = PlotCanvas(central_widget)
            main_layout.addWidget(self.plot_canvas)

            # Create the heatmap instance
            self.heatmap = PropagationHeatmap(
                electrode_data=self.electrode_data,
                peaks_dict=peaks_data,
                plot_canvas=self.plot_canvas,
                window_size=window_size_input.value(),
                percentage=percentage_input.value()
            )

            def refresh_heatmap():
                if hasattr(self, 'progress') and self.progress:
                    self.progress.setValue(0)
                    self.progress.show()
                
                # Update heatmap parameters
                self.heatmap.window_size = window_size_input.value()
                self.heatmap.percentage = percentage_input.value()
                
                # Clear the current plot
                self.plot_canvas.fig.clear()
                
                # Start new analysis
                self.heatmap.start()

            # Add refresh button
            refresh_button = QPushButton("Refresh")
            refresh_button.clicked.connect(refresh_heatmap)
            toolbar_layout.addWidget(refresh_button)

            # Add export button
            export_button = QPushButton("Export")
            
            def export_heatmap():
                filename, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save Heatmap",
                    "",
                    "PNG Files (*.png);;All Files (*)"
                )
                if filename:
                    self.plot_canvas.fig.savefig(filename, dpi=300, bbox_inches='tight')

            export_button.clicked.connect(export_heatmap)
            toolbar_layout.addWidget(export_button)

            # Add spacer to toolbar
            toolbar_layout.addStretch()

            def update_heatmap(heatmap_data):
                try:
                    if hasattr(self, 'progress') and self.progress:
                        self.progress.setValue(100)
                        self.progress.close()
                        self.progress = None
                except Exception as e:
                    print(f"Error in update_heatmap: {str(e)}")
                    if hasattr(self, 'progress') and self.progress:
                        self.progress.close()
                        self.progress = None

            def show_error(message):
                if hasattr(self, 'progress') and self.progress:
                    self.progress.close()
                    self.progress = None
                if hasattr(self, 'heatmap_window') and self.heatmap_window:
                    self.heatmap_window.close()
                QMessageBox.critical(self, "Error", message)

            # Connect signals
            self.heatmap.propagation.connect(update_heatmap)
            self.heatmap.error.connect(show_error)

            # Show window and start processing
            self.heatmap_window.show()
            QApplication.processEvents()
            self.heatmap.start()

        except Exception as e:
            if hasattr(self, 'progress') and self.progress:
                self.progress.close()
                self.progress = None
            
            print("An error occurred while generating the heatmap:")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to generate heatmap: {str(e)}")

            
    #TABLE VISUALIZATION AND SAVING

    def show_table(self, df):
        """Displays the DataFrame in a dialog with the option to save as CSV."""
        dialog = QDialog()
        dialog.setWindowTitle("Results Table")

        layout = QVBoxLayout()

        # Set up the table widget
        table_widget = QTableWidget()
        table_widget.setRowCount(len(df))  # Imposta il numero di righe
        table_widget.setColumnCount(len(df.columns))  # Imposta il numero di colonne

        # Imposta le intestazioni delle colonne
        table_widget.setHorizontalHeaderLabels(df.columns.tolist())

        # Popola il table widget con i dati del DataFrame
        for row in range(len(df)):
            for column in range(len(df.columns)):
                item = QTableWidgetItem(str(df.iat[row, column]))
                table_widget.setItem(row, column, item)

        layout.addWidget(table_widget)

        # Button to save as CSV
        save_button = QPushButton("Save as CSV")
        save_button.clicked.connect(lambda: self.save_csv(df))
        layout.addWidget(save_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def save_csv(self,df):
   
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(None, "Save CSV", "", "CSV Files (*.csv)")

        if file_path:
            try:
                df.to_csv(file_path, index=False)
                QMessageBox.information(None, "Success", "CSV file saved successfully!")
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Error during saving: {str(e)}")
        else:
            QMessageBox.information(None, "Cancelled", "Saving cancelled.")
        
        
# Application entry point
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())