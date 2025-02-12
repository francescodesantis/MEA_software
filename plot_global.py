import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PyQt5.QtWidgets import QPushButton, QLabel
import pandas as pd


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=12, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

        # Initialize variables
        self.axes = []
        self.expanded_ax = None
        self.time = None
        self.data = None
        self.vertical_amplitudes_info = {}
        self.windows = {}
        self.filtered_data = {}
        self.sampling_rate = 10000
        self.is_expanded = False
        self.vertical_threshold = 0
        
        # Add state variables for plotting options
        self.confidence_intervals = None
        self.show_peaks = False
        self.peaks_info = None
        self.show_amplitudes = False
        
        # Create toolbar but keep it hidden initially
        self.toolbar = NavigationToolbar(self, parent)
        self.toolbar.hide()
        
        # Connect events
        self.setup_connections()

    def plot(self, time_axis, data_df, electrode_config, show_peaks=False, peaks_info=None, 
             show_amplitudes=False, confidence_intervals=None):
        """Plot the data with peaks, amplitude labels, and confidence intervals"""
        # Store all state
        self.time = time_axis
        self.data = data_df
        self.electrode_config = electrode_config
        self.confidence_intervals = confidence_intervals
        self.show_peaks = show_peaks
        self.peaks_info = peaks_info
        self.show_amplitudes = show_amplitudes
        
        self._update_plot()

    def _update_plot(self):
        """Internal method to update the plot based on current state"""
        self.fig.clear()
        
        if self.is_expanded and hasattr(self, 'expanded_header'):
            self._plot_single_electrode(self.expanded_header)
        else:
            self._plot_grid()
            
        self.draw()

    def _plot_single_electrode(self, header):
        """Plot a single electrode in expanded view"""
        gs = GridSpec(1, 1, figure=self.fig)
        ax = self.fig.add_subplot(gs[0, 0])
        
        # Plot the electrode data
        ax.plot(self.time, self.data[header], 'b-', label='Mean')
        
        # Add confidence intervals
        if self.confidence_intervals is not None and header in self.confidence_intervals:
            lower_ci, upper_ci = self.confidence_intervals[header]
            ax.fill_between(self.time, lower_ci, upper_ci,
                          color='b', alpha=0.2, label='95% CI')
        
        # Add peaks and labels
        if self.show_peaks and self.peaks_info and header in self.peaks_info:
            self._add_peaks_and_labels(ax, self.peaks_info[header], self.show_amplitudes)
        
        # Set title and labels
        ax.set_title(f'Electrode {header}')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (µV)')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Always add legend
        ax.legend(fontsize=8, loc='upper right')
        
        # Show toolbar for navigation
        self.toolbar.show()
        
        # Adjust layout
        self.fig.tight_layout()

    def _plot_grid(self):
        """Plot grid of electrodes"""
        n_rows = (len(self.data.columns) + 3) // 4
        n_cols = min(4, len(self.data.columns))
        
        # Dynamically adjust figure size
        fig_height = max(6, n_rows * 2)
        self.fig.set_size_inches(12, fig_height)
        
        # Create subplots with proper spacing
        self.fig.subplots_adjust(hspace=0.6, wspace=0.3)
        
        self.axes = []  # Clear existing axes
        
        for idx, electrode in enumerate(self.data.columns):
            ax = self.fig.add_subplot(n_rows, n_cols, idx + 1)
            ax.header = electrode
            self.axes.append(ax)
            
            # Plot the waveform
            ax.plot(self.time, self.data[electrode], 'b-', label='Mean')
            
            # Add confidence intervals
            if self.confidence_intervals is not None and electrode in self.confidence_intervals:
                lower_ci, upper_ci = self.confidence_intervals[electrode]
                ax.fill_between(self.time, lower_ci, upper_ci,
                              color='b', alpha=0.2, label='95% CI')
                ax.legend(fontsize=6, loc='upper right')
            
            # Add peaks and labels
            if self.show_peaks and self.peaks_info and electrode in self.peaks_info:
                self._add_peaks_and_labels(ax, self.peaks_info[electrode], self.show_amplitudes)
            
            # Set title and labels
            ax.text(0.5, 0.9, f'{electrode}', horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes, fontsize=6)
            ax.set_xlabel('Time (ms)', fontsize=8)
            ax.set_ylabel('Amplitude (µV)', fontsize=8)
            
            ax.set_xticks(np.linspace(self.time[0], self.time[-1], num=3))
            ax.grid(True, linestyle='--', alpha=0.7)
        
        self.fig.tight_layout(pad=2.0)

    def on_click(self, event):
        """Handle double-click events"""
        if event.dblclick:
            if self.is_expanded:
                # Return to grid view
                self.is_expanded = False
                self.toolbar.hide()
                self._update_plot()
            else:
                # Check if click was within an axis
                for ax in self.axes:
                    if ax == event.inaxes:
                        self.is_expanded = True
                        self.expanded_header = ax.header
                        self._update_plot()
                        break
                        
    def _add_peaks_and_labels(self, ax, peaks, show_amplitudes):
        """Helper method to add peaks and labels to a plot"""
        for peak in peaks:
            color = 'red' if peak['type'] == 'positive' else 'green'
            ax.plot(peak['time'], peak['amplitude'], 'o', 
                    color=color, markersize=6)
            
            # Determine label text
            label_text = ''
            if 'name' in peak:
                label_text = peak['name']
            elif show_amplitudes:
                label_text = f"{peak['amplitude']:.1f}"
            
            if label_text:
                base_offset = 10 if peak['type'] == 'positive' else -15
                if show_amplitudes and 'name' in peak:
                    # Stack name and amplitude
                    ax.annotate(peak['name'],
                              xy=(peak['time'], peak['amplitude']),
                              xytext=(0, base_offset * 2),
                              textcoords='offset points',
                              ha='center', va='bottom' if peak['type'] == 'positive' else 'top',
                              fontsize=8,
                              bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
                    
                    ax.annotate(f"{peak['amplitude']:.1f}µV",
                              xy=(peak['time'], peak['amplitude']),
                              xytext=(0, base_offset),
                              textcoords='offset points',
                              ha='center', va='bottom' if peak['type'] == 'positive' else 'top',
                              fontsize=8,
                              bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
                else:
                    # Single label
                    ax.annotate(label_text,
                              xy=(peak['time'], peak['amplitude']),
                              xytext=(0, base_offset),
                              textcoords='offset points',
                              ha='center', va='bottom' if peak['type'] == 'positive' else 'top',
                              fontsize=8,
                              bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    
    def clear_plot(self):
        """Clear the plot canvas"""
        self.fig.clear()
        self.is_expanded = False
        self.toolbar.hide()
        self.draw()
    
    def setup_connections(self):
        """Setup all event connections"""
        if hasattr(self, '_cid'):
            self.mpl_disconnect(self._cid)
        self._cid = self.mpl_connect('button_press_event', self.on_click)
   
        
    def plot_electrode_data(self, ax, header, expanded=False):
        """Plot data for a single electrode with consistent peak detection visualization"""
        # Plot basic signal
        plot_time = self.time[:10000] if not expanded else self.time
        plot_signal = self.data[header][:10000] if not expanded else self.data[header]
        
        # Plot filtered signal in orange
        ax.plot(plot_time, plot_signal, 'orange', label='Filtered' if expanded else None, alpha=0.7)
        
        # Plot peaks only in expanded view
        if expanded and header in self.vertical_amplitudes_info:
            all_peaks = self.vertical_amplitudes_info[header]
            vertical_peaks = []
            other_peaks = []
            
            # Separate peaks based on threshold direction
            for time, amplitude in all_peaks:
                if self.vertical_threshold > 0:
                    if amplitude >= self.vertical_threshold:
                        vertical_peaks.append((time, amplitude))
                    else:
                        other_peaks.append((time, amplitude))
                else:
                    if amplitude <= self.vertical_threshold:
                        vertical_peaks.append((time, amplitude))
                    else:
                        other_peaks.append((time, amplitude))
            
            # Plot peaks
            if vertical_peaks:
                times, amplitudes = zip(*vertical_peaks)
                ax.plot(times, amplitudes, 'ro', label='R-Peaks', markersize=5)
            
            if other_peaks:
                times, amplitudes = zip(*other_peaks)
                ax.plot(times, amplitudes, 'go', label='Other Peaks', markersize=5)
        
        # Add activation windows if available
        if expanded and header in self.windows and self.windows[header]:
            window_added = False
            for start, end in self.windows[header]:
                ax.axvspan(start, end, color='yellow', alpha=0.2,
                        label='Activation Window' if not window_added else None)
                window_added = True
        
        # Set axis properties
        if not expanded:
            ax.set_ylim([-200, 300])
            ax.text(0.5, 0.9, f'{header}', horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes, fontsize=8)
        else:
            ax.set_title(f'Electrode {header}')
            
        ax.set_xlim(left=0)
        
        # Y-axis label handling
        if not expanded:
            # Show y-axis labels for leftmost plots and first/last electrode of second column
            grid_pos = self.electrode_config.header_to_coordinate(header)
            if grid_pos:
                col = grid_pos[0] // 210
                if col == 0:  # First column
                    ax.set_ylabel('A [uV]')
                elif col == 1:  # Second column
                    # Get all electrodes in the second column
                    second_col_electrodes = [h for h in self.data.columns 
                                          if self.electrode_config.header_to_coordinate(h) and 
                                          self.electrode_config.header_to_coordinate(h)[0] // 210 == 1]
                    second_col_y_coords = [self.electrode_config.header_to_coordinate(h)[1] 
                                         for h in second_col_electrodes]
                    
                    # Check if this is the first or last electrode in second column
                    if grid_pos[1] == min(second_col_y_coords) or grid_pos[1] == max(second_col_y_coords):
                        ax.set_ylabel('A [uV]')
                    else:
                        ax.set_yticklabels([])
                else:
                    ax.set_yticklabels([])
        else:
            ax.set_ylabel('A [uV]')
        
        # Show x-axis label only for bottom row or expanded view
        if expanded or (grid_pos and grid_pos[1] == max(coord[1] for coord in 
                    [self.electrode_config.header_to_coordinate(h) for h in self.data.columns])):
            ax.set_xlabel('Time [ms]')
        
        ax.tick_params(axis='both', which='major', labelsize=5)
        
        # Show legend only in expanded view
        if expanded:
            ax.legend(fontsize=8, loc='upper right')

    def mouseDoubleClickEvent(self, event):
        """Handle double-click events"""
        # Ensure zoom functionality is preserved by calling the parent method
        super().mouseDoubleClickEvent(event)
        
        # Call custom zoom behavior if needed
        self.draw()  # Refresh the plot


    def update_vertical_threshold(self, threshold):
        """Update the vertical threshold value and refresh the plot"""
        self.vertical_threshold = threshold
        if self.time is not None and self.data is not None:
            self.plot(self.time, self.data, self.electrode_config)
    

    
    



    
    
    