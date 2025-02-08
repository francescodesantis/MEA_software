import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PyQt5.QtWidgets import QPushButton, QLabel
#from find_peaks_9_simple_new_all import find_peaks


    

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=12, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi) # Create figure. dpi = resolution
        super().__init__(self.fig)
        self.setParent(parent)

        # Initialization of variables
        self.axes = [] 
        self.expanded_ax = None 
        self.time = None
        self.data = None
        self.peaks = {}
        self.vertical_amplitudes_info={}
        self.pre_potentials = {} 
        self.sampling_rate=10000
        # self.q_peaks = {}
        # self.s_peaks = {}
        self.t_peaks = {}
        self.points = []
        self.plotted_points = []
       



        self.toolbar = NavigationToolbar(self, parent)
        self.toolbar.hide()  # Initially hide toolbar

        self.fig.canvas.mpl_connect('button_press_event', self.on_click) # Connect click event to method
       
       
    # Create grid of subplots
    def plot(self, time, data, electrode_config, vertical_amplitudes_info=None):
        self.time = time
        self.data = data
        self.electrode_config = electrode_config

        #_,_,_,_,vertical_amplitudes_info,_=find_peaks(self.data)

        #self.vertical_amplitudes_info = vertical_amplitudes_info  # Salva i dati dei picchi


        # Clear figure and reset axes before new plots are drawn
        self.fig.clear()
        self.axes.clear()

        # Map headers to grid coordinated based on ElectrodeConfig
        grid_layout = {header: self.electrode_config.header_to_coordinate(header) for header in data.columns}
        valid_grid_layout = {header: coord for header, coord in grid_layout.items() if coord is not None}

        if not valid_grid_layout:
            print("No valid coordinates found for plotting.")
            return

        # Define grid layout size based on coordinates
        x_coords = [coord[0] for coord in valid_grid_layout.values()]
        y_coords = [coord[1] for coord in valid_grid_layout.values()]

        if not x_coords or not y_coords:
            print("No valid coordinates found for plotting.")
            return

        # Compute number of rows and columns for grid
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        rows = max(1, (max_y - min_y) // 210 + 1)
        cols = max(1, (max_x - min_x) // 210 + 1)

        # Create GridSpec to layout subplots (GridSpec = literal grid layout to locate grid cells within figure)
        gs = GridSpec(rows, cols, figure=self.fig)

        position_map = {}  # Map grid coordinates to header names (useful during exclusion phase to slide electrodes)
        # Create subplots for each header in data file
        for header, coords in sorted(valid_grid_layout.items(), key=lambda item: item[1]):
            row = (coords[1] - min_y) // 210
            col = (coords[0] - min_x) // 210

            if (row, col) in position_map:
                print(f"Warning: Overlapping position for {header} at ({row}, {col}).")
                continue

            position_map[(row, col)] = header

            if row < gs.nrows and col < gs.ncols:  # Check if coordinates are within grid. Necessary to deal with exclusion of electrodes
                ax = self.fig.add_subplot(gs[row, col])
                self.axes.append(ax)
                ax.set_ylim([-200, 300])  # Set y-axis limit to have comparable subplots
                ax.plot(self.time[:10000], self.data[header][:10000])

                ax.text(0.5, 0.9, f'{header}', horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes, fontsize=10)

                ax.header = header  # Store header in a custom attribute for later use

                # Show x-axis label only for bottom row subplots
                if row == rows - 1:
                    ax.set_xlabel('Time [ms]')
                else:
                    ax.set_xticklabels([])

                # Show y-axis label only for first column subplots
                if col == 0:
                    ax.set_ylabel('A [uV]')
                else:
                    ax.set_yticklabels([])

                ax.tick_params(axis='both', which='major', labelsize=8)

        self.fig.subplots_adjust(wspace=0, hspace=0)  # Remove space between subplots
        self.draw_idle()



    # Allow to double-click on a subplot to expand it or collapse it back 
    def on_click(self, event):
        if event.dblclick:
            if self.expanded_ax is not None:
                # If already expanded, collapse all subplots back to original grid layout
                self.collapse_all_subplots()
            else:
                # Expand selected subplot
                for ax in self.axes:
                    if ax == event.inaxes:
                        self.expand_subplot(ax)
                        break
        self.draw_idle()
    
    
    """def expand_subplot(self, ax): #versione che plotta picchi con ampiezze sbagliate
        # Hide all axes except expanded one
        for other_ax in self.axes:
            if other_ax != ax:
                other_ax.set_visible(False)  # Hide other subplots

        # Set expanded subplot to fill figure
        gs = GridSpec(1, 1, figure=self.fig)
        ax.set_position(gs[0, 0].get_position(self.fig))
        ax.set_subplotspec(gs[0, 0])

        header = getattr(ax, 'header', None)

        if header and header in self.data.columns:
            ax.cla()  # Clear only current axes
            initial_display_limit = 10000  # Start with small amount of data to display
            display_data = self.data[header][:initial_display_limit]
            ax.plot(self.time[:len(display_data)], display_data)

            ax.text(0.5, 0.9, f'{header}', horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes, fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=10)

            # Make sure axis labels are visible in expanded view
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            ax.set_xlabel('Time [ms]')
            ax.set_ylabel('A [uV]')

            self.toolbar.show()  # Show toolbar for navigation on expanded subplot
            self.expanded_ax = ax

            # Incrementally add more data to the plot if it's too long
            if len(self.data[header]) > initial_display_limit:
                self.incremental_plot(ax, header, initial_display_limit)
            else:
                print("Error: Could not find header or data for the expanded subplot.")

            # Plot detected peaks as red dots
            if header in self.peaks:
                peak_times = self.peaks[header]  # Get the peak times for the current channel
                peak_indices = np.round(np.array(peak_times) * self.sampling_rate).astype(int)  # Convert times to indices

                # Filter out indices that are out of bounds
                valid_peak_indices = peak_indices[(peak_indices >= 0) & (peak_indices < len(self.data[header]))]

                if valid_peak_indices.size > 0:
                    ax.plot(self.time[valid_peak_indices], self.data[header].iloc[valid_peak_indices], "ro", label="Detected Peaks")

            # Show legend if needed
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())"""



    
    """def expand_subplot(self, ax):
        # Hide all axes except expanded one
        for other_ax in self.axes:
            if other_ax != ax:
                other_ax.set_visible(False) # Hide other subplots

        # Set expanded subplot to fill figure
        gs = GridSpec(1, 1, figure=self.fig)
        ax.set_position(gs[0, 0].get_position(self.fig))
        ax.set_subplotspec(gs[0, 0])

        header = getattr(ax, 'header', None)

        if header and header in self.data.columns:
            ax.cla()  # Clear only current axes
            initial_display_limit = 10000  # Start with small amount of data to display
            display_data = self.data[header][:initial_display_limit]
            ax.plot(self.time[:len(display_data)], display_data)

            ax.text(0.5, 0.9, f'{header}', horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes, fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=10)

            # Make sure axis labels are visible in expanded view
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            ax.set_xlabel('Time [ms]')
            ax.set_ylabel('A [uV]')

            self.toolbar.show()  # Show toolbar for navigation on expanded subplot
            self.expanded_ax = ax

            # Incrementally add more data to the plot if it's too long
            if len(self.data[header]) > initial_display_limit:
                self.incremental_plot(ax, header, initial_display_limit)
            else:
                print("Error: Could not find header or data for the expanded subplot.")

            header = ax.header 

            # Plot detected peaks on averaged data
            if isinstance(self.peaks, dict) and 'R_peak' in self.peaks: # Distinguish between dictionary peaks of normal data and nested dictionary peaks if averaged data
                # Averaged signals
                # if header in self.peaks['pre_potential']:
                #     ppeaks_indices = np.round(self.peaks['pre_potential'][header]).astype(int)
                #     valid_pindices = ppeaks_indices[ppeaks_indices < len(self.data[header])]
                #     if valid_pindices.size > 0:
                #         ax.plot(self.time[valid_pindices], self.data[header].iloc[valid_pindices], "x", label="Detected p Peaks")

                if header in self.peaks['R_peak']:
                    rpeaks_indices = np.round(self.peaks['R_peak'][header]).astype(int) # Be sure to have integers for indexes
                    rpeaks_indices = rpeaks_indices[~np.isnan(rpeaks_indices)] # Remove NaN values
                    valid_rindices = rpeaks_indices[(rpeaks_indices >= 0) & (rpeaks_indices < len(self.data[header]))] # Remove invalid indices and stay in bounds
                    if valid_rindices.size > 0:
                        ax.plot(self.time[valid_rindices], self.data[header].iloc[valid_rindices], "x", label="Detected R Peaks")

                if header in self.peaks['R_onset']:
                    qpeaks_indices = np.round(self.peaks['R_onset'][header]).astype(int)
                    if np.any(np.isnan(qpeaks_indices)) or np.any(np.isinf(qpeaks_indices)):
                        print(f"Invalid Q peaks detected for {header}: {qpeaks_indices}")
                    qpeaks_indices = qpeaks_indices[~np.isnan(qpeaks_indices)]
                    qpeaks_indices = qpeaks_indices[np.isfinite(qpeaks_indices)]
                    valid_qindices = qpeaks_indices[(qpeaks_indices >= 0) & (qpeaks_indices < len(self.data[header]))]
                    if valid_qindices.size > 0:
                        ax.plot(self.time[valid_qindices], self.data[header].iloc[valid_qindices], "o", label="Detected R onset Peaks")

                if header in self.peaks['s_peak']:
                    speaks_indices = np.round(self.peaks['s_peak'][header]).astype(int)
                    speaks_indices = speaks_indices[~np.isnan(speaks_indices)]
                    valid_sindices = speaks_indices[(speaks_indices >= 0) & (speaks_indices < len(self.data[header]))]
                    if valid_sindices.size > 0:
                        ax.plot(self.time[valid_sindices], self.data[header].iloc[valid_sindices], "v", label="Detected S Peaks")

                if header in self.peaks['t_peak']:
                    tpeaks_indices = np.round(self.peaks['t_peak'][header]).astype(int)
                    tpeaks_indices = tpeaks_indices[~np.isnan(tpeaks_indices)]
                    valid_tindices = tpeaks_indices[(tpeaks_indices >= 0) & (tpeaks_indices < len(self.data[header]))]
                    if valid_tindices.size > 0:
                        ax.plot(self.time[valid_tindices], self.data[header].iloc[valid_tindices], "s", label="Detected T Peaks")

            else:
                # Plot detected peaks on normal data
                if header in self.peaks:
                    rpeaks_indices = np.round(self.peaks[header]).astype(int)
                    valid_rindices = rpeaks_indices[rpeaks_indices < len(self.data[header])]
                    if valid_rindices.size > 0:
                        ax.plot(self.time[valid_rindices], self.data[header].iloc[valid_rindices], "x", label="Detected R Peaks")
                if header in self.pre_potentials:
                    ppeaks_indices = np.round(self.pre_potentials[header]).astype(int)
                    valid_pindices = ppeaks_indices[ppeaks_indices < len(self.data[header])]
                    if valid_pindices.size > 0:
                        ax.plot(self.time[valid_pindices], self.data[header].iloc[valid_pindices], "o", label="Detected P Peaks")"""
                # if header in self.t_peaks:
                #     #tpeaks_indices = np.round(self.t_peaks[header]).astype(int)
                #     if isinstance(self.t_peaks[header], list):
                #         tpeaks_filtered = [x for x in self.t_peaks[header] if not np.isnan(x)]
                #         tpeaks_indices = np.round(tpeaks_filtered).astype(int)
                #         valid_tindices = tpeaks_indices[tpeaks_indices < len(self.data[header])]
                #         if valid_tindices.size > 0:
                #             ax.plot(self.time[valid_tindices], self.data[header].iloc[valid_tindices], "x", label="Detected T Peaks")
                #     else:
                #         print(f"Invalid type for self.t_peaks[header]: {type(self.t_peaks[header])}")
            

            # Compute distance between two points when Compute Distance button is clicked
    def expand_subplot(self, ax):
        # Hide all axes except expanded one
        for other_ax in self.axes:
            if other_ax != ax:
                other_ax.set_visible(False)  # Hide other subplots

        # Set expanded subplot to fill figure
        gs = GridSpec(1, 1, figure=self.fig)
        ax.set_position(gs[0, 0].get_position(self.fig))
        ax.set_subplotspec(gs[0, 0])

        header = getattr(ax, 'header', None)

        if header and header in self.data.columns:
            ax.cla()  # Clear only current axes
            initial_display_limit = 10000  # Start with small amount of data to display
            display_data = self.data[header][:initial_display_limit]
            ax.plot(self.time[:len(display_data)], display_data)

            ax.text(0.5, 0.9, f'{header}', horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes, fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=10)

            # Make sure axis labels are visible in expanded view
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            ax.set_xlabel('Time [ms]')
            ax.set_ylabel('A [uV]')

            self.toolbar.show()  # Show toolbar for navigation on expanded subplot
            self.expanded_ax = ax

            # Incrementally add more data to the plot if it's too long
            if len(self.data[header]) > initial_display_limit:
                self.incremental_plot(ax, header, initial_display_limit)
            else:
                print("Error: Could not find header or data for the expanded subplot.")

            # Plot detected peaks using vertical_amplitudes to get time and amplitude
            if header in self.vertical_amplitudes_info:
                # Get the time and amplitude of the peaks for the current channel
                peak_data = self.vertical_amplitudes_info[header]  # This is a list of (time, amplitude) tuples

                # Extract times and amplitudes
                peak_times = [peak_time for peak_time, _ in peak_data]
                peak_amplitudes = [amplitude for _, amplitude in peak_data]

                # Convert peak times to indices
                peak_indices = np.round(np.array(peak_times) * self.sampling_rate).astype(int)

                # Filter out indices that are out of bounds
                valid_peak_indices = peak_indices[(peak_indices >= 0) & (peak_indices < len(self.data[header]))]

                if valid_peak_indices.size > 0:
                    # Plot the peaks as red dots at the corresponding time and amplitude
                    ax.plot(self.time[valid_peak_indices], 
                            np.array(self.data[header].iloc[valid_peak_indices]),
                            'ro', label="Detected Peaks")

                    # Optionally, annotate the peaks with their amplitude values
                    for i, peak_idx in enumerate(valid_peak_indices):
                        ax.text(self.time[peak_idx], self.data[header].iloc[peak_idx],
                                f'{peak_amplitudes[i]:.2f}', color='red', fontsize=8,
                                ha='center', va='bottom')  # Label each peak with its amplitude

            # Show legend if needed
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
    
   

# Incrementally add more data to handle long signals
    def incremental_plot(self, ax, header, start_idx):
        total_length = len(self.data[header])
        chunk_size = 1000  # Number of points to add per iteration

        for i in range(start_idx, total_length, chunk_size):
            end_idx = min(i + chunk_size, total_length)  # End index for current chunk
            ax.plot(self.time[i:end_idx], self.data[header][i:end_idx], color='blue')  # Plot chunk of data
            self.draw_idle()
            if end_idx >= total_length:  # Stop when all data has been plotted
                break


    # Restore all subplots to their original layout
    def collapse_all_subplots(self):
        self.fig.clear()
        self.expanded_ax = None
        self.plot(self.time, self.data, self.electrode_config)
        self.toolbar.hide()  # Hide toolbar when subplots are collapsed

    #def update_peaks(self, peaks):
        """
        Aggiorna i picchi per tutti i canali e ridisegna il grafico.

        Parameters:
        -----------
        peaks : dict
            Dizionario con i picchi calcolati per ciascun canale.
        """
        #peaks_in_ms = {channel: [time * 1000 for time in times] for channel, times in peaks.items()}
        
        # Stampa i picchi calcolati in millisecondi
        #print("Peaks in milliseconds:", peaks_in_ms)
        
       # self.peaks = peaks  # Aggiorna il dizionario dei picchi
        
        #self.plot(self.time, self.data, self.electrode_config, self.vertical_amplitudes_info)
        
        #self.plot(self.time, self.data, self.electrode_config)  # Ridisegna il grafico con i nuovi picchi

   

    
    