import numpy as np
import re
from PyQt5.QtCore import QObject


# Class containing electrode names and corresponding coordinates in x,y form, 
# units of micrometers (microns, um)
class ElectrodeConfig(QObject):
    # Electrode names and coordinates, using the system defined by CSD where 
    # origin (0,0) is at upper left corner of MEA.  Configured for 200x30um 
    # inter-electrode spacing and electrode diameter, respectively.

    def __init__(self, raw_data):
        super().__init__()
        self.raw_data = raw_data
        #self.electrodes = electrodes
        
        # CONSIDER OUR CASE ONLY: 60 electrodes
        self.mea_60_coordinates = {
            '47': [690, 1380], '48': [690, 1610], '46': [690, 1150],
            '45': [690, 920], '38': [460, 1610], '37': [460, 1380],
            '28': [230, 1610], '36': [460, 1150], '27': [230, 1380],
            '17': [0, 1380], '26': [230, 1150], '16': [0, 1150],
            '35': [460, 920], '25': [230, 920], '15': [0, 920],
            '14': [0, 690], '24': [230, 690], '34': [460, 690],
            '13': [0, 460], '23': [230, 460], '12': [0, 230],
            '22': [230, 230], '33': [460, 460], '21': [230, 0],
            '32': [460, 230], '31': [460, 0], '44': [690, 690],
            '43': [690, 460], '41': [690, 0], '42': [690, 230],
            '52': [920, 230], '51': [920, 0], '53': [920, 460],
            '54': [920, 690], '61': [1150, 0], '62': [1150, 230],
            '71': [1380, 0], '63': [1150, 460], '72': [1380, 230],
            '82': [1610, 230], '73': [1380, 460], '83': [1610, 460],
            '64': [1150, 690], '74': [1380, 690], '84': [1610, 690],
            '85': [1610, 920], '75': [1380, 920], '65': [1150, 920],
            '86': [1610, 1150], '76': [1380, 1150], '87': [1610, 1380],
            '77': [1380, 1380], '66': [1150, 1150], '78': [1380, 1610],
            '67': [1150, 1380], '68': [1150, 1610], '55': [920, 920],
            '56': [920, 1150], '58': [920, 1610], '57': [920, 1380]
            }
        self.electrode_toggle(raw_data)


    def electrode_toggle(self, raw_data):
        self.electrode_names = list(self.mea_60_coordinates.keys())
        self.electrode_coords_x = np.array([i[0] for i in self.mea_60_coordinates.values()])
        self.electrode_coords_y = np.array([i[1] for i in self.mea_60_coordinates.values()])

    def extract_headers(self, header_line):
        headers = re.split(r'\s+', header_line.strip())
        cleaned_headers = []
        for i in range(1, len(headers), 2):
          if headers[i] == 'El':
               cleaned_headers.append(headers[i] + headers[i+1])
        else:
               cleaned_headers.append(headers[i])
        return ['t'] + cleaned_headers
    
    def extract_electrode_number(self, header):
         match = re.match(r'\[µV\]\.(\d+)', header)  # Adjust regex pattern to match your header format
         if match:
            return [int(match.group(1)), int(match.group(2))] if match.group(2) else int(match.group(1))
        #return None

    # Extract electrode number from header
    def header_to_coordinate(self, header):
        match = re.match(r'El_(\d+)', header)
        if match:
            electrode_number = match.group(1)
            return self.mea_60_coordinates.get(electrode_number)
        return None
