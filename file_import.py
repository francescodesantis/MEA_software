import pandas as pd
import re
from PyQt5.QtCore import QThread, pyqtSignal


# A thread in Python is a separate flow of execution. This means that your program can 
# have multiple threads running concurrently, allowing you to perform multiple tasks at 
# once within the same process.
#Main Thread: When you start a Python program, it runs in a single thread called the main 
# thread.
#Additional Threads: You can create additional threads to run other functions 
# concurrently. This is useful for tasks that can run independently of the main program, 
# like loading data, waiting for user input, or performing background computations.

# Load data in background while the main application remains responsive
class DataLoadingThread(QThread):
    # Signal to send data to main thread: time, electrode_data, sorted_electrode_headers
    data_loaded = pyqtSignal(pd.Series, pd.DataFrame, list) 

    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name

    # Main function that runs when the thread starts
    def run(self):
        # Read data
        data = pd.read_csv(self.file_name, sep=r'\s+', skiprows=3, header=0, 
                           lineterminator='\n', encoding='iso-8859-15', 
                           skipinitialspace=True, low_memory=False)
        
        # Extract headers from file
        with open(self.file_name, 'r', encoding='iso-8859-15') as file:
            header_line = file.readlines()[2] # Headers are read from third line

        headers = self.extract_headers(header_line)
        electrode_headers = headers[1:] # List of 'El12', 'El13', etc.

        # Extact electrode numbers from headers
        electrode_numbers = [self.extract_electrode_number(header) for header in electrode_headers] # Extract actual numbers from headers
        valid_electrode_indices = [i for i, num in enumerate(electrode_numbers) if num is not None] # If no number, return None
        sorted_electrode_indices = sorted(valid_electrode_indices, key=lambda i: electrode_numbers[i]) # Put in numerical order indices
        sorted_electrode_headers = [f'El_{electrode_numbers[i]}' for i in sorted_electrode_indices] # Create new headers with sorted indices

        # Extract column indices for time and electrode data
        column_indices = [0] + [i + 1 for i in sorted_electrode_indices]
        column_indices = [idx for idx in column_indices if idx < data.shape[1]] # Stay into bound of data shape

        try:
            sorted_data = data.iloc[:, column_indices] # Extract columns from dataframe in a sorted way
        except IndexError as e:
            print(f"IndexError: {e}")
            print(f"Data Shape: {data.shape}")
            print(f"Column Indices: {column_indices}")
            return

        # Separate time and electrode information
        time = sorted_data.iloc[:, 0]
        electrode_data = sorted_data.iloc[:, 1:]

        # Check if number of columns matches number of headers
        if electrode_data.shape[1] != len(sorted_electrode_headers):
            print(f"Column mismatch: electrode_data has {electrode_data.shape[1]} columns, but headers list has {len(sorted_electrode_headers)} elements.")
            return

        electrode_data.columns = sorted_electrode_headers # Set column names 

        # Emit signal with loaded data (time, electrode_data, sorted_electrode_headers, peaks_dict)
        self.data_loaded.emit(time, electrode_data, sorted_electrode_headers)



    def extract_headers(self, header_line):
        # Extract and clean headers from header line
        headers = re.split(r'\s+', header_line.strip())
        cleaned_headers = []
        for i in range(1, len(headers), 2): 
            # Skip first element and iterate over every second element, in pratica ogni volta legge un elemento, lo salva e rilegge la riga ma skippa quello prima per poter salvare solo l'info nuova
            if headers[i] == 'El':
                cleaned_headers.append(headers[i] + headers[i+1])
            else:
                cleaned_headers.append(headers[i])
        return ['t'] + cleaned_headers

    def extract_electrode_number(self, header):
        # Extract electrode number from header
        match = re.match(r'El(\d+)', header)
        if match:
            return int(match.group(1))
        return None

