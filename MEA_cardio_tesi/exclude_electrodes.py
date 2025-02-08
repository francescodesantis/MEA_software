# Code to exclude electrodes. Two option are possible:
# Manual exclusion: directly check off the electrodes to exclude from visualization
# Automatic exclusion: exclude electrodes based on frequency (count Rpeaks)

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QDialogButtonBox, QListWidget, QListWidgetItem, 
                             QMessageBox, QPushButton, QInputDialog, QWhatsThis)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QIcon

# Class to exclude electrodes manually
class ManuallyExcludeElectrodes(QDialog):
    electrodes_remaining = pyqtSignal(list) # Create a list of not excluded electrodes

    def __init__(self, sorted_electrode_headers, selected_headers, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manually Exclude Electrodes")
        self.setModal(True)

        QWhatsThis.leaveWhatsThisMode() # Question Mark icon
        self.setWhatsThis("Deselect the electrodes you want to exclude from visualization and/or analysis.") 

        self.sorted_electrode_headers = sorted_electrode_headers # List of all electrode headers
        self.selected_headers = selected_headers # List of selected headers
        
        self.initUI()
        
    def initUI(self):
        self.layout = QVBoxLayout()
        
        self.electrode_list_widget = QListWidget(self) # List of all electrodes
        self.electrode_list_widget.setSelectionMode(QListWidget.MultiSelection) # Allow multiple selection
        
        for header in self.sorted_electrode_headers:
            item = QListWidgetItem(header)
            item.setCheckState(Qt.Checked if header in self.selected_headers else Qt.Unchecked) # Check if electrode is selected
            self.electrode_list_widget.addItem(item)
        
        self.layout.addWidget(self.electrode_list_widget)

        self.exclude_all_button = QPushButton("Exclude All") # Button to exclude all electrodes
        self.exclude_all_button.clicked.connect(lambda: self.exclude_all())
        self.exclude_all_button.setWhatsThis("Click 'Exclude All' to deselect all electrodes.")
        self.layout.addWidget(self.exclude_all_button)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(lambda: self.apply_manual_exclusion()) # When OK is clicked, apply exclusion
        button_box.rejected.connect(self.reject) # Close window when Cancel is clicked
        self.layout.addWidget(button_box)
        self.setLayout(self.layout)

    def exclude_all(self):
        for i in range(self.electrode_list_widget.count()):
            item = self.electrode_list_widget.item(i)
            item.setCheckState(Qt.Unchecked) # Uncheck all electrodes

    def apply_manual_exclusion(self):
        # Update selected headers based on user selection (so exclusion is allowed by keeping only headers still checked in the list)
        self.selected_headers = [item.text() for item in self.electrode_list_widget.findItems("*", Qt.MatchWrap | Qt.MatchWildcard) if item.checkState() == Qt.Checked] 
        update_selected_headers(self.electrodes_remaining, self.selected_headers)
        self.accept() 


# Class to exclude electrodes based on bpm value
class AutoExcludeElectrodes(QMessageBox, QObject):
    electrodes_remaining = pyqtSignal(list) # Create list of not excluded electrodes

    def __init__(self, avg_rates, sorted_electrode_headers, selected_headers, parent=None):
        super().__init__(parent)
        self.selected_headers = selected_headers
        self.sorted_electrode_headers = sorted_electrode_headers

        quit_msg = "Are you sure you want to apply automatic exclusion of electrodes?"
        msg_box = QMessageBox() # Message box to ask user if they want to apply automatic exclusion
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setText(quit_msg)
        msg_box.setWindowTitle('Message')
        self.setWindowIcon(QIcon.fromTheme("dialog-question"))
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        reply = msg_box.exec_()
        
        if reply == self.Yes: # If user clicks Yes, insert bpm value
            self.insert_bpm(avg_rates) 
        else:
            self.close()

    def insert_bpm(self, avg_rates):
        rate_value, ok = QInputDialog.getDouble(self, 'Input Dialog', 'Electrodes with higher rate than the following are kept:') # Ask user to insert bpm value
        QWhatsThis.leaveWhatsThisMode()
        self.setWhatsThis("Insert a value in BPM: electrodes with a higher bpm will be kept, while the electrodes with a lower one will be excluded.")
        
        if ok:   
            self.compute_auto_exclusion(avg_rates, rate_value) # Compute automatic exclusion
        else:
            self.close()
    

    def compute_auto_exclusion(self, avg_rates, rate_value):
            if isinstance(avg_rates, dict): # Check if avg_rates is an existing dictionary
                auto_selected_headers = [header for header, rate in avg_rates.items() if rate >= rate_value] # Keep electrodes with bpm higher than the value inserted by the user
        
                # Combine automatic exclusion with manual exclusion
                combined_selected_headers = list(set(self.selected_headers) & set(auto_selected_headers))
                
                self.selected_headers = combined_selected_headers # Update selected headers
                update_selected_headers(self.electrodes_remaining, self.selected_headers)
                self.accept() 
            else:
                self.close()

# Function to update selected headers from both methods
def update_selected_headers(electrodes_remaining, selected_headers):
    electrodes_remaining.emit(selected_headers)
    
       

    

