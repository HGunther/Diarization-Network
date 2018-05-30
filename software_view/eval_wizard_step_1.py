from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget, QPushButton, QFileDialog
from PyQt5.QtCore import QSize
from pathlib import Path
from PyQt5.Qt import QListWidget, QLineEdit, QMessageBox
from software_view.eval_wizard_step_2 import EvalWizardStep2Window

class EvalWizardStep1Window(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setMinimumSize(QSize(900, 500))    
        self.setWindowTitle("Diarization - Diarize Files (Step 1 of 2)") 

        centralWidget = QWidget(self)          
        self.setCentralWidget(centralWidget)   

        gridLayout = QGridLayout(self)     
        centralWidget.setLayout(gridLayout)  

        title = QLabel("Select the files to be diarized:", self) 
        title.setAlignment(QtCore.Qt.AlignCenter)
        
        open_files_btn = QPushButton("Select .wav File(s)", self)
        open_files_btn.clicked.connect(self._select_wav_files) 
        
        self._filenames_list = QListWidget(self)
        self._filenames_list.setAlternatingRowColors(True)
        
        self._output_directory = QLineEdit(self)
        self._output_directory.setReadOnly(True)
        self._output_directory.setDisabled(True)
        
        open_dir_btn = QPushButton("Select Output Directory", self)
        open_dir_btn.clicked.connect(self._select_output_directory) 
        
        self._network_location = QLineEdit(self)
        self._network_location.setReadOnly(True)
        self._network_location.setDisabled(True)
        
        open_network_btn = QPushButton("Select Evaluation Neural Network", self)
        open_network_btn.clicked.connect(self._select_network)
        
        
        
        next_btn = QPushButton("Next", self)
        next_btn.clicked.connect(self._next_actions)
        
        
        gridLayout.setRowStretch(0, 1)
        gridLayout.addWidget(title, 1, 1)
        gridLayout.setRowStretch(1, 1)
        gridLayout.addWidget(self._filenames_list, 2, 0, 1, 3)
        gridLayout.addWidget(open_files_btn, 2, 3, 1, 1)
        gridLayout.addWidget(self._output_directory, 3, 0, 1, 3)
        gridLayout.addWidget(open_dir_btn, 3, 3, 1, 1)
        gridLayout.addWidget(self._network_location, 4, 0, 1, 3)
        gridLayout.addWidget(open_network_btn, 4, 3, 1, 1)
        gridLayout.setRowStretch(49, 1)
        gridLayout.addWidget(next_btn, 50, 3)
        
    def _next_actions(self):
        try:
            # Clean Input a Bit
            # # Change variables to string type
            wav_files =  [str(self._filenames_list.item(i).text()) for i in range(self._filenames_list.count())]
            output_directory = self._output_directory.text()
            network_location = self._network_location.text()
            # Substring network location to give true value needed
            network_location = network_location[0 : network_location.rindex(".")]
            self._step2 = EvalWizardStep2Window(wav_files, output_directory, network_location)
            self._step2.show()
            self.close()
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("There was an error. Double check your input files and try again.")
            msg.setWindowTitle("Error")
            msg.exec_()
        

    def _select_wav_files(self):
        try:
            data = QFileDialog.getOpenFileNames(None, '', str(Path.home()), '*.wav', '')
            self._filenames_list.addItems(data[0])
        except:
            None
    
    def _select_output_directory(self):
        try:
            data = QFileDialog.getExistingDirectory(None, '', str(Path.home()))
            self._output_directory.setText(data)
        except:
            None 

    def _select_network(self):
        try:
            data = QFileDialog.getOpenFileName(None, '', str(Path.home()), '*.ckpt.meta', '')
            self._network_location.setText(data[0])
        except:
            None  
    