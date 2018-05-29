import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget, QPushButton
from PyQt5.QtCore import QSize   
from EvalWizardGUI.EvalWizardStep1 import EvalWizardStep1Window 

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setMinimumSize(QSize(1000, 400))    
        self.setWindowTitle("Diarization") 

        centralWidget = QWidget(self)          
        self.setCentralWidget(centralWidget)   

        gridLayout = QGridLayout(self)     
        centralWidget.setLayout(gridLayout)  

        title = QLabel("Welcome to Your Diarization Software", self) 
        title.setAlignment(QtCore.Qt.AlignCenter) 
        
        diarize_btn = QPushButton("Diarize previously unseen .wav file(s)", self)
        diarize_btn.clicked.connect(self.wizard_initalizer_evaluate_wav_files)
        
        train_btn = QPushButton("     Train the software based on existing diarizied files     ", self)
        #train_btn.setAlignment(QtCore.Qt.AlignCenter)
        train_btn.clicked.connect(self.wizard_initializer_train_net)
        gridLayout.setColumnStretch(0,1)
        gridLayout.setRowStretch(0, 1)
        gridLayout.addWidget(title, 1, 1)
        gridLayout.setRowStretch(1, 1)
        gridLayout.addWidget(diarize_btn, 3, 1)
        gridLayout.addWidget(train_btn, 4, 1)
        gridLayout.setRowStretch(5, 1)
        gridLayout.setColumnStretch(2,1)
        
    def wizard_initalizer_evaluate_wav_files(self):
        self._diarize_wizard = EvalWizardStep1Window()
        self._diarize_wizard.show()

    def wizard_initializer_train_net(self):
        print("Wiz2")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit( app.exec_() )