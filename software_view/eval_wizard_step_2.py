from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget
from PyQt5.QtCore import QSize
from PyQt5.Qt import QProgressBar
import time
from threading import Thread


class EvalWizardStep2Window(QMainWindow):

    def __init__(self, wav_files, output_directory, network_location):
        QMainWindow.__init__(self)

        self.wav_files = wav_files
        self.output_directory = output_directory
        self.network_location = network_location

        self.setMinimumSize(QSize(900, 500))
        self.setWindowTitle("Diarization - Diarize Files (Step 2 of 2)")

        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)

        gridLayout = QGridLayout(self)
        centralWidget.setLayout(gridLayout)

        title = QLabel("Please Wait While The .wav Files Are Annotated", self)
        title.setAlignment(QtCore.Qt.AlignCenter)

        self._all_files_progress = QProgressBar(self)
        self._all_files_progress.setMinimum(1)
        self._all_files_progress.setMaximum(len(wav_files))

        self._file_name_lbl = QLabel("File: ", self)
        self._file_name_lbl.setAlignment(QtCore.Qt.AlignCenter)

        self._file_progress_lbl = QLabel("0%", self)
        self._file_progress_lbl.setAlignment(QtCore.Qt.AlignCenter)

        gridLayout.setRowStretch(0, 0)
        gridLayout.addWidget(title, 1, 0)
        gridLayout.setRowStretch(2, 0)
        gridLayout.addWidget(self._all_files_progress, 3, 0)
        gridLayout.addWidget(self._file_name_lbl, 4, 0)
        gridLayout.addWidget(self._file_progress_lbl, 5, 0)
        gridLayout.setRowStretch(49, 0)
        gridLayout.setRowStretch(50, 0)

        thread = Thread(target=self.exe_controler, args=())
        thread.start()

    def exe_controler(self):
        # net = network.open(self.network_location)
        i = 0
        for file_name in self.wav_files:
            self._file_name_lbl.setText("File: " + file_name)
            # net.annotate_wav_file(file_name, self.output_directory)

            time.sleep(0.1)  # Delete This Line Shortly
            self._all_files_progress.setValue(i)
            self._file_progress_lbl.setText(str(i * 100 / len(self.wav_files)) + "%")
            i += 1
        self._all_files_progress.setValue(len(self.wav_files))
        self._file_progress_lbl.setText("100%; Your files have now been annotated. Please close this window.")
