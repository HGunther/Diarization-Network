import sys
from PyQt5 import QtWidgets
from software_model.neural_network import NeuralNetwork
from software_model.diarizer import Diarizer
from software_view.view import MainWindow


"""
Below are examples highlighting how the software can be used
"""


def show_gui():
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())


def train_net():
    files_to_train_on = ['HS_D{0:0=2d}'.format(i) for i in range(1, 38)]
    del files_to_train_on[files_to_train_on.index('HS_D11')]
    del files_to_train_on[files_to_train_on.index('HS_D22')]

    net = NeuralNetwork()
    place_to_save_model = "Model/may30.ckpt"
    # reviously_saved_network_model = "Model/ultimate_model_saved_weights.ckpt"
    previously_saved_network_model = None

    # Note that you can adjust the default parameters, such as epochs or batch size here too.
    net.train_network(files_to_train_on, place_to_save_model, in_model_location=previously_saved_network_model)


def evaluate_on_a_particular_file():
    previously_saved_network_model = "Model/ultimate_model_saved_weights.ckpt"
    dire = Diarizer(previously_saved_network_model)
    dire.annotate_wav_file("HS_D01")


if __name__ == '__main__':
    print("Go to line 38 in the file '__main__.py' and uncomment a subsequent line")
