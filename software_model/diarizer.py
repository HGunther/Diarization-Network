from software_model.neural_network import NeuralNetwork
from software_model.network_data_preprocessor import NetworkDataPreprocessor
from software_model.network_data_postprocessor import NetworkDataPostprocessor


class Diarizer:

    def __init__(self, neural_network_location):
        self._neural_network_location = neural_network_location

    def _pre_processing(self, wav_file_name):
        return NetworkDataPreprocessor([wav_file_name]).get_all_chunks_in_file()

    def _post_processing(self, network_output, wav_file_name):
        postprocessor = NetworkDataPostprocessor(network_output, wav_file_name)
        postprocessor.write_to_csv()

    def annotate_wav_file(self, wav_file_name):
        network_input = self._pre_processing(wav_file_name)
        net = NeuralNetwork()
        network_output = net.evaluate_chunks(network_input, self._neural_network_location)
        self._post_processing(network_output, wav_file_name)

    def train_network(self):
        None
