import numpy as np
from software_model.constants import PREDICTON_FILES_LOCATION, CHUNK_SIZE_S
import csv
class NetworkDataPostprocessor:



    def __init__(self, network_output, wav_file_name):
        self._network_output = network_output
        self._wav_file_name = wav_file_name

    def write_to_csv(self):
        speaker0_prediction_array = self._network_output[:,0]
        speaker1_prediction_array = self._network_output[:,1]
        
        speaker0_prediction_array = self._moving_average(speaker0_prediction_array)
        speaker1_prediction_array = self._moving_average(speaker1_prediction_array)
        
        speaker0_prediction_array = np.round(speaker0_prediction_array)
        speaker1_prediction_array = np.round(speaker1_prediction_array)
        
        speaker0_csv_array = self._prediction_array_to_csv_data(speaker0_prediction_array, 0)
        speaker1_csv_array = self._prediction_array_to_csv_data(speaker1_prediction_array, 1)
        
        self.__write_csv_file(speaker0_csv_array, self._wav_file_name, 0)
        self.__write_csv_file(speaker1_csv_array, self._wav_file_name, 1)
        
    def _prediction_array_to_csv_data(self, predict_array, speaker_id):
        
        start_time_in_block = 0
        current_time_in_block = 0
        toggle = predict_array[0]
        
        to_return = []
        
        for value in predict_array:
            current_time_in_block += CHUNK_SIZE_S
            if value != toggle:
                toggle = value
                to_return.append([round(start_time_in_block, 6), round(current_time_in_block, 6), int(toggle), speaker_id+1])
                start_time_in_block = current_time_in_block
        # Add value for the very last block
        to_return.append([round(start_time_in_block, 6), round(current_time_in_block, 6), int(toggle), speaker_id+1])
        
        return to_return    
        
    def __write_csv_file(self, prediction_array_as_csv_data, file_name, speaker_id):
        with open(PREDICTON_FILES_LOCATION+file_name+"_Spk"+str(speaker_id+1)+".csv", 'w', newline="\n", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # Write the header
            writer.writerow(['tmi0', 'tmax', 'text', 'tier'])
            for row in prediction_array_as_csv_data:
                writer.writerow(row)
        
    def _moving_average(self, data_array, window=5):
        data_array = np.array(data_array)
        return np.convolve(data_array, np.ones((window,))/window, mode='valid')
        