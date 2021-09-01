
from pydub import AudioSegment
from pydub import AudioSegment
import os

import librosa
from scipy.io import wavfile
import scipy.signal as sps
import librosa
import tensorflow as tf
import numpy as np
from pydub import AudioSegment, effects  

SAVED_MODEL_PATH = "model01.h5"

SAMPLES_TO_CONSIDER = 22050

class _Keyword_Spotting_Service:


    model = None
    _mapping = [
           "\u0631",
        "\u0628",
        "\u0643",
        "\u0645",
        "\u0639",
        "\u0634",
        "\u0623",
        "\u0632",
        "\u0647",
        "\u0646",
        "\u0636",
        "\u0635",
        "\u062a",
        "\u0648",
        "\u062f",
        "\u062d",
        "\u064a",
        "\u0638",
        "\u0642",
        "\u063a",
        "\u062b",
        "\u0633",
        "\u0630",
        "\u062c",
        "\u0641",
        "\u0644",
        "\u0637",
        "\u062e"
    ]
    

    _instance = None


    def predict(self, file_path):


        # extract MFCC
        MFCCs = self.preprocess(file_path)

        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
        # print(MFCCs.shape)

        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword


    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):


        # load audio file

        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
        MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                     hop_length=hop_length)
        # MFCCs = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=2048, hop_length=1024)
        # MFCCs = librosa.power_to_db(MFCCs, ref=np.max)
        
        # print(MFCCs.shape)
        nx, ny= MFCCs.shape
        # print(nx,ny)
        # MFCCs= np.reshape(MFCCs, ((-1, 22, 128)))


        return MFCCs.T


def Keyword_Spotting_Service():


    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance




if __name__ == "__main__":
    file=("/test/أ11 .wav")

    # sound = AudioSegment.from_file(file)
    # print("----------Before Conversion--------")
    # print("Frame Rate", sound.frame_rate)
    # print("Channel", sound.channels)
    # print("Sample Width",sound.sample_width)

    # # Change Frame Rate
    
    # sound = sound.set_frame_rate(22000)
    # sound = effects.normalize(sound)  

    # # Change Channel
    # sound = sound.set_channels(1)
    # # Change Sample Width
    # sound = sound.set_sample_width(2)
    
    # sound.export(file, format ="wav")
    
    kss = Keyword_Spotting_Service()
    kss1 = Keyword_Spotting_Service()

    assert kss is kss1

    # make a prediction
    keyword = kss.predict(file)
    print(keyword)
