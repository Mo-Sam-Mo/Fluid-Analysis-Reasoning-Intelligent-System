from Reasoning import Reasoning_Model
from Model import Classification_Model
from Functions import enrich_and_reorder_features, sample_conversion, organize_report
import pandas as pd
import numpy as np
from Voice import TextToSpeach

FEATURE_NAMES = ['Cu', 'Fe', 'Cr', 'Al', 'Si', 'Pb', 'Sn', 'Ni', 'Na', 'B', 'P', 'Zn',
                'Mo', 'Ca', 'Mg', 'TBN', 'V100', 'V40', 'OXI', 'TAN', 'water_flag', 'antifreeze_flag']


class FARIS():
    def __init__(self):
        # self.OCR = 
        self.Model = Classification_Model()
        self.Reasoning = Reasoning_Model()
        self.Voice = TextToSpeach()

    def predict(self, sample):
        X = pd.DataFrame(np.array(sample).reshape(1, -1), columns=FEATURE_NAMES)
        X = enrich_and_reorder_features(X)
        txt_sample = sample_conversion(X)
        

        cls = self.Model.classification(X)
        reasoning = self.Reasoning.generate_response(cls, txt_sample)
        reasoning, voice = organize_report(reasoning)
        audio_filename = self.Voice.speach(voice)

        return cls, reasoning, audio_filename
        