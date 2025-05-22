from Reasoning import Reasoning_Model
from Model import Classification_Model
from Functions import enrich_and_reorder_features, sample_conversion, organize_report
import pandas as pd
import numpy as np
from gtts import gTTS
import uuid

FEATURE_NAMES = ['Cu', 'Fe', 'Cr', 'Al', 'Si', 'Pb', 'Sn', 'Ni', 'Na', 'B', 'P', 'Zn',
                'Mo', 'Ca', 'Mg', 'TBN', 'V100', 'V40', 'OXI', 'TAN', 'water_flag', 'antifreeze_flag']


class FARIS():
    def __init__(self):
        # self.OCR = 
        self.Model = Classification_Model()
        self.Reasoning = Reasoning_Model()

    def predict(self, sample):
        X = pd.DataFrame(np.array(sample).reshape(1, -1), columns=FEATURE_NAMES)
        X = enrich_and_reorder_features(X)
        txt_sample = sample_conversion(X)
        

        cls = self.Model.classification(X)
        reasoning = self.Reasoning.generate_response(cls, txt_sample)
        reasoning, voice = organize_report(reasoning)

        #prepare the reasoning shape
        tts = gTTS(voice)
        audio_filename = f"audio/output_{uuid.uuid4()}.mp3"
        tts.save(audio_filename)

        #make the voice
        return cls, reasoning, audio_filename
        