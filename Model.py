from keras.models import Sequential
from keras.layers import Dense
import joblib


NUMERICAL_FEATS = ['Cu', 'Fe', 'Al', 'Si', 'Pb', 'Sn', 'Na', 'B', 'P', 'Zn', 'Mo',
                'Ca', 'Mg', 'TBN', 'V100', 'V40', 'OXI', 'TAN', 'delta_visc_40',
                'metal_sum', 'iron_to_copper_ratio']

DESCTION_TREE_PICKLE = 'Weights/DecisionTreeModel.pkl'
YEO_PICKLE = 'Weights/PowerTransformer.pkl'
DEEP_NET_WEIGHTS = 'Weights/DeepNet.keras'
ENCODER_PICKLE = 'Weights/Encoders.pkl'
SCALER_PICKLE = 'Weights/Scaler.pkl'


class Classification_Model():
    
    def __init__(self):
        self.DeepNet = Sequential([
            Dense(64, activation='relu', input_shape=(25,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(6, activation='softmax')
        ])

        self.DescionTree = joblib.load(DESCTION_TREE_PICKLE)
        self.DeepNet.load_weights(DEEP_NET_WEIGHTS)
        self.scaler = joblib.load(SCALER_PICKLE)
        self.yeo = joblib.load(YEO_PICKLE)
        self.encoder = joblib.load(ENCODER_PICKLE)


    def preproces_deep(self, sample):
        sample = self.yeo.transform(sample)
        sample = self.encoder.transform(sample)
        sample[:, NUMERICAL_FEATS] = self.scaler.transofrm(sample[:, NUMERICAL_FEATS])

        return sample


    def train_deep(self, sample):
        sample = self.preproces_deep(sample)
        self.DeepNet.fit(sample)
        self.DeepNet.save(DEEP_NET_WEIGHTS)

    
    def classification(self, sample, model='DT'):

        if model == 'DT':
            # store the sample
            return self.DescionTree.predict(sample)[0]
        elif model == 'DN':
            # store the sample
            sample = self.preproces_deep(sample)
            return self.DeepNet.predict(sample)[0]
        else:
            return 'Model Not Found'