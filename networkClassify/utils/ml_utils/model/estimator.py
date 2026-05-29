
import os
import sys

from networkClassify.constants.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME
from networkClassify.exception.exception import NetworkSecurityException
from networkClassify.logging.logger import logging

class NetworkModel:
    def __init__(self,preprocessor,encoder,model):
        try:
            self.preprocessor = preprocessor
            self.encoder=encoder
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            prediction = self.encoder.inverse_transform(y_hat)

            return prediction
        except Exception as e:
            raise NetworkSecurityException(e,sys)
