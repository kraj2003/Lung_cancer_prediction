import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:

            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 GENDER:int,AGE:int,SMOKING:int,YELLOW_FINGERS:int,ANXIETY:int,PEER_PRESSURE:int,CHRONIC_DISEASE:int,
                 FATIGUE :int,ALLERGY:int,WHEEZING:int,
                 ALCOHOL_CONSUMING:int,COUGHING:int,SHORTNESS_OF_BREATH:int,SWALLOWING_DIFFICULTY:int,CHEST_PAIN:int
                ):
        self.GENDER=GENDER
        self.AGE=AGE
        self.SMOKING=SMOKING
        self.ANXIETY=ANXIETY
        self.PEER_PRESSURE=PEER_PRESSURE
        self.FATIGUE=FATIGUE
        self.WHEEZING=WHEEZING
        self.ALLERGY=ALLERGY
        self.ALCOHOL_CONSUMING=ALCOHOL_CONSUMING
        self.COUGHING=COUGHING
        self.SWALLOWING_DIFFICULTY=SWALLOWING_DIFFICULTY
        self.SHORTNESS_OF_BREATH=SHORTNESS_OF_BREATH
        self.CHEST_PAIN=CHEST_PAIN
        self.YELLOW_FINGERS=YELLOW_FINGERS
        self.CHRONIC_DISEASE=CHRONIC_DISEASE

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "GENDER":[self.GENDER],
                "AGE": [self.AGE],
                "SMOKING":[self.SMOKING],
                "PEER_PRESSURE":[self.PEER_PRESSURE], 
                "ANXIETY":[self.ANXIETY],
                "FATIGUE ":[self.FATIGUE],
                "WHEEZING":[self.WHEEZING],
                "ALLERGY ":[self.ALLERGY],
                "COUGHING":[self.COUGHING],
                "ALCOHOL CONSUMING":[self.ALCOHOL_CONSUMING],
                "SWALLOWING DIFFICULTY":[self.SWALLOWING_DIFFICULTY],
                "SHORTNESS OF BREATH":[self.SHORTNESS_OF_BREATH],
                "CHRONIC DISEASE":[self.CHRONIC_DISEASE],
                "YELLOW_FINGERS":[self.YELLOW_FINGERS],
                "CHEST PAIN":[self.CHEST_PAIN],

            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)


    