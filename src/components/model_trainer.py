import numpy as np
import pandas as pd
from dataclasses import dataclass
import os
import sys

from sklearn.metrics import mean_squared_error, confusion_matrix,classification_report,precision_score,recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Ridge,Lasso, LogisticRegression
from sklearn.svm import SVC

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models,save_object

@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join("artifacts",'model.pkl')
    
    
class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config=ModelTrainerConfig()

    def InitiateModelTrainer(self,train_array,test_array):

        try: 
            logging.info("Splitting training and testing data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Logistic Regression":LogisticRegression(),
                # "SVM":SVC(),
                "Decision Tree":DecisionTreeClassifier(),
                "KNN":KNeighborsClassifier()
            }

            # hyperparameters={
            #     "Decision Tree": {
            #             'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            #             # 'splitter':['best','random'],
            #             # 'max_features':['sqrt','log2'],
            #         },
            #     "KNN":{
            #         'n_estimators': [8,16,32,64,128,256]
            #     },
            #     "SVM":{'C': [0.1, 1, 10, 100, 1000],  
            #       'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
            #       'kernel': ['rbf']} 
            # }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            print(model_report)
            logging.info(f"model report {model_report}")
            # # to get best model score from dict
            best_model_score=max(sorted(model_report.values()))
            
            ## 
            best_model_name=list(model_report.keys())[
                    list(model_report.values()).index(best_model_score)]
                
            best_model=models[best_model_name]

            if best_model_score<0.7:
                    raise CustomException("no best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            recall_scores=recall_score(y_test,predicted)
            return recall_scores

        except Exception as e:
            raise CustomException(e,sys)
