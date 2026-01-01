import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model, hyper_parameter_tuning


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(
                "Split train and test data into independent and target features")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "SVR(Support Vector)": SVR(),
                "KNN": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "GradientBoostRegressor": GradientBoostingRegressor()
            }

            logging.info("Evaluation on different models")
            model_report: dict = evaluate_model(
                X_train, y_train, X_test, y_test, models=models)

            # best model score from dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(
                model_report.values()).index(best_model_score)]

            logging.info("Running Hyperparameter Tuning on model with highest score")
            # send the best model for hyperparameter tuning
            hyper_best_score_, hyper_train_model_score, hyper_test_model_score = hyper_parameter_tuning(best_model_name,models[best_model_name], X_train, y_train, X_test, y_test)

            if best_model_score < 0.6 and hyper_best_score_ < 0.6:
                raise CustomException("No Best Model Found")

            best_model = models[best_model_name]

            logging.info(f"Best Model found on both training and testing dataset {best_model_name}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,obj =best_model
            )

            predicted =best_model.predict(X_test)
            r2Score =r2_score(y_test,predicted)

            return r2Score
            
        except Exception as e:
            raise CustomException(e,sys)
