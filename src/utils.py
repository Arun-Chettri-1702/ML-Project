import os
import sys
import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def evaluate_model(X_train, y_train, X_test, y_test, models:dict):
    try:
        report ={}
        for i in range(len(models)):
            model = list(models.values())[i]
            model_name =list(models.keys())[i]

            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred =model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)

def hyper_parameter_tuning(model_name,model,X_train,y_train,X_test,y_test):
    try:
        param_grids = {

            "LinearRegression": {
                "fit_intercept": [True, False],
                "positive": [True, False]
            },

            "Ridge": {
                "alpha": [0.01, 0.1, 1, 10, 100],
                "solver": ["auto", "svd", "cholesky"]
            },

            "Lasso": {
                "alpha": [0.001, 0.01, 0.1, 1],
                "max_iter": [1000, 5000]
            },

            "SVR(Support Vector)": {
                "kernel": ["rbf", "linear"],
                "C": [0.1, 1, 10],
                "epsilon": [0.01, 0.1]
            },

            "KNN": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "metric": ["minkowski"]
            },

            "Decision Tree": {
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },

            "XGBRegressor": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 6],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            },

            "CatBoostRegressor": {
                "iterations": [200, 500],
                "learning_rate": [0.05, 0.1],
                "depth": [6, 8],
                "l2_leaf_reg": [3, 5]
            },

            "AdaBoostRegressor": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.05, 0.1, 1.0],
                "loss": ["linear", "square"]
            },

            "GradientBoostRegressor": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]
            }
        }


        PARAMS_GRID = param_grids[model_name]

        grid =GridSearchCV(model,param_grid=PARAMS_GRID,cv=5,scoring="r2",n_jobs=-1)

        grid.fit(X_train,y_train)

        y_train_pred =grid.predict(X_train)
        y_test_pred =grid.predict(X_test)

        train_model_score = r2_score(y_train,y_train_pred)
        test_model_score = r2_score(y_test,y_test_pred)

        return grid.best_score_,train_model_score,test_model_score
    except Exception as e:
        raise CustomException(e,sys)

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        CustomException(e,sys)
