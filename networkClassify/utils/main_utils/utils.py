from networkClassify.exception.exception import NetworkSecurityException
from networkClassify.logging.logger import logging
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,  precision_score, recall_score, f1_score, r2_score 
# import yaml
import os,sys,pickle
import numpy as np
import optuna

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)

# def read_yaml_file(file_path: str)->dict:
#     try:
#         with open(file_path,"rb") as yaml_file:
#             return yaml.safe_load(yaml_file)
#     except Exception as e:
#         raise NetworkSecurityException(e,sys) from e
    
# def write_yaml_file(file_path:str,content:object,replace:bool=False)->None:
#     try:
#         if replace:
#             if os.path.exists(file_path):
#                 os.remove(file_path)
#         os.makedirs(os.path.dirname(file_path),exist_ok=True)
#         with open(file_path,"w") as file:
#             yaml.dump(content,file)
#     except Exception as e:
#         raise NetworkSecurityException(e,sys)

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
  

    
def evaluate_models(X_train, y_train,X_test,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            best_params = tune_model_optuna(
                list(models.keys())[i],
                X_train,
                y_train
            )

            model.set_params(**best_params)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)

            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def tune_model_optuna(model_name, X_train, y_train):

    def objective(trial):

        if model_name == "Random Forest":

            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 50, 300
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", 5, 30
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 10
                )
            }

            model = RandomForestClassifier(
                **params,
                random_state=42,
                n_jobs=-1
            )

        elif model_name == "Gradient Boosting":

            params = {

                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.05,
                    0.15
                ),

                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    20,
                    80
                ),

                "subsample": trial.suggest_float(
                    "subsample",
                    0.8,
                    1.0
                )
            }

            model = GradientBoostingClassifier(
                **params,
                random_state=42
            )

        elif model_name == "Decision Tree":

            params = {
                "max_depth": trial.suggest_int(
                    "max_depth",
                    3,
                    20
                ),

                "criterion": trial.suggest_categorical(
                    "criterion",
                    ["gini", "entropy"]
                )
            }

            model = DecisionTreeClassifier(
                **params,
                random_state=42
            )

        score = cross_val_score(
            model,
            X_train,
            y_train,
            cv=3,
            scoring="accuracy"
        ).mean()

        return score

    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=5)

    return study.best_params
    