import os
import sys
import mlflow

from networkClassify.exception.exception import NetworkSecurityException 
from networkClassify.logging.logger import logging

from networkClassify.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networkClassify.entity.config_entity import ModelTrainerConfig



from networkClassify.utils.ml_utils.model.estimator import NetworkModel
from networkClassify.utils.main_utils.utils import save_object,load_object
from networkClassify.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networkClassify.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
#  from urllib.parse import urlparse


# os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/ankityadavdm/ML_Network_Project.mlflow"
# os.environ["MLFLOW_TRACKING_USERNAME"] = "ankityadavdm"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "<be32e038d689637213b9e45628db9dc77b21ade0>"
# be32e038d689637213b9e45628db9dc77b21ade0
# runmlflowtoken 
#  2ef9dc84997807e30e446d81563efe2eec3e4489
# e4d0b5b2adac39bdb5bc9ac520d96db2aad7e342


import dagshub

# dagshub.auth.add_app_token(
#     username=os.getenv("DAGSHUB_USER"),
#     token=os.getenv("DAGSHUB_TOKEN"),
# )

dagshub.init(repo_owner='ankityadavdm', repo_name='Modular-N-W-classify', mlflow=True)




class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def track_mlflow(self,best_model,best_model_name,train_metric,test_metric):
        # mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("Network_Classify")

        with mlflow.start_run():
            logging.info("MLFLOW TRACKING STARTED")
            mlflow.log_param("model_name",best_model_name)
            mlflow.log_metric("train_accuracy",train_metric.accuracy_score)
            mlflow.log_metric("test_accuracy",test_metric.accuracy_score)
            mlflow.log_metric("train_f1_score",train_metric.f1_score)
            mlflow.log_metric("test_f1_score",test_metric.f1_score)
            mlflow.log_metric("train_precision",train_metric.precision_score)
            mlflow.log_metric("test_precision",test_metric.precision_score)
            mlflow.log_metric("train_recall",train_metric.recall_score)
            mlflow.log_metric("test_recall",test_metric.recall_score)

            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model"
            )
        logging.info("MLFLOW TRACKING ENDED")

    
    def train_model(self,X_train,y_train,x_test,y_test):
        models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            }

        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                          models=models)
        
        best_model_score = max(sorted(model_report.values()))

        ## To get best model name from dict

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        y_train_pred=best_model.predict(X_train)

        classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)
        le=load_object(self.data_transformation_artifact.label_encoder_file_path)
        print(classification_report(y_train,y_train_pred,target_names=le.classes_))
        
        print(f"Model with best accuracy: ",best_model_name)
        print(classification_train_metric)
        ## Track the experiements with mlflow
        # self.track_mlflow(best_model,classification_train_metric,)

        y_test_pred=best_model.predict(x_test)

        print(classification_report(y_test,y_test_pred,target_names=le.classes_))
        
        classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)
        print(classification_test_metric)
        # self.track_mlflow(best_model,classification_test_metric,)

        self.track_mlflow(
            best_model=best_model,
            best_model_name=best_model_name,
            train_metric=classification_train_metric,
            test_metric=classification_test_metric
        )

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        encoder=load_object(file_path=self.data_transformation_artifact.label_encoder_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        Network_Model=NetworkModel(preprocessor=preprocessor,encoder=encoder,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=Network_Model)
        #model pusher
        save_object("final_model/model.pkl",best_model)
        

        ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric
                             )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact


        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

            
        except Exception as e:
            raise NetworkSecurityException(e,sys)