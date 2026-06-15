from networkClassify.components.data_ingestion import DataIngestion
from networkClassify.components.data_transformation import DataTransformation
from networkClassify.components.model_trainer import ModelTrainer

from networkClassify.exception.exception import NetworkSecurityException
from networkClassify.logging.logger import logging
from networkClassify.entity.config_entity import DataIngestionConfig,DataTransformationConfig,ModelTrainerConfig
from networkClassify.entity.config_entity import TrainingPipelineConfig
import sys

if __name__=='__main__':
    try:
        trainingPipelineConfig=TrainingPipelineConfig()
        dataIngestionConfig=DataIngestionConfig(trainingPipelineConfig)
        data_ingestion=DataIngestion(dataIngestionConfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        print(dataingestionartifact)

        data_transformation_config=DataTransformationConfig(trainingPipelineConfig)
        logging.info("data Transformation started")
        data_transformation=DataTransformation(dataingestionartifact,data_transformation_config)
        
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("data transformation completed")

        logging.info("Model Training Started")

        
        model_trainer_config=ModelTrainerConfig(trainingPipelineConfig)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact=model_trainer.initiate_model_trainer()

        logging.info("Model Training artifact created ")

        
    except Exception as e:
        raise NetworkSecurityException(e,sys)