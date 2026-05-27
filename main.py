from networkClassify.components.data_ingestion import DataIngestion
# from networkClassify.components.data_validation import DataValidation
# from networkClassify.components.data_transformation import DataTransformation
# from networkClassify.components.model_trainer import ModelTrainer

from networkClassify.exception.exception import NetworkSecurityException
from networkClassify.logging.logger import logging
from networkClassify.entity.config_entity import DataIngestionConfig
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
    except Exception as e:
        raise NetworkSecurityException(e,sys)