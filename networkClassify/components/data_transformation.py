import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from networkClassify.entity.artifact_entity import DataIngestionArtifact
from networkClassify.constants.training_pipeline import TARGET_COLUMN
from networkClassify.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from networkClassify.entity.artifact_entity import (
    DataTransformationArtifact,
)

from networkClassify.entity.config_entity import DataTransformationConfig
from networkClassify.exception.exception import NetworkSecurityException 
from networkClassify.logging.logger import logging
from networkClassify.utils.main_utils.utils import save_numpy_array_data,save_object

class DataTransformation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_ingestion_artifact:DataIngestionArtifact=data_ingestion_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def get_data_transformer_object(cls)->Pipeline:
        logging.info(
            "Entered get_data_trnasformer_object method of Trnasformation class"
        )
        try:
           imputer:KNNImputer=KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
           logging.info(
                f"Initialise KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )
           processor:Pipeline=Pipeline([("imputer",imputer)])
           return processor
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    
    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            train_df=DataTransformation.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df=DataTransformation.read_data(self.data_ingestion_artifact.test_file_path)

            ## training dataframe
            input_feature_train_df=train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            # target_feature_train_df = target_feature_train_df.replace(-1, 0)

            #testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            # target_feature_test_df = target_feature_test_df.replace(-1, 0)

            label_encoder = LabelEncoder()

            target_feature_train_df = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_df = label_encoder.transform(target_feature_test_df)

            preprocessor=self.get_data_transformer_object()

            preprocessor_object=preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature=preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature =preprocessor_object.transform(input_feature_test_df)
             

            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df) ]
            test_arr = np.c_[ transformed_input_test_feature, np.array(target_feature_test_df) ]

            #save numpy array data
            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)
            save_object( self.data_transformation_config.encoder_object_file_path, label_encoder,)

            save_object( "final_model/preprocessor.pkl", preprocessor_object,)
            save_object("final_model/label_encoder.pkl",label_encoder,)

            #preparing artifacts

            data_transformation_artifact=DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                label_encoder_file_path=self.data_transformation_config.encoder_object_file_path

            )
            return data_transformation_artifact


            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
