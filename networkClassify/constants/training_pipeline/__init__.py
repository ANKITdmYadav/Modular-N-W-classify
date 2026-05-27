import os,sys
import numpy as np
import pandas as pd

# COMMON CONSTANT VARIABLE FOR TRAINING PIPELINE

TARGET_COLUMN :str="ProtocolName"
PIPELINE_NAME:str="NetworkClassify"
ARTIFACT_DIR:str="Artifacts"
FILE_NAME:str="final_nw_dataset.csv"
TRAIN_FILE_NAME:str="train.csv"
TEST_FILE_NAME:str="test.csv"

SCHEMA_FILE_PATH=os.path.join("data_schema","schema.yaml")

SAVED_MODEL_DIR=os.path.join("saved_models")
MODEL_FILE_NAME="model.pkl"

# Data ingestion related constant

DATA_INGESTION_COLLECTION_NAME: str="NetworkDataCPT"
DATA_INGESTION_DATABASE_NAME:str="ANKITAI"
DATA_INGESTION_DIR_NAME: str="data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str="feature_store"
DATA_INGESTION_INGESTED_DIR: str="ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float=0.2

