from dataclasses import dataclass

@dataclass 
class DataIngestionArtifact:
    trained_file_path:str
    test_file_path:str

@dataclass 
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    label_encoder_file_path: str


@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score:float
    recall_score:float
    accuracy_score:float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact
