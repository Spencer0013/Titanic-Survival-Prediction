import os
import sys
import re
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def extract_title(self, name):
        match = re.search(r',\s*([^\.]+)\.', name)
        return match.group(1).strip() if match else 'Missing'

    def extract_ticket_prefix(self, ticket):
        parts = ticket.split()
        if len(parts) > 1:
            return parts[0].replace('.', '').replace('/', '').upper()
        else:
            return 'None'

    def get_data_transformer_object(self, input_df):
        try:
            numerical_columns = [col for col in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'family'] if col in input_df.columns]
            categorical_columns = [col for col in ['title', 'Sex', 'Embarked', 'ticket_update'] if col in input_df.columns]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Numerical columns used: {numerical_columns}")
            logging.info(f"Categorical columns used: {categorical_columns}")

            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data successfully.")

            # Feature Engineering
            for df in [train_df, test_df]:
                df['family'] = df['SibSp'] + df['Parch']
                df['ticket_update'] = df['Ticket'].apply(self.extract_ticket_prefix)
                df['title'] = df['Name'].apply(self.extract_title)

            # Drop unused columns
            drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
            train_df.drop(columns=drop_cols, inplace=True, errors='ignore')
            test_df.drop(columns=drop_cols, inplace=True, errors='ignore')
            logging.info("Dropped Unnecessary Columns")

            target_column_name = "Survived"

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            preprocessing_obj = self.get_data_transformer_object(input_feature_train_df)

            logging.info("Applying preprocessing pipeline to train and test data.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Convert sparse matrix to dense numpy arrays if necessary
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()

            print("Before concatenation:")
            print("input_feature_train_arr shape:", input_feature_train_arr.shape)
            print("type(input_feature_train_arr):", type(input_feature_train_arr))
            print("target_feature_train_df shape:", target_feature_train_df.shape)
            print("target_feature_train_df.to_numpy() shape:", target_feature_train_df.to_numpy().shape)
            print("target_feature_train_df.to_numpy() ndim:", target_feature_train_df.to_numpy().ndim)

            train_arr = np.hstack((input_feature_train_arr, target_feature_train_df.to_numpy().reshape(-1, 1)))
            test_arr = np.hstack((input_feature_test_arr, target_feature_test_df.to_numpy().reshape(-1, 1)))

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing object saved successfully.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)





        








  