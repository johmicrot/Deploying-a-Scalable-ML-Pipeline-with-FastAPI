# test_ml.py

import pytest
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import your functions from the ml module
from ml.model import train_model, inference, save_model, load_model
from ml.data import process_data

# Define the categorical features as per your main script. It would be better to move this to a config file
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def data():
    """
    Fixture to load the data only once for all tests.
    """
    data_path = os.path.join("data", "census.csv")
    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope="module")
def processed_data(data):
    """
    Fixture to process the data for testing.
    Returns:
        X_train, y_train, X_test, y_test, encoder, lb
    """
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    return X_train, y_train, X_test, y_test, encoder, lb


def test_train_model(processed_data):
    """
    Test that the train_model function returns a RandomForestClassifier instance.
    """
    X_train, y_train, _, _, _, _ = processed_data
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "The trained model should be a RandomForestClassifier instance."


def test_inference(processed_data):
    """
    Test that the inference function returns predictions of the correct type and length.
    """
    X_train, y_train, X_test, _, _, _ = processed_data
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray), "Predictions should be a numpy array."
    assert len(preds) == len(X_test), "Number of predictions should match number of test samples."


def test_save_and_load_model(processed_data, tmp_path):
    """
    Test that the model can be saved and loaded correctly.
    """
    X_train, y_train, _, _, _, _ = processed_data
    model = train_model(X_train, y_train)

    # Save the model
    model_path = tmp_path / "model.pkl"
    save_model(model, model_path)
    assert os.path.exists(model_path), "Model file was not saved."

    # Load the model
    loaded_model = load_model(model_path)
    assert isinstance(loaded_model, RandomForestClassifier), "Loaded model should be a RandomForestClassifier instance."

    # Ensure the loaded model can make predictions
    preds = inference(loaded_model, X_train)
    assert len(preds) == len(y_train), "Loaded model predictions do not match input size."
