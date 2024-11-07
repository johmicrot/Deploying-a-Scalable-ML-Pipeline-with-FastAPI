import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)


# Define the compute_metrics_on_slices function here (as shown above)
def compute_metrics_on_slices(
    data: pd.DataFrame,
    categorical_features: list,
    label: str,
    model,
    encoder,
    lb,
    output_file: str = "slice_output.txt"
):
    """
    Computes model metrics on slices of the data and writes the results to a structured file.

    Parameters:
    - data (pd.DataFrame): The dataset to slice and evaluate.
    - categorical_features (list): List of categorical feature names.
    - label (str): The target label column name.
    - model: The trained machine learning model.
    - encoder: The trained encoder for categorical features.
    - lb: The trained label binarizer.
    - output_file (str): The path to the output file.
    """
    # Define the header with fixed-width columns
    header = f"{'Feature':<20} {'Value':<35} {'Count':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}\n"
    separator = "-" * (20 + 35 + 10 + 10 + 10 + 10) + "\n"

    with open(output_file, "w") as f:
        # Write the header and separator
        f.write(header)
        f.write(separator)

        for col in categorical_features:
            unique_values = sorted(data[col].unique())
            for slice_value in unique_values:
                slice_data = data[data[col] == slice_value]
                count = slice_data.shape[0]

                # Skip slices with very few samples to avoid unreliable metrics
                if count < 10:
                    f.write(f"{col:<20} {slice_value:<35} {count:>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}\n")
                    continue

                p, r, fb = performance_on_categorical_slice(
                    data=slice_data,
                    column_name=col,
                    slice_value=slice_value,
                    categorical_features=categorical_features,
                    label=label,
                    encoder=encoder,
                    lb=lb,
                    model=model
                )

                # Write the metrics in fixed-width columns
                f.write(f"{col:<20} {slice_value:<35} {count:>10,} {p:>10.4f} {r:>10.4f} {fb:>10.4f}\n")



# TODO: Load the census.csv data
project_path = "."
data_path = os.path.join(project_path, "data", "census.csv")
print(f"Loading data from: {data_path}")
data = pd.read_csv(data_path)

# TODO: Split the data into train and test datasets
train, test = train_test_split(data, test_size=0.2, random_state=42)

# DO NOT MODIFY: Define categorical features
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

# TODO: Process the training data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label='salary',
    training=True
)

# Process the testing data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# TODO: Train the model
model = train_model(X_train, y_train)

# Save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)

# Load the model to verify
model = load_model(model_path)

# TODO: Run inference on the test dataset
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# TODO: Compute performance on model slices and write to slice_output.txt
compute_metrics_on_slices(
    data=test,
    categorical_features=cat_features,
    label="salary",
    model=model,
    encoder=encoder,
    lb=lb,
    output_file="slice_output.txt"
)

print("Slice metrics have been written to slice_output.txt")
