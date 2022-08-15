# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Add code to load in the data.
data = pd.read_csv(r"starter/data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
pd.to_pickle(model, "starter/model/model.pkl")

# Save encode and LB
pd.to_pickle(encoder, "starter/model/encoder.pkl")
pd.to_pickle(lb, "starter/model/lb.pkl")

# Calculate classification metrics on test set
preds = inference(model, X_test)
metrics = compute_model_metrics(y_test, preds)

# Write to slice_output.txt
file = open("./test_metrics.txt", "w")
file.write(f"Test metrics")
file.write("\n")
file.write("\n")
file.write(f"precision: {metrics[0]}")
file.write("\n")
file.write(f"recall: {metrics[1]}")
file.write("\n")
file.write(f"fbeta: {metrics[2]}")
file.write("\n")
file.close()
