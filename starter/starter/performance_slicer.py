import pandas as pd
from ml.data import process_data
from ml.model import inference, compute_model_metrics

# Load data and model files
data = pd.read_csv(r"starter/data/census.csv")
model = pd.read_pickle(r"starter/model/model.pkl")
encoder = pd.read_pickle(r"starter/model/encoder.pkl")
lb = pd.read_pickle(r"starter/model/lb.pkl")

def performance_slice(model, data, col, encoder, lb):

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

    # Get each slice from categorical feature
    slices = data.loc[:, col].unique()

    # Write to slice_output.txt
    file = open("./slice_output.txt", "w")
    file.write(f"Performance for the slices of feature: {col}")
    file.write("\n")
    file.write("\n")

    for slice in slices:
        file.write(f"{slice}:")
        file.write("\n")
        sliced_data = data.loc[data.loc[:, col] == slice,:]
        X_test, y_test, encoder, lb = process_data(
            sliced_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)
        preds = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        file.write(f"Precision: {precision}")
        file.write("\n")
        file.write(f"Recall: {recall}")
        file.write("\n")
        file.write(f"fbeta: {fbeta}")
        file.write("\n")
        file.write("\n")
    file.close()


# Test on education column
if __name__ == "__main__":
    performance_slice(model, data, "education", encoder, lb)