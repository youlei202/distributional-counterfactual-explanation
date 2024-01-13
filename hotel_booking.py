# main.py

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from models.mlp import BlackBoxModel
import pickle
import os
from explainers.dce import DistributionalCounterfactualExplainer
from explainers.distances import bootstrap_1d, bootstrap_sw
from utils.logger_config import setup_logger


logger = setup_logger()

absolute_path = "/zhome/b2/8/197929/GitHub/distributional-counterfactual-explanation"

read_data_path = os.path.join(absolute_path, "data/hotel_booking")
dump_data_path = os.path.join(absolute_path, "data/hotel_booking/")
csv_file = "hotel_bookings.csv"
target_name = "is_canceled"

# Parameters
sample_num = 60
delta = 0.25
alpha = 0.10
U_1 = 0.5
U_2 = 0.5
interval_left = 0.2
interval_right = 1.0

y_target = torch.distributions.beta.Beta(0.1, 0.9).sample((sample_num,))


def main():
    # Load dataset and create a copy for manipulation
    df_ = pd.read_csv(os.path.join(read_data_path, csv_file))
    df = df_.copy()

    logger.info("Dataset loaded.")

    # Define target column
    target = df[target_name]

    # Initialize a label encoder and a dictionary to store label mappings
    label_encoder = LabelEncoder()
    label_mappings = {}

    # Convert categorical columns to numerical representations using label encoding
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].fillna("Unknown")  # Handle missing values
            df[column] = label_encoder.fit_transform(df[column])
            label_mappings[column] = dict(
                zip(label_encoder.classes_, range(len(label_encoder.classes_)))
            )

    # Impute missing values in numerical columns with their median
    for column in df.columns:
        if df[column].isna().any():
            median_val = df[column].median()
            df[column].fillna(median_val, inplace=True)

    logger.info("Data preprocessing done.")

    # Define features for model training
    features = ["lead_time", "booking_changes"]

    df_X = df[features].copy()
    df_y = target

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2)

    # Normalize training and test datasets
    std = X_train.std()
    mean = X_train.mean()
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)

    # Model initialization, defining loss and optimizer
    model = BlackBoxModel(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Model training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Model evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        y_pred_tensor = (test_outputs > 0.5).float()
        correct_predictions = (y_pred_tensor == y_test_tensor).float().sum()
        accuracy = correct_predictions / y_test_tensor.shape[0]

    logger.info(f"Accuracy = {accuracy.item()}")

    indice = (X_test.sample(sample_num)).index
    X = X_test.loc[indice].values
    y = model(torch.FloatTensor(X))
    y_true = y_test.loc[indice]

    # Counterfactual explanation
    logger.info("Counterfactual explanation optimization started.")
    explainer = DistributionalCounterfactualExplainer(
        model=model,
        X=X,
        y_target=y_target,
        lr=1e-1,
        epsilon=0.5,
        lambda_val=100,
        delta=delta,
        n_proj=500,
    )

    logger.info(
        f"WD: {np.sqrt(explainer.wd.distance(y, y_target, delta=delta)[0].item())}"
    )
    logger.info(
        f"WD Exact Interval: {explainer.wd.distance_interval(y, y_target, delta=delta, alpha=alpha)}",
    )
    logger.info(
        f"WD Bootstrap Interval: {bootstrap_1d(y, y_target, delta=delta, alpha=alpha)}"
    )

    explainer.optimize(
        U_1=U_1, U_2=U_2, l=interval_left, r=interval_right, max_iter=20, tau=1e3
    )

    logger.info(
        f"SWD: {np.sqrt(explainer.swd.distance(explainer.best_X, explainer.X_prime, delta)[0].item())}",
    )
    logger.info(
        f"SWD Exact Interval: {explainer.swd.distance_interval(explainer.X, explainer.X_prime, delta, alpha=alpha)}",
    )
    logger.info(
        f"SWD Bootstrap Interval: {bootstrap_sw(explainer.best_X, explainer.X_prime, delta=delta, alpha=alpha)}",
    )

    factual_X = df[df_X.columns].loc[indice].copy()
    counterfactual_X = pd.DataFrame(
        explainer.best_X.detach().to("cpu").numpy() * std[df_X.columns].values
        + mean[df_X.columns].values,
        columns=df_X.columns,
    )
    factual_y = pd.DataFrame(
        y.detach().to("cpu").numpy(), columns=[target_name], index=factual_X.index
    )
    counterfactual_y = pd.DataFrame(
        explainer.best_y.detach().to("cpu").numpy(),
        columns=[target_name],
        index=factual_X.index,
    )

    counterfactual_X.index = factual_X.index
    counterfactual_X[target_name] = counterfactual_y
    factual_X[target_name] = factual_y

    factual_X.to_csv(os.path.join(dump_data_path, "factual.csv"), index=False)
    counterfactual_X.to_csv(
        os.path.join(dump_data_path, "counterfactual.csv"), index=False
    )
    with open(os.path.join(dump_data_path, "explainer.pkl"), "wb") as file:
        pickle.dump(explainer, file)
    logger.info("Files dumped.")


if __name__ == "__main__":
    logger.info("Hotel booking analysis started")
    main()
