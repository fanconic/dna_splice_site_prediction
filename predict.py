import os
import numpy as np
import pandas as pd
import torch.utils.data as data
from src.data.preprocessing import (
    string_transform_labels,
    string_transform_onehot_char,
    smote_sampling,
    over_sample,
    under_sample,
    onehot_encode,
    onehot_encode_kmers,
)
import src.data.utils as utils
from src.data.loader import DataLoader_testing, DataLoader_split


# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from src.utils.utils import load_model
from settings import *

# loading and preprocessing testing data
preprocess_transforms = [onehot_encode]

np.random.seed(seed)

# differentiating among datasets
if data == "humans":
    loader = DataLoader_testing(preprocess_X=preprocess_transforms)
    test_x = loader.x.copy()

elif data == "celegans":
    loader = DataLoader_split(
        data_path + celegans_seq, preprocess_X=preprocess_transforms
    )
    test_x = loader.test_x.copy()

else:
    print("data not available. Only 'humans' or 'celegans' DNA sequences.")
    exit()

# models for testing (make sure the model name is exactly the same as in train.py)
models = [
    "K-Nearest Neighbours",
    "Logistic Regression",
    "Linear Support Vector Machine",
    "Support Vector Machine",
    "Gradient Boosting",
    "MLP",
    "Random Forest",
]

for name in models:
    print("### loading model {} ###".format(name))
    model = load_model(name + "_" + data)

    predictions = model.predict(test_x)
    try:
        predict_probas = model.predict_proba(test_x)[:, 1]
    except:
        predict_probas = model.decision_function(test_x)

    # saving model predictions
    with open(results_dir + name + "_" + data + "_results.npy", "wb") as file:
        np.save(file, predictions)
    with open(results_dir + name + "_" + data + "_probas.npy", "wb") as file:
        np.save(file, predict_probas)
    print("### predictions saved ###")

print("### process completed ###")
