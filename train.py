import torch
import numpy as np
import pandas as pd
import torch.utils.data as data

from settings import data_path, celegans_seq
from src.data.preprocessing import (
    string_transform_labels,
    string_transform_onehot_char,
    smote_sampling,
    onehot_encode_kmers,
    onehot_encode,
    under_sample,
)

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
import lightgbm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

import src.data.utils as utils
from src.models.models import k_NN
from src.data.loader import DataLoader_training, DataLoader_split
from settings import *
from src.utils.utils import save_model

np.random.seed(seed)

# loading and preprocessing training data
preprocess_transforms = [onehot_encode]

if data == "humans":
    # loading and preprocessing training data
    train_loader = DataLoader_training(preprocess_X=preprocess_transforms)

    train_x = train_loader.x.copy()
    train_y = train_loader.y.copy()

    if predictionOnTestingSet:
        test_loader = DataLoader_sk(
            data_path + hum_seq_test, shuffle=False, preprocess_X=preprocess_transforms
        )
        test_x = test_loader.x.copy()
        test_y = test_loader.y.copy()

elif data == "celegans":
    loader = DataLoader_split(
        data_path + celegans_seq,
        preprocess_X=preprocess_transforms,
    )
    train_x = loader.train_x.copy()
    train_y = loader.train_y.copy()
    test_x = loader.test_x.copy()
    test_y = loader.test_y.copy()

else:
    print("data not available. Only 'humans' or 'celegans' DNA sequences.")
    exit()


# defining the models with its hyperparameters derived from tuning
models = {
    # "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=n_neighbors),
    "Logistic Regression": (LogisticRegression(class_weight="balanced"), None, None),
    "Linear Support Vector Machine": (LinearSVC(class_weight="balanced"), None, None),
    "Support Vector Machine": (SVC(class_weight="balanced"), [under_sample], [1]),
    "Gradient Boosting": (
        lightgbm.LGBMClassifier(
            n_estimators=100, num_leaves=20, class_weight="balanced"
        ),
        None,
        None,
    ),
    "MLP": (MLPClassifier(), None, None),
    "Random Forest": (RandomForestClassifier(class_weight="balanced"), None, None),
}

# training all models and save them thereafter
for name, (model, samplings, ratios) in models.items():

    # sampling
    if samplings is not None:
        for sampling, ratio in zip(samplings, ratios):
            train_x, train_y = sampling(train_x, train_y, ratio)

    print("### fitting model {} ###".format(name))
    model.fit(train_x, train_y)

    print("### saving trained model {} ###".format(name))
    save_model(model, name + "_" + data)

    if predictionOnTestingSet:
        # evaluating performance on given testing set
        predictions = model.predict(test_x)

        print("### performance on testing set ###")
        utils.model_eval(predictions, test_y)


print("### training completed ###")
