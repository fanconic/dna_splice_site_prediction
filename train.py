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
from src.data.loader import DataLoader_training, DataLoader_split, DataLoader_sk
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
        test_x = test_loader.x
        test_y = test_loader.y

elif data == "celegans":
    loader = DataLoader_split(
        data_path + celegans_seq, preprocess_X=preprocess_transforms, save_test_df=False
    )
    train_x = loader.train_x
    train_y = loader.train_y
    test_x = loader.test_x
    test_y = loader.test_y

else:
    print("data not available. Only 'humans' or 'celegans' DNA sequences.")
    exit()


# defining the models with its hyperparameters derived from tuning
models = {
    "K-Nearest Neighbours": (
        KNeighborsClassifier(n_neighbors=14, p=1),
        [under_sample],
        [0.3],
    ),
    "Logistic Regression": (LogisticRegression(class_weight="balanced"), None, None),
    "Linear Support Vector Machine": (LinearSVC(class_weight="balanced"), None, None),
    "Support Vector Machine": (SVC(class_weight="balanced"), [under_sample], [1]),
    "Gradient Boosting": (
        lightgbm.LGBMClassifier(n_estimators=500, num_leaves=50, random_state=seed),
        [under_sample, smote_sampling],
        [0.5, 1],
    ),
    "MLP": (
        MLPClassifier(
            hidden_layer_sizes=(1592,),
            activation="tanh",
            batch_size=200,
            early_stopping=False,
        ),
        [under_sample],
        [0.3],
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=700, max_features=50, random_state=seed),
        [under_sample, smote_sampling],
        [0.1, 1],
    ),
}

# training all models and save them thereafter
for name, (model, samplings, ratios) in models.items():
    # sampling
    if samplings is not None:
        for sampling, ratio in zip(samplings, ratios):
            train_x_sampled, train_y_sampled = sampling(train_x, train_y, ratio)
    else:
        train_x_sampled, train_y_sampled = train_x, train_y

    print("### fitting model {} ###".format(name))
    model.fit(train_x_sampled, train_y_sampled)

    print("### saving trained model {} ###".format(name))
    save_model(model, name + "_" + data)

    if predictionOnTestingSet:
        # evaluating performance on given testing set
        predictions = model.predict(test_x)
        try:
            predict_probas = model.predict_proba(test_x)[:, 1]
        except:
            predict_probas = model.decision_function(test_x)

        print("### performance on testing set ###")
        utils.model_eval(predictions, test_y, predict_probas)


print("### training completed ###")
