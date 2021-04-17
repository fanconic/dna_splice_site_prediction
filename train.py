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
from sklearn.svm import SVC
import lightgbm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

import src.data.utils as utils
from src.models.models import k_NN
from src.data.loader import DataLoader_training, DataLoader_sk
from settings import *
from src.utils.utils import save_model


# defining the models with its hyperparameters derived from tuning
models = {
    # "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=n_neighbors),
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "Gradient Boosting": lightgbm.LGBMClassifier(n_estimators=100, num_leaves=20),
    # "MLP": MLPClassifier(),
    "Random Forest": RandomForestClassifier(),
}

# loading and preprocessing training data
preprocess_transforms = [onehot_encode]
train = DataLoader_training(preprocess_X=preprocess_transforms)

train.x, train.y = under_sample(train.x, train.y, 1)
train.x, train.y = smote_sampling(train.x, train.y)

# training all models and save them thereafter
for name, model in models.items():
    print("### fitting model {} ###".format(name))
    model.fit(train.x, train.y)

    print("### saving trained model {} ###".format(name))
    save_model(model, name)

    if predictionOnTestingSet:
        # evaluating performance on given testing set
        test = DataLoader_sk(
            data_path + hum_seq_test, shuffle=False, preprocess_X=preprocess_transforms
        )
        predictions = model.predict(test.x)

        print("### performance on testing set ###")
        utils.model_eval(predictions, test.y)


print("### training completed ###")
