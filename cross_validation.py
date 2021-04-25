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
from src.data.loader import DataLoader_folds

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
import lightgbm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from src.utils.utils import save_model

# import all the settings variables for the models
from settings import *

preprocess_transforms = [onehot_encode]
if data == "humans":
    kfold_obj = DataLoader_folds(
        data_path + hum_seq_train, n_folds, preprocess_X=preprocess_transforms
    )

elif data == "celegans":
    kfold_obj = DataLoader_folds(
        data_path + celegans_seq, n_folds, preprocess_X=preprocess_transforms
    )

else:
    print("data not available. Only 'humans' or 'celegans' DNA sequences.")
    exit()

models = {
    "K-Nearest Neighbours": (
        KNeighborsClassifier(n_neighbors=14, p=1),
        [under_sample, smote_sampling],
        [0.3, 1],
    ),
    "Logistic Regression": (LogisticRegression(class_weight="balanced"), None, None),
    "Linear Support Vector Machine": (LinearSVC(class_weight="balanced"), None, None),
    "Support Vector Machine": (SVC(class_weight="balanced"), [under_sample], [1]),
    "Gradient Boosting": (
        lightgbm.LGBMClassifier(n_estimators=500, num_leaves=50, random_state=seed),
        [under_sample, smote_sampling],
        [0.5, 1],
    ),
    "MLP": (MLPClassifier(), [under_sample], [0.3]),
    "Random Forest": (
        RandomForestClassifier(n_estimators=700, max_features=50, random_state=seed),
        [under_sample, smote_sampling],
        [0.01, 1],
    ),
}


# K_Fold iteration loop
for name, (model, samplings, ratios) in models.items():
    print(name)
    roc_auc_collect = []
    auprc_collect = []
    for fold, (train_idx, dev_idx) in enumerate(
        kfold_obj.kfold.split(kfold_obj.x, kfold_obj.y)
    ):
        # creating sets
        train_x = kfold_obj.x[train_idx]
        train_y = kfold_obj.y[train_idx]
        test_x = kfold_obj.x[dev_idx]
        test_y = kfold_obj.y[dev_idx]

        # sampling
        if samplings is not None:
            for sampling, ratio in zip(samplings, ratios):
                train_x, train_y = sampling(train_x, train_y, ratio)

        # model training & testing
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        try:
            predict_probas = model.predict_proba(test_x)[:, 1]
        except:
            predict_probas = model.decision_function(test_x)

        print("### FOLD {} ###".format(fold))
        roc_auc, auprc = utils.model_eval(predictions, test_y, predict_probas)
        roc_auc_collect.append(roc_auc)
        auprc_collect.append(auprc)

    print(
        "AUPRC score mean: {0:.4f}+-{1:.4f}\n".format(
            np.mean(auprc_collect), np.std(auprc_collect)
        ),
        "AUC score mean: {0:.4f}+-{1:.4f}\n".format(
            np.mean(roc_auc_collect), np.std(roc_auc_collect)
        ),
    )
