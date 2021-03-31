import numpy as np
import pandas as pd
import torch.utils.data as data
from src.data.preprocessing import (
    string_transform_labels,
    string_transform_onehot_char,
    smote_sampling,
    onehot_encode,
)
import src.data.utils as utils
from src.data.loader import DataLoader_folds

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# import all the settings variables for the models
from settings import *

kfold_obj = DataLoader_folds(data_path + celegans_seq, 3)


models = {
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=n_neighbors),
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
}


# K_Fold iteration loop
for name, model in models.items():
    print(name)
    auc_collect = []
    for fold, (train_idx, dev_idx) in enumerate(
        kfold_obj.kfold.split(kfold_obj.dataset)
    ):
        # creating sets
        train_x = kfold_obj.dataset["sequences"][train_idx]
        train_y = kfold_obj.dataset["labels"][train_idx]
        test_x = kfold_obj.dataset["sequences"][dev_idx]
        test_y = kfold_obj.dataset["labels"][dev_idx]

        # data preprocessing
        # train_x, test_x = string_transform_onehot_char(train_x, test_x)
        train_x, test_x = onehot_encode(train_x), onehot_encode(test_x)

        # sampling
        # train_x, train_y = smote_sampling(train_x, train_y)

        # model training & testing
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        predictions = pd.DataFrame(predictions).applymap(
            lambda x: 1 if (x >= 0) else -1
        )

        print("### FOLD {} ###".format(fold))
        auc_collect.append(utils.model_eval(predictions, test_y))

    print(
        "AUC mean: {0:.4f}+-{1:.4f}\n".format(np.mean(auc_collect), np.std(auc_collect))
    )
