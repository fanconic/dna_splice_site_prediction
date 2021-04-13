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
from sklearn.svm import SVC
import lightgbm

# import all the settings variables for the models
from settings import *


kfold_obj = DataLoader_folds(data_path + celegans_seq, n_folds)


models = {
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=n_neighbors),
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "Gradient Boosting": lightgbm.LGBMClassifier(n_estimators=100, num_leaves=20),
}


# K_Fold iteration loop
for name, model in models.items():
    print(name)
    auc_collect = []
    for fold, (train_idx, dev_idx) in enumerate(
        kfold_obj.kfold.split(kfold_obj.x, kfold_obj.y)
    ):
        # creating sets
        train_x = kfold_obj.x[train_idx]
        train_y = kfold_obj.y[train_idx]
        test_x = kfold_obj.x[dev_idx]
        test_y = kfold_obj.y[dev_idx]

        # data preprocessing
        train_x, test_x = onehot_encode(train_x), onehot_encode(test_x)
        # train_x, test_x = onehot_encode_kmers(train_x, test_x) #note to us: kmer size = 3 could be beneficial for kNN; otherwise one-hot better

        # sampling
        train_x, train_y = under_sample(train_x, train_y, 1)

        train_x, train_y = smote_sampling(train_x, train_y)

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
