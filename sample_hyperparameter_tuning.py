import numpy as np
import pandas as pd
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
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


from scipy.stats import randint
import src.data.preprocessing
from src.data.loader import DataLoader_sk
from src.data.utils import model_eval


# import all settings variables for the models
from settings import *

random_state = 42

# preprocessing
train_data = DataLoader_sk(data_path + hum_seq_train)
train_data.x = onehot_encode(train_data.x)
train_data.x, train_data.y = smote_sampling(train_data.x, train_data.y)


# define your model here
model = RandomForestClassifier(class_weight="balanced", random_state=random_state)

"""
# random search for hyperparameter tuning
random_search_boost = RandomizedSearchCV(model,
                                     {
                                         'n_estimators': randint(low=1000, high=2000),
                                         'max_features': randint(low=1, high=30),
                                     },
                                     scoring='average_precision',
                                     cv=3,
                                     n_iter = 3, # how many candidates are sampled
                                     refit=True, # retrain on whole data once the best parameters are chosen
                                     verbose=5, # get some output, as it may take a long time
                                     n_jobs=-1,
                                     random_state=random_state)
"""

# grid search for hyperparameter tuning
grid_search_boost = GridSearchCV(
    model,
    param_grid={
        "n_estimators": [1500],  # [1500, 1600, 1700, 1800, 1900],
        "max_features": [10, 20, 30],
    },
    scoring="average_precision",
    cv=3,
    refit=True,  # retrain on whole data once the best parameters are chosen
    verbose=5,  # get some output, as it may take a long time
    n_jobs=-1,
)

clf = grid_search_boost.fit(train_data.x, train_data.y)

print("Best score: {}, best parameters: {}".format(clf.best_score_, clf.best_params_))

best_model = clf.best_estimator_


# validation with best performing model
print("### Testing best model on validation set ###")

val_data = DataLoader_sk(data_path + hum_seq_val)
val_data.x = onehot_encode(val_data.x)

predictions = best_model.predict(val_data.x)
model_eval(predictions, val_data.y)
