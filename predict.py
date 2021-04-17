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
from src.data.loader import DataLoader_testing



# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from src.utils.utils import load_model

from settings import *

# models for testing (make sure the model name is exactly the same as in train.py)
models = [
    # "K-Nearest Neighbours",
    "Logistic Regression",
    #"Support Vector Machine",
    #"Gradient Boosting",
    # "MLP": MLPClassifier(),
    "Random Forest",
]

# loading and preprocessing testing data
preprocess_transforms = [onehot_encode]
test = DataLoader_testing(csv_file = celegans_seq, preprocess_X = preprocess_transforms)

for name in models:
	print("### loading model {} ###".format(name))
	model = load_model(name)

	predictions = model.predict(test.x)

	# saving model predictions
	with open(out_dir + name + "_results.npy", "wb") as file:
		np.save(file, predictions)
	print("### predictions saved ###")

print("### process completed ###")