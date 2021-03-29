import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
from sklearn.ensemble import RandomForestClassifier

from settings import (
	data_path,
	celegans_seq
	)
from src.data.preprocessing import (
	string_transform_labels,
	smote_sampling,
	random_split
	)

import src.data.utils as utils

from src.models.models import k_NN

from src.data.loader import DataLoader_sk


# TODO implement pipelines
# TODO scores
# TODO model configs in settings
# TODO comments & readme



# loading data

data_obj = DataLoader_sk(data_path+celegans_seq)
train_x, test_x, train_y, test_y = random_split(data_obj)

# transforming sequences to numerical values

train_x, test_x = string_transform_labels(train_x, test_x)


train_x, train_y = smote_sampling(train_x, train_y)

# defining model & training

k_nn = k_NN()

k_nn.clf.fit(train_x, train_y)

# testing & eval model

predictions = k_nn.clf.predict(test_x)

utils.model_eval(predictions, test_y)