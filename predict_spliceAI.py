import numpy as np
import pandas as pd
import tensorflow as tf

from settings import data_path, celegans_seq
from src.data.preprocessing import (
    onehot_encode,
)

from src.models.spliceai import SpliceAI400, SpliceAI80

import src.data.utils as utils
from src.data.loader import DataLoader_sk, DataLoader_split
from settings import *
from src.utils.utils import save_model
from sklearn.utils import class_weight
from keras import backend as K
from sklearn.model_selection import train_test_split

# loading and preprocessing testing data
preprocess_transforms = [onehot_encode]
if data == "humans":
    loader = DataLoader_testing(
        csv_file=celegans_seq, preprocess_X=preprocess_transforms, flatten=False
    )
    test_x = loader.x.copy()

elif data == "celegans":
    loader = DataLoader_split(
        data_path + celegans_seq,
        preprocess_X=preprocess_transforms,
        flatten=False,
        augment=True,
    )
    test_x = loader.test_x.copy()

else:
    print("data not available. Only 'humans' or 'celegans' DNA sequences.")
    exit()

# models for testing (make sure the model name is exactly the same as in train.py)
models = {
    "SpliceAI80": (SpliceAI80(), (157, -159), (1, 82, 4)),
    "SpliceAI400": (SpliceAI400(), (0, 399), (1, 398, 4)),
}

for name, (model, (start, end), input_shape) in models.items():
    print("### loading model {} ###".format(name))
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        metrics=[tf.keras.metrics.AUC(curve="PR")],
    )

    x = tf.random.normal(input_shape)
    model(x)
    print(model.summary())

    model.load_weights(out_dir + name + "_" + data + ".hdf5")

    prediction_probas = model.predict(test_x[:, start:end, :])
    predictions = prediction_probas > 0.5

    # saving model predictions
    with open(results_dir + name + "_" + data + "_results.npy", "wb") as file:
        np.save(file, predictions)
    print("### predictions saved ###")

print("### process completed ###")
