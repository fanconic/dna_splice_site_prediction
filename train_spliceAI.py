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

np.random.seed(seed)
tf.random.set_seed(seed)

preprocess_transforms = [onehot_encode]

if data == "humans":
    # loading and preprocessing training data
    train_loader = DataLoader_sk(
        data_path + hum_seq_train, preprocess_X=preprocess_transforms, flatten=False
    )
    val_loader = DataLoader_sk(
        data_path + hum_seq_val, preprocess_X=preprocess_transforms, flatten=False
    )
    test_loader = DataLoader_sk(
        data_path + hum_seq_test,
        shuffle=False,
        preprocess_X=preprocess_transforms,
        flatten=False,
    )
    X_train, y_train = train_loader.x, train_loader.y
    X_val, y_val = val_loader.x, val_loader.y
    X_test, y_test = test_loader.x, test_loader.y

elif data == "celegans":
    loader = DataLoader_split(
        data_path + celegans_seq,
        preprocess_X=preprocess_transforms,
        flatten=False,
        augment=True,
    )
    X_train, y_train = loader.train_x, loader.train_y
    X_test, y_test = loader.test_x, loader.test_y
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, random_state=seed, stratify=y_train
    )

else:
    print("data not available. Only 'humans' or 'celegans' DNA sequences.")
    exit()


models = {
    "SpliceAI80": (SpliceAI80(), (157, -159), (1, 82, 4)),
    "SpliceAI400": (SpliceAI400(), (0, 399), (1, 398, 4)),
}


def scheduler(epoch, lr):
    if epoch < 6:
        return lr
    else:
        return lr * 1 / 2


class_weights = class_weight.compute_class_weight(
    "balanced", np.unique(y_train), y_train
)
class_weights = {i: class_weights[i] for i in range(2)}


# training all models and save them thereafte
for name, (model, (start, end), input_shape) in models.items():

    # Callbacks
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(scheduler),
        AUPRC((X_val[:, start:end, :], y_val), batch_size=batch_size),
        tf.keras.callbacks.ModelCheckpoint(
            out_dir + name + "_" + data + ".hdf5",
            save_best_only=True,
            monitor="val_auc",
            mode="max",
        ),
    ]

    print("### fitting model {} ###".format(name))

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        metrics=[tf.keras.metrics.AUC(curve="PR")],
    )

    x = tf.random.normal(input_shape)
    model(x)
    print(model.summary())

    model.fit(
        x=X_train[:, start:end, :],
        y=y_train,
        validation_data=(X_val[:, start:end, :], y_val),
        class_weight=class_weights,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=[callbacks],
    )

    prediction_probas = model.predict(X_val[:, start:end, :])
    predictions = prediction_probas > 0.5
    print("### performance on valdiation set ###")
    utils.model_eval(predictions.reshape(-1), y_val, prediction_probas.reshape(-1))

    if predictionOnTestingSet:
        # evaluating performance on given testing set

        prediction_probas = model.predict(X_test[:, start:end, :])
        predictions = prediction_probas > 0.5
        print("### performance on testing set ###")
        utils.model_eval(predictions.reshape(-1), y_test, prediction_probas.reshape(-1))

    K.clear_session()
print("### training completed ###")
