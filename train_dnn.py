import numpy as np
import pandas as pd
import tensorflow as tf

from settings import data_path, celegans_seq
from src.data.preprocessing import (
    onehot_encode,
)

from src.models.spliceai import SpliceAI400

import src.data.utils as utils
from src.data.loader import DataLoader_training, DataLoader_sk
from settings import *
from src.utils.utils import save_model
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_curve, auc
from src.models.metrics import AUPRC

# loading and preprocessing training data
preprocess_transforms = [onehot_encode]
train = DataLoader_sk(data_path + hum_seq_train, preprocess_X=preprocess_transforms)
val = DataLoader_sk(data_path + hum_seq_val, preprocess_X=preprocess_transforms)

models = {
    "SpliceAI400": SpliceAI400(),
}


def scheduler(epoch, lr):
    if epoch < 6:
        return lr
    else:
        return lr * 1 / 2


epochs = 10
batch_size = 256
class_weights = class_weight.compute_class_weight(
    "balanced", np.unique(train.y), train.y
)
class_weights = {i: class_weights[i] for i in range(2)}
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
auprc_callback = AUPRC((val.x, val.y), batch_size = batch_size)


# training all models and save them thereafte
for name, model in models.items():
    print("### fitting model {} ###".format(name))

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        metrics=[tf.keras.metrics.AUC(curve="PR"), tf.keras.metrics.AUC(curve="ROC")],
    )

    input_shape = (1, 398, 4)
    x = tf.random.normal(input_shape)
    model(x)
    print(model.summary())

    model.fit(
        x=train.x,
        y=train.y,
        validation_data=(val.x, val.y),
        class_weight=class_weights,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=[callback, auprc_callback],
    )

    print("### saving trained model {} ###".format(name))
    model.save(out_dir + name)

    prediction_probas = model.predict(val.x).reshape(-1)
    predictions = prediction_probas > 0.5
    print("### performance on valdiation set ###")
    utils.model_eval(predictions, val.y, prediction_probas)
    results = model.evaluate(val.x, val.y, batch_size=256, verbose=2)
    print("val loss, val auprc, val auroc:", results)

    if predictionOnTestingSet:
        # evaluating performance on given testing set
        test = DataLoader_sk(
            data_path + hum_seq_test, shuffle=False, preprocess_X=preprocess_transforms
        )
        predictions = model.predict(test.x)
        predictions = predictions > 0.5
        print("### performance on testing set ###")
        utils.model_eval(predictions.reshape(-1), test.y)
        results = model.evaluate(test.x, test.y, batch_size=256, verbose=2)
        print("val loss, val auprc, val auroc:", results)

print("### training completed ###")
