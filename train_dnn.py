import numpy as np
import pandas as pd
import tensorflow as tf

from settings import data_path, celegans_seq
from src.data.preprocessing import (
    onehot_encode,
)

from src.models.spliceai import SpliceAI400

import src.data.utils as utils
from src.data.loader import DataLoader_training
from settings import *
from src.utils.utils import save_model


# loading and preprocessing training data
preprocess_transforms = [onehot_encode]
train = DataLoader_training(preprocess_X=preprocess_transforms)

models = {
    "SpliceAI400": SpliceAI400(),
}

# training all models and save them thereafte
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