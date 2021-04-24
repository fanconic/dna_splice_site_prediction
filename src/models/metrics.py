import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

class AUPRC(tf.keras.callbacks.Callback):
    def __init__(self, val_data, batch_size = 128):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size
        
    def on_train_begin(self, logs={}):
        self._data = [] 

    def on_epoch_end(self, epoch, logs={}):
        batches = len(self.validation_data)
        total = batches * self.batch_size

        xVal, yVal = self.validation_data
        yPred = self.model.predict(xVal, verbose= 0)
        #yPred = yPred > 0.5
            
        yPred = np.squeeze(yPred)
        precision, recall, thresholds = precision_recall_curve(yVal, yPred)
        auprc_score = auc(recall, precision)
        
        print('val auprc: ', auprc_score)
        self._data.append({'val_auprc': auprc_score})
        return