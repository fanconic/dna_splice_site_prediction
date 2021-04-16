import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from settings import *


class DataLoader_sk:
    """DataLoader for sklearn"""

    def __init__(self, csv_file, shuffle=True, preprocess_X=None):

        self.dataset = pd.read_csv(csv_file)
        self.shuffle = shuffle
        if self.shuffle:
            self.dataset = self.dataset.sample(frac=1)

        self.x = self.dataset["sequences"]
        if preprocess_X is not None:
            print("preprocessing data...")
            for transform in preprocess_X:
                self.x = transform(self.x)
        
        if "C_elegans_acc_seq" in csv_file:
            # sequences derived from C.elegans, so T instead of U
            self.x = self.x.apply(lambda x: x.replace("T", "U"))
        self.y = self.dataset["labels"]


class DataLoader_folds:
    """DataLoader for n_folds (cross-validation)"""

    def __init__(self, csv_file, numFolds, shuffle=True, preprocess_X=None):
        """Initialize a stratified K fold dataloader
        Args:
            csv_file: path of the csv_file
            numFolds: number of folds for cross validation
            shuflle (default True): Shuffle dataset
            preprocess_X: list of preprocessing to be applyed on the data
        """
        self.dataset = pd.read_csv(csv_file)
        if "C_elegans_acc_seq" in csv_file:
            # sequences derived from C.elegans, so T instead of U
            self.dataset["sequences"] = self.dataset["sequences"].apply(
                lambda x: x.replace("T", "U")
            )

        self.x = self.dataset["sequences"]
        if preprocess_X is not None:
            print("preprocessing data...")
            for transform in preprocess_X:
                self.x = transform(self.x)
        self.y = self.dataset["labels"]

        self.shuffle = shuffle
        self.numFolds = numFolds
        self.kfold = StratifiedKFold(
            n_splits=numFolds, shuffle=self.shuffle, random_state=42
        )

class DataLoader_training:
    """DataLoader for final training with best hyperparameters"""

    def __init__(self, csv_file_1 = hum_seq_train, csv_file_2 = hum_seq_val, shuffle=True, preprocess_X=None):

        self.dataset_1 = pd.read_csv(data_path+csv_file_1)
        self.dataset_2 = pd.read_csv(data_path+csv_file_2)

        self.dataset = self.dataset_1.append(self.dataset_2, ignore_index = True)

        self.shuffle = shuffle
        if(self.shuffle):
            self.dataset = self.dataset.sample(frac=1)

        self.x = self.dataset["sequences"]
        if preprocess_X is not None:
            print("preprocessing data...")
            for transform in preprocess_X:
                self.x = transform(self.x)

        self.y = self.dataset["labels"]


class DataLoader_testing:
    """DataLoader for final testing with trained models"""

    def __init__(self, csv_file = hum_seq_hidden, preprocess_X=None):

        self.dataset = pd.read_csv(data_path+csv_file)

        self.x = self.dataset["sequences"]
        if preprocess_X is not None:
            print("preprocessing data...")
            for transform in preprocess_X:
                self.x = transform(self.x)


