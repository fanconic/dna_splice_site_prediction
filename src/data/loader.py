import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from settings import *


class DataLoader_sk:
    """DataLoader for sklearn"""

    def __init__(self, csv_file, shuffle=True, preprocess_X=None, flatten=True):
        """Initialize dataloader
        Args:
            csv_file: path of the csv_file
            shuffle (default True): Shuffle dataset
            preprocess_X: list of preprocessing to be applyed on the data
            flatten (default True): whether transformation during preprocessing should be flattened
        """
        self.dataset = pd.read_csv(csv_file)
        self.shuffle = shuffle
        if self.shuffle:
            self.dataset = self.dataset.sample(frac=1)

        self.x = self.dataset["sequences"]

        if "C_elegans_acc_seq" in csv_file:
            # sequences derived from C.elegans, so T instead of U
            self.x = self.x.apply(lambda x: x.replace("U", "T"))

        if preprocess_X is not None:
            print("preprocessing data...")
            for transform in preprocess_X:
                self.x = transform(self.x, flatten=flatten)

        self.y = self.dataset["labels"]
        self.y = self.y.replace(-1, 0).values


class DataLoader_folds:
    """DataLoader for n_folds (cross-validation)"""

    def __init__(
        self, csv_file, numFolds, shuffle=True, preprocess_X=None, flatten=True
    ):
        """Initialize a stratified K fold dataloader
        Args:
            csv_file: path of the csv_file
            numFolds: number of folds for cross validation
            shuflle (default True): Shuffle dataset
            preprocess_X: list of preprocessing to be applyed on the data
        """
        self.dataset = pd.read_csv(csv_file)

        self.x = self.dataset["sequences"]
        if preprocess_X is not None:
            print("preprocessing data...")
            for transform in preprocess_X:
                self.x = transform(self.x, flatten=flatten)
        self.y = self.dataset["labels"]
        self.y = self.y.replace(-1, 0).values

        self.shuffle = shuffle
        self.numFolds = numFolds
        self.kfold = StratifiedKFold(
            n_splits=self.numFolds, shuffle=self.shuffle, random_state=seed
        )


class DataLoader_split:
    """DataLoader to generate training and test split"""

    def __init__(
        self,
        csv_file,
        test_size=0.2,
        doStratify=True,
        random_state=seed,
        preprocess_X=None,
        flatten=True,
        augment=False,
        save_test_df=False,
    ):
        """Initialize dataloader for generating stratified train & test split
        Args:
            csv_file: path of the csv_file
            test_size: fraction of dataset to be in testing set
            doStratify (default True): Whether the split should be stratified or not
            random_state: fixing seed for split
            preprocess_X: list of preprocessing to be applyed on the data
            flatten (default True): whether transformation during preprocessing should be flattened
            augment (default False): whether to augment the sequence size or not
            save_test_df (default False): whether to save the test dataset or not
        """
        self.dataset = pd.read_csv(csv_file)
        self.x = self.dataset["sequences"]
        self.y = self.dataset["labels"]
        self.y = self.y.replace(-1, 0).values

        if "C_elegans_acc_seq" in csv_file and augment:
            self.x = "N" * 157 + self.x + "N" * 159

        self.test_size = test_size

        if preprocess_X is not None:
            print("preprocessing data...")
            for transform in preprocess_X:
                self.x = transform(self.x, flatten=flatten)

        if doStratify:
            self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
                self.x, self.y, test_size=test_size, stratify=self.y, random_state=seed
            )

        else:
            self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
                self.x, self.y, test_size=test_size, stratify=None, random_state=seed
            )

        if save_test_df:
            test_df = pd.DataFrame()
            test_df["sequences"] = self.test_x
            test_df["labels"] = self.test_y
            test_df["labels"] = test_df["labels"].replace(0, -1)
            test_df.to_csv("./exercise_data/C_elegans_test_seq.csv")


class DataLoader_training:
    """DataLoader for final training with best hyperparameters"""

    def __init__(
        self,
        csv_file_1=hum_seq_train,
        csv_file_2=hum_seq_val,
        shuffle=True,
        preprocess_X=None,
        flatten=True,
    ):
        """Initialize dataloader for final training with best hyperparameters (see report)
        Args:
            csv_file_1 (default hum_seq_train): path of the first csv file
            csv_file_2 (default hum_seq_val): path of the second csv file
            shuflle (default True): Shuffle dataset
            test_size: fraction of dataset to be in testing set
            preprocess_X: list of preprocessing to be applyed on the data
            flatten (default True): whether transformation during preprocessing should be flattened
        """
        self.dataset_1 = pd.read_csv(data_path + csv_file_1)
        self.dataset_2 = pd.read_csv(data_path + csv_file_2)

        self.dataset = self.dataset_1.append(self.dataset_2, ignore_index=True)

        self.shuffle = shuffle
        if self.shuffle:
            self.dataset = self.dataset.sample(frac=1)

        self.x = self.dataset["sequences"]
        if preprocess_X is not None:
            print("preprocessing data...")
            for transform in preprocess_X:
                self.x = transform(self.x, flatten=flatten)

        self.y = self.dataset["labels"]
        self.y = self.y.replace(-1, 0).values


class DataLoader_testing:
    """DataLoader for final testing with trained models"""

    def __init__(self, csv_file=hum_seq_hidden, preprocess_X=None, flatten=True):
                """Initialize dataloader for final training with best hyperparameters (see report)
        Args:
            csv_file (default hum_seq_hidden): path of the testing csv file
            preprocess_X: list of preprocessing to be applyed on the data
            flatten (default True): whether transformation during preprocessing should be flattened
        """
        self.dataset = pd.read_csv(data_path + csv_file)

        self.x = self.dataset["sequences"]
        if preprocess_X is not None:
            print("preprocessing data...")
            for transform in preprocess_X:
                self.x = transform(self.x, flatten=flatten)
