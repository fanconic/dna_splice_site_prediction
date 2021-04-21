from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np

from src.data.utils import getKmers


def under_sample(X, y, sample_perc):
    """randomly undersample the majority class"""
    undersample = RandomUnderSampler(sampling_strategy=sample_perc, random_state=42)
    X_under, y_under = undersample.fit_resample(X, y)
    return X_under, y_under


def over_sample(X, y, sample_perc):
    """randomly oversample the minority class"""
    oversample = RandomOverSampler(sampling_strategy=sample_perc, random_state=42)
    X_over, y_over = oversample.fit_resample(X, y)
    return X_over, y_over


def onehot_encode(X):
    """Encode in One-Hot format
    Args:
        X (pandas.Series): input splice in its original format
    Returns:
        One hot encoded splice data in np.array format
    """
    f = lambda x: list(x)
    X = X.apply(f)
    X = pd.DataFrame(X.values.tolist(), index=X.index)
    enc = OneHotEncoder(sparse=False, categories = [['A', 'C', 'G', 'T'] for _ in range(X.shape[1])])
    enc.fit_transform(X)
    X_1h = enc.transform(X)
    return X_1h


def onehot_encode_kmers(X_train, X_test, kmers_size=1):
    """Encoding kmers of sequence to One-Hot format"""
    X_train = X_train.apply(getKmers)  # transforming sequences to kmers
    X_train = pd.DataFrame(X_train.values.tolist(), index=X_train.index)

    X_test = X_test.apply(getKmers)  # transforming sequences to kmers
    X_test = pd.DataFrame(X_test.values.tolist(), index=X_test.index)

    one_hot_enc = OneHotEncoder(handle_unknown="ignore")
    X_train = one_hot_enc.fit_transform(X_train).toarray()
    X_test = one_hot_enc.transform(X_test).toarray()
    return X_train, X_test


def over_under_sample(X, y, sample_strat_over, sample_strat_under):
    """doing a mixture of over- and under-sampling """

    # oversampling
    over = RandomOverSampler(sampling_strategy=sample_strat_over)
    X_over, y_over = over.fit_resample(X, y)
    # undersampling
    under = RandomUnderSampler(sampling_strategy=sample_strat_under)
    X_transformed, y_transformed = under.fit_resample(X_over, y_over)
    return X_transformed, y_transformed


def smote_sampling(X, y, sampling_strategy_perc=None):
    """ SMOTE, synthesizes new examples for the minority clas"""
    smote_sample = (
        SMOTE()
        if sampling_strategy_perc == None
        else SMOTE(sampling_strategy=sampling_strategy_perc, random_state=42)
    )
    X_transformed, y_transformed = smote_sample.fit_resample(X, y)
    return X_transformed, y_transformed


def string_transform_labels(train_ds, val_ds):
    """ Encoding the sequence characters into a DataFrame of integers using label encoder"""
    label_encoder = LabelEncoder()
    label_encoder.fit(["a", "c", "g", "u"])

    lis = []
    for idx in range(train_ds.shape[0]):
        tmp = [
            label_encoder.transform([i]).astype(float)[0]
            for i in train_ds.iloc[idx].lower()
        ]
        lis.append(tmp)
    X_train = pd.DataFrame(lis)

    lis = []
    for idx in range(val_ds.shape[0]):
        tmp = [
            label_encoder.transform([i]).astype(float)[0]
            for i in val_ds.iloc[idx].lower()
        ]
        lis.append(tmp)
    X_val = pd.DataFrame(lis)

    return X_train, X_val


def string_transform_hash(train_X, val_X):
    """ Encoding the sequence characters into a DataFrame of integers using a hash function"""
    tmp = dict()
    for i in range(train_X.shape[0]):
        kmers = getKmers(train_X.iloc[0])
        hashed_kmers = [hash_kmer(kmer) for kmer in kmers]
        tmp[i] = pd.Series(hashed_kmers)
    new_df_train = pd.DataFrame.from_dict(tmp, orient="index")

    tmp = dict()
    for i in range(val_X.shape[0]):
        kmers = getKmers(val_X.iloc[0])
        hashed_kmers = [hash_kmer(kmer) for kmer in kmers]
        tmp[i] = pd.Series(hashed_kmers)
    new_df_val = pd.DataFrame.from_dict(tmp, orient="index")

    return new_df_train, new_df_val


def string_transform_onehot_char(train_X, val_X):
    """Transforming the sequence characters into a DataFrame of one-hot encodings (position dependent)"""

    def split(x):
        return list(x[0])

    train_tmp = np.apply_along_axis(split, 1, pd.DataFrame(train_X))
    train_X = pd.get_dummies(pd.DataFrame(train_tmp))

    val_tmp = np.apply_along_axis(split, 1, pd.DataFrame(val_X))
    val_X = pd.get_dummies(pd.DataFrame(val_tmp))

    return train_X, val_X
