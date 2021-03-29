
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

import src.data.utils

def under_sample(X, y, sample_perc):
  undersample = RandomUnderSampler(sampling_strategy=sample_perc)
  X_under, y_under = undersample.fit_resample(X, y)
  return X_under, y_under


def over_sample(X, y, sample_perc):
  oversample = RandomOverSampler(sampling_strategy=sample_perc)
  X_over, y_over = oversample.fit_resample(X, y)
  return X_over, y_over


def over_under_sample(X, y, sample_strat_over, sample_strat_under):
  # oversampling
  over = RandomOverSampler(sampling_strategy=sample_strat_over)
  X_over, y_over = over.fit_resample(X, y)
  # undersampling
  under = RandomUnderSampler(sampling_strategy=sample_strat_under)
  X_transformed, y_transformed = under.fit_resample(X_over, y_over)
  return X_transformed, y_transformed


def smote_sampling(X, y, sampling_strategy_perc=None):
  smote_sample = SMOTE() if sampling_strategy_perc==None else SMOTE(sampling_strategy=sampling_strategy_perc)
  X_transformed,y_transformed = smote_sample.fit_resample(X, y)
  return X_transformed, y_transformed



def string_transform_labels(train_ds, val_ds):
  label_encoder = LabelEncoder()
  label_encoder.fit(['a','c','g','u'])

  lis = []
  for idx in range(train_ds.shape[0]):
    tmp = [label_encoder.transform([i]).astype(float)[0] for i in train_ds.iloc[idx].lower()]
    lis.append(tmp)
  X_train = pd.DataFrame(lis)

  lis = []
  for idx in range(val_ds.shape[0]):
    tmp = [label_encoder.transform([i]).astype(float)[0] for i in val_ds.iloc[idx].lower()]
    lis.append(tmp)
  X_val = pd.DataFrame(lis)

  return X_train, X_val


def string_transform_hash(train_X, val_X):
  tmp = dict()
  for i in range(train_X.shape[0]):
    kmers = getKmers(train_X.iloc[0])
    hashed_kmers = [hash_kmer(kmer) for kmer in kmers]
    tmp[i] = pd.Series(hashed_kmers)
  new_df_train = pd.DataFrame.from_dict(tmp, orient='index')

  tmp = dict()
  for i in range(val_X.shape[0]):
    kmers = getKmers(val_X.iloc[0])
    hashed_kmers = [hash_kmer(kmer) for kmer in kmers]
    tmp[i] = pd.Series(hashed_kmers)
  new_df_val = pd.DataFrame.from_dict(tmp, orient='index')

  return new_df_train, new_df_val


def random_split(data_obj):
  X_train, X_test, y_train, y_test = train_test_split(data_obj.x, data_obj.y, test_size=0.2, random_state=42)
  return X_train, X_test, y_train, y_test