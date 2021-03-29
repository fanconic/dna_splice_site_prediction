import torch.utils.data as data
import pandas as pd
from sklearn.model_selection import KFold

class DataLoader_torch(data.Dataset):
  ''' DataLoader for pytorch

  '''
  def __init__(self, csv_file, transform=None, shuffle=False, downsample=False, upsample=False):
    self.dataset = pd.read_csv(csv_file)
    self.transform = transform
    self.shuffle = shuffle
    self.downsample = downsample
    self.upsample = upsample


    if self.shuffle:
      self.dataset = self.dataset.sample(frac=1)

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    sample = self.dataset.iloc[idx]
    if self.transform:
      sample = self.transform(sample)

    return sample


class DataLoader_sk():
  ''' DataLodader for sklearn
  
  '''

  def __init__(self, csv_file, shuffle=True):

    self.dataset = pd.read_csv(csv_file)
    self.shuffle = shuffle
    if self.shuffle:
      self.dataset = self.dataset.sample(frac=1)

    self.x = self.dataset['sequences']
    if('C_elegans_acc_seq' in csv_file):
      # sequences derived from C.elegans, so T instead of U
      self.x = self.x.apply(lambda x: x.replace('T','U'))
    self.y = self.dataset['labels']


class DataLoader_folds():
  ''' DataLoader for n_folds (cross-validation)

  '''

  def __init__(self, csv_file, numFolds, shuffle=True):
    self.dataset = pd.read_csv(csv_file)
    if('C_elegans_acc_seq' in csv_file):
      # sequences derived from C.elegans, so T instead of U
      self.dataset['sequences'] = self.dataset['sequences'].apply(lambda x: x.replace('T','U'))


    self.shuffle = shuffle
    self.numFolds = numFolds
    self.kfold = KFold(n_splits=numFolds, shuffle=self.shuffle, random_state=42)
