class DataLoader_torch(data.Dataset):
  
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