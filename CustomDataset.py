import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch import nn
from torch.optim import Adam

class FashionITIS(Dataset):
  def __init__(self, csv_file, transform= None):
    self.data = pd.read_csv(csv_file)
    self.transform = transform

    self.labels = self.data.iloc[:,0].values
    self.images = self.data.iloc[:,1:].values

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    flat_image = self.images[idx]
    image = flat_image.reshape(28,28).astype(np.uint8)

    from PIL import Image
    image = Image.fromarray(image)

    label= self.labels[idx]

    if self.transform:
       image = self.transform(image)

    return image,label
