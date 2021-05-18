import torch
import torchvision
import numpy as np
from nilearn import plotting
import clinicadl
import pandas as pd
import matplotlib.pyplot as plt
from torchsummary import summary
from sklearn.metrics import log_loss
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from math import floor
import pickle

# torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# clinicaDL
from clinicadl.tools.tsv.data_split import create_split
from clinicadl.tools.deep_learning.data import generate_sampler, return_dataset, MRIDataset, MRIDatasetImage, MRIDatasetSlice, get_transforms
from torch.utils.data import DataLoader
from clinicadl.tools.deep_learning.cnn_utils import train, get_criterion, test
from clinicadl.tools.deep_learning.models.random import RandomArchitecture
from clinicadl.tools.deep_learning import EarlyStopping

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# own imports
from tools.models.CN5_FC3 import *
from tools.callbacks import *
from tools.data import *
from train.train_CNN import *

# global parameters
caps_directory = '/network/lustre/dtlake01/aramis/datasets/adni/caps/caps_v2021/'
batch_size = 512
num_workers = 1
loss = 'default'
optimizer_name = 'Adam'
learning_rate = 1e-4
weight_decay = 1e-4

# fold iterator
fold_iterator = range(2)

# load dataframes
AD = pd.read_csv('subjects/AD.tsv',sep='\t')
CN = pd.read_csv('subjects/CN.tsv',sep='\t')

# remove samples with NaN
AD.drop(AD[AD.isna().sum(axis=1) > 0].index, inplace=True)
CN.drop(CN[CN.isna().sum(axis=1) > 0].index, inplace=True)

# split data between training and validation sets
training_df, valid_df = create_split('AD', AD, 'diagnosis',0.2)
df_CN = create_split('CN', CN, 'diagnosis',0.2)
training_df = training_df.append(df_CN[0]).reset_index()#.iloc[np.array([0,1,2,-1,-2,-3])]
valid_df = valid_df.append(df_CN[1]).reset_index()#.iloc[np.array([0,1,2,-1,-2,-3])]

# drop index column
training_df.drop(columns = ['index'], inplace=True)
valid_df.drop(columns = ['index'], inplace=True)

train_transforms, all_transforms = get_transforms('image',
                                                  minmaxnormalization=False,
                                                  data_augmentation=None )
# fetch volumetric data
df_add_data = fetch_add_data(training_df)

# all_transforms = torchvision.transforms.Compose([])

# follow structure of ``train_single_cnn``

# training_df['slice_id'] = 85
# valid_df['slice_id'] = 85
# dataset iterator
# data_train = MRIDatasetSlice(caps_directory, training_df, slice_index=85, mixed=True, df_add_data=df_add_data) #train_transformations=all_transforms
# data_valid = MRIDatasetSlice(caps_directory, valid_df, slice_index=85, mixed=True, df_add_data=df_add_data) #train_transformations=all_transforms,

data_train = MRIDatasetSlice(caps_directory, training_df, df_add_data=df_add_data) #train_transformations=all_transforms
data_valid = MRIDatasetSlice(caps_directory, valid_df, df_add_data=df_add_data) #train_transformations=all_transforms,

# sampler
train_sampler = generate_sampler(data_train)
valid_sampler = generate_sampler(data_valid)
# loaders
train_loader = DataLoader(data_train,
                         batch_size=batch_size,
                         sampler=train_sampler,
                         num_workers=num_workers,
                         pin_memory=True)

valid_loader = DataLoader(data_valid,
                         batch_size=batch_size,
                         sampler=valid_sampler,
                         num_workers=num_workers,
                         pin_memory=True)

# get sample
sample = data_train[0]
# build model
model = Net(sample, [8, 16, 32, 64, 128])
if torch.cuda.is_available():
    print("To cuda")
    model.cuda()
model.summary()

# define number of epochs
nb_epochs = 20
# optimizer
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
# device
cuda = torch.device('cuda')
# record losses
train_losses = {
    'classification': np.zeros(nb_epochs),
    'volumes': np.zeros(nb_epochs),
    'age': np.zeros(nb_epochs),
    'sex': np.zeros(nb_epochs),
    'train': np.zeros(nb_epochs)
}
test_losses = {
    'classification': np.zeros(nb_epochs),
    'volumes': np.zeros(nb_epochs),
    'age': np.zeros(nb_epochs),
    'sex': np.zeros(nb_epochs),
    'test': np.zeros(nb_epochs)
}

# callbacks
ES = EarlyStopping(patience=5)
MC = ModelCheckpoint()

# training
for epoch in range(nb_epochs):
    update_dict(train_losses, train(epoch, model, optimizer, cuda, train_loader), epoch)
    update_dict(test_losses, test(model, cuda, valid_loader), epoch)
    if ES.step(train_losses['train'][epoch]):
        break
    MC.step(train_losses['train'][epoch], epoch, model, optimizer) # path

# save training curves
f = open("train_losses.pkl","wb")
pickle.dump(train_losses,f)
f.close()

f = open("val_losses.pkl","wb")
pickle.dump(test_losses,f)
f.close()