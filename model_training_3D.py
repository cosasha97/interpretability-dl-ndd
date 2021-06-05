import torch.optim as optim
import os
import argparse
import re

# clinicaDL
from clinicadl.tools.tsv.data_split import create_split
from clinicadl.tools.deep_learning.data import generate_sampler, return_dataset, MRIDataset, get_transforms
from torch.utils.data import DataLoader
from clinicadl.tools.deep_learning import EarlyStopping
from clinicadl.tools.deep_learning.iotools import commandline_to_json

# own imports
from tools.models.CN5_FC3_3D import *
from tools.callbacks import *
from tools.data import *
from train.train_CNN import *

# parser
parser = argparse.ArgumentParser(description='Train 4-branch CNN')
parser.add_argument('--name', type=str, default=None,
                    help='Name of the job')
parser.add_argument('--output_dir', type=str, default='results/models',
                    help='path to store training results (model, optimizer, losses, metrics)')
parser.add_argument('-d', '--dropout', type=float, default=0.2,
                    help='rate of dropout that will be applied to dropout layers in CNN.')
parser.add_argument('-bs', '--batch_size', type=int, default=4,
                    help='size of batches used during training/evaluation')
parser.add_argument('-e', '--nb_epochs', type=int, default=30,
                    help='number of epochs during training')
parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                    help='path to store training results (model, optimizer, losses, metrics)')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4,
                    help='weight decay')

args = parser.parse_args()

print("Beginning of the script - TRAINING")

# paths
caps_directory = '/network/lustre/dtlake01/aramis/datasets/adni/caps/caps_v2021/'
if args.name is None:
    model_number = max([int(re.search('\d+', file).group()) for file in os.listdir(args.output_dir)])
    args.name = 'model_' + str(model_number)
output_path = os.path.join(args.output_dir, args.name)
# save commandline
commandline_to_json(args)

# load dataframes
AD = pd.read_csv('subjects/AD.tsv', sep='\t')
CN = pd.read_csv('subjects/CN.tsv', sep='\t')

# remove samples with NaN
AD.drop(AD[AD.isna().sum(axis=1) > 0].index, inplace=True)
CN.drop(CN[CN.isna().sum(axis=1) > 0].index, inplace=True)

# split data between training and validation sets
training_df, valid_df = create_split('AD', AD, 'diagnosis', 0.2)
df_CN = create_split('CN', CN, 'diagnosis', 0.2)
training_df = training_df.append(df_CN[0]).reset_index()  # .iloc[np.array([0,1,2,-1,-2,-3])]
valid_df = valid_df.append(df_CN[1]).reset_index()  # .iloc[np.array([0,1,2,-1,-2,-3])]

# drop index column
training_df.drop(columns=['index'], inplace=True)
valid_df.drop(columns=['index'], inplace=True)

train_transforms, all_transforms = get_transforms('image', minmaxnormalization=True, data_augmentation=None)
# fetch volumetric data
df_add_data = fetch_add_data(training_df)

# all_transforms = torchvision.transforms.Compose([])

data_train = MRIDatasetImage(caps_directory, training_df, df_add_data=df_add_data,
                             all_transformations=all_transforms)  # train_transformations=all_transforms
data_valid = MRIDatasetImage(caps_directory, valid_df, df_add_data=df_add_data,
                             all_transformations=all_transforms)  # train_transformations=all_transforms,

# sampler
train_sampler = generate_sampler(data_train)
valid_sampler = generate_sampler(data_valid)
# loaders
train_loader = DataLoader(data_train,
                          batch_size=args.batch_size,
                          sampler=train_sampler,
                          num_workers=args.num_workers,
                          pin_memory=True)

valid_loader = DataLoader(data_valid,
                          batch_size=args.batch_size,
                          sampler=valid_sampler,
                          num_workers=args.num_workers,
                          pin_memory=True)

# get sample
sample = data_train[0]
# build model
model = Net(sample, [8, 16, 32, 64, 128], args.dropout)
if torch.cuda.is_available():
    print("To cuda")
    model.cuda()
model.summary(batch_size=args.batch_size)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
# record losses
train_losses = dict()
test_losses = dict()

# callbacks
ES = EarlyStopping(patience=5)
MC = ModelCheckpoint()

print("Beginning of the training")

# training
for epoch in range(args.nb_epochs):
    update_dict(train_losses, train(epoch, model, optimizer, train_loader, to_cuda=True))
    update_dict(test_losses, test(model, valid_loader, to_cuda=True))
    if ES.step(train_losses['train'][epoch]):
        break
    MC.step(train_losses['train'][epoch], epoch, model, optimizer, path)  # path


# save training curves
def save_loss(loss, name="loss"):
    df = pd.DataFrame.from_dict(loss)
    df[df.any(axis=1)].to_csv(name + '.csv', index=False)


save_loss(train_losses, os.path.join(args.output_dir, 'train_losses'))
save_loss(test_losses, os.path.join(args.output_dir, 'val_losses'))
