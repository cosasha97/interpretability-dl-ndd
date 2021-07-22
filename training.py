import torch
import torch.optim as optim
import os
import argparse
import re
import logging
import json

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
from tools.logger import *

# parser
parser = argparse.ArgumentParser(description='Train 4-branch CNN')
parser.add_argument('--name', type=str, default=None,
                    help="""Name of the job. In case it is the same name as an an already existing job:
                    - erase and restard from scratch if resume_training is False
                    - resume training if resume_training is True
                    """)
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
parser.add_argument('--monitor', type=str, default='train',
                    help='metric used to monitor progress during training')
parser.add_argument('-lw', '--loss_weights', nargs='+', type=float, default=[1., 1., 1., 1.],
                    help='weights to assign to each branch loss')
parser.add_argument('--patience', type=int, default=10,
                    help='patience of Early Stopping')
parser.add_argument("--resume_training", action='store_true', default=False,
                    help="Load pretrained model and resume training.")
parser.add_argument('--config_path', type=str, default=None,
                    help="""Path to configuration. """)


args = parser.parse_args()

# load existing config
if args.config_path is not None:
    with open(args.config_path, "r") as f:
        json_data = json.load(f)
    for key in json_data:
        setattr(args, key, json_data[key])

# paths
caps_directory = '/network/lustre/dtlake01/aramis/datasets/adni/caps/caps_v2021/'
if args.name is None:
    # fetch the highest ID already used
    ids = [re.search('\d+', file) for file in os.listdir(args.output_dir)]
    ids = [int(match.group()) for match in ids if match is not None]
    if len(ids) == 0:
        model_number = 0
    else:
        model_number = max(ids) + 1
    args.name = 'model_' + str(model_number)
args.output_dir = os.path.join(args.output_dir, args.name)
# configure logger
stdout_logger = config_logger(args.output_dir)

# resume training ?
if args.resume_training:
    # RESUME TRAINING
    with open(os.path.join(args.output_dir, 'commandline.json'), "r") as f:
        json_data = json.load(f)
    for key in json_data:
        setattr(args, key, json_data[key])

    # load data split
    training_df = pd.read_csv(os.path.join(args.output_dir, 'training_df.csv'))
    valid_df = pd.read_csv(os.path.join(args.output_dir, 'valid_df.csv'))

else:
    # NEW TRAINING
    # save commandline
    commandline_to_json(args, logger=stdout_logger)

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

print(args)
print("Beginning of the script - TRAINING")

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

# callbacks
ES = EarlyStopping(patience=args.patience)
MC = ModelCheckpoint()

# record losses
train_metrics = dict()
val_metrics = dict()

if args.resume_training:
    # resume training
    print("###### Resume training ######")
    last_checkpoint = torch.load('results/models/{}/last_model.pt'.format(args.name))
    first_epoch = last_checkpoint['epoch'] + 1
    print('starting epoch: %d' % first_epoch)
    MC.best = last_checkpoint['loss']
    if 'train_metrics' in last_checkpoint.keys():
        train_metrics = last_checkpoint['train_metrics']
    if 'val_metrics' in last_checkpoint.keys():
        val_metrics = last_checkpoint['val_metrics']
    model.load_state_dict(last_checkpoint['model_state_dict'])
    optimizer.load_state_dict(last_checkpoint['optimizer_state_dict'])

else:
    # start a new training
    print("###### Start training ######")
    # first epoch
    first_epoch = 0

# training
for epoch in range(first_epoch, args.nb_epochs):
    update_dict(train_metrics,
                train(epoch, model, optimizer, train_loader, loss_weights=args.loss_weights, to_cuda=True))
    update_dict(val_metrics, test(model, valid_loader, loss_weights=args.loss_weights, to_cuda=True))
    if ES.step(train_metrics[args.monitor][-1]):
        break
    MC.step(train_metrics[args.monitor][-1],
            epoch,
            model,
            optimizer,
            train_metrics,
            val_metrics,
            args.output_dir)


# save training curves
def save_loss(loss, name="loss"):
    df = pd.DataFrame.from_dict(loss)
    df[df.any(axis=1)].to_csv(name + '.csv', index=False)


save_loss(train_metrics, os.path.join(args.output_dir, 'train_metrics'))
save_loss(val_metrics, os.path.join(args.output_dir, 'val_metrics'))
# gradients
# save_loss({'gradient_norms': model.gradient_norms}, os.path.join(args.output_dir, 'gradients'))
