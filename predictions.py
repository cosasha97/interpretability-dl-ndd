import os
import argparse
import json
import torch
import logging
import numpy as np
import pandas as pd

from clinicadl.tools.deep_learning.data import generate_sampler, return_dataset, MRIDataset, get_transforms
from torch.utils.data import DataLoader
from clinicadl.tools.deep_learning.iotools import commandline_to_json

# own imports
from tools.models.CN5_FC3_3D import *
from tools.data import *
from tools.logger import *
from tools.settings import *
from train.train_CNN import *

# debug
import pdb

parser = argparse.ArgumentParser(description='Predictions generation')
parser.add_argument('--model_folder', type=str, default='results/models/model_85',
                    help="""Path to configuration (folder). """)
parser.add_argument('--dataset', type=str, default='test',
                    help="Dataset to make predictions on.")
parser.add_argument('--debug', action='store_true', default=False,
                    help="Launch debug model (use a small size dataset).")

args = parser.parse_args()

# load existing config
if args.model_folder is not None:
    with open(os.path.join(args.model_folder, 'commandline.json'), "r") as f:
        json_data = json.load(f)
    for key in json_data:
        if key != 'debug':
            setattr(args, key, json_data[key])
else:
    raise Exception("config_path is empty!")

if not hasattr(args, 'convolutions'):
    setattr(args, 'convolutions', [8, 16, 32, 64, 128])
if not hasattr(args, 'preprocessing'):
    setattr(args, 'preprocessing', 't1-linear')
if not hasattr(args, 'save_gradient_norm'):
    setattr(args, 'save_gradient_norm', False)

# fix seed
set_seed(args.seed)

# build  structure
output_path = os.path.join(args.model_folder, 'predictions', args.dataset)
os.makedirs(output_path, exist_ok=True)

# load training dataframe
training_df = pd.read_csv(os.path.join(args.model_folder, 'training_df.csv'))

# determine study
if args.dataset == 'train':
    study = 'adni'
    subjects = training_df
elif args.dataset == 'val':
    study = 'adni'
    subjects = pd.read_csv(os.path.join(args.model_folder, 'valid_df.csv'))
elif args.dataset == 'test':
    study = 'aibl'
    subjects = get_subjects('aibl')
else:
    raise Exception("Unknown dataset!")

# build target dataframe
raw_target_df = compute_target_df(study=study)
raw_target_df = pd.merge(raw_target_df, subjects[['participant_id', 'session_id']], on=['participant_id', 'session_id'])
raw_target_df.to_csv(os.path.join(output_path, 'raw_target_df.csv'), index=False)
means, stds = get_normalization_factors(training_df)
target_df = normalize_df(raw_target_df, means, stds)

if args.debug:
    target_df = target_df.iloc[:10]

# get transformations
train_transforms, all_transforms = get_transforms('image', minmaxnormalization=True, data_augmentation=None)

# build data loader
data_processor = MRIDatasetImage(caps_directory[study],
                                 target_df[['participant_id', 'session_id', 'diagnosis']],
                                 df_add_data=target_df,
                                 preprocessing=args.preprocessing,
                                 all_transformations=all_transforms)

data_loader = DataLoader(data_processor,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=args.num_workers,
                         pin_memory=True)

# get sample
sample = data_processor[0]
# build model
model = Net(sample, args.convolutions, args.dropout, args.save_gradient_norm).cuda()
# load pretrained weights on validation set
saved_data = torch.load(os.path.join(args.model_folder, 'test_best_model.pt'))
model.load_state_dict(saved_data['model_state_dict'])

if next(model.parameters()).is_cuda:
    # summary automatically moves the model to GPU !!!
    # avoid the call if model must be on CPU
    model.summary(batch_size=args.batch_size)
print('Is model on GPU?', next(model.parameters()).is_cuda)

# test model on dataset
predictions, losses = test(model,
                           data_loader,
                           loss_weights=args.loss_weights,
                           to_cuda=not args.cpu,
                           rescaling=pd.Series(stds, index=get_scalar_columns(target_df)),
                           eval_mode=args.eval_mode,
                           save_predictions=True)

predictions['volumes'] = stds[1:].reshape((1, -1)) * predictions['volumes'] + means[1:].reshape((1, -1))
predictions['age'] = stds[0].reshape((1, 1)) * predictions['age'] + means[0].reshape((1, 1))

for key in predictions.keys():
    # saving predictions
    np.save(os.path.join(output_path, key+'.npy'), predictions[key])

# saving losses
losses_file = open(os.path.join(output_path, 'losses.json'), "w")
json.dump(losses, losses_file)
losses_file.close()
