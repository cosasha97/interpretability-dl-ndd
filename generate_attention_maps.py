"""
Script to generate attention maps for subjects in test set.
One attention map is generated for age, sex, AD/CN, as well as for each volume (119 overall).

Train & test sets come from ADNI.
Validation set comes from AIBL.
"""

import torch
import torch.optim as optim
import os
import argparse
import re
import logging
import json
import random
from tqdm import tqdm

# clinicaDL
from clinicadl.tools.deep_learning.data import generate_sampler, return_dataset, MRIDataset, get_transforms
from torch.utils.data import DataLoader
from clinicadl.tools.deep_learning.iotools import commandline_to_json

# own imports
from tools.models.CN5_FC3_3D import *
from tools.callbacks import *
from tools.data import *
from tools.logger import *
from tools.settings import *
from tools.explanations.GradCam import *

# debug
import pdb

parser = argparse.ArgumentParser(description='Attribution maps generation')
parser.add_argument('--model_folder', type=str, default=None,
                    help="""Path to configuration (folder). """)
parser.add_argument('--dataset', type=str, default='val',
                    help="Dataset used to generate attribution maps.")
parser.add_argument('--method', type=str, default='GC',
                    help="Method used to generate attribution maps.")
parser.add_argument('--debug', action='store_true', default=False,
                    help="Launch debug model (use a small size dataset).")

args = parser.parse_args()

# load training subjects
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
model_folder = args.model_folder

# output path
att_maps_path = os.path.join(model_folder, 'attribution_maps', args.method, args.dataset)
if not args.debug:
    # configure logger
    stdout_logger = config_logger(att_maps_path)
print('Saving maps to {}'.format(att_maps_path))

# load existing config
if args.model_folder is not None:
    with open(os.path.join(args.model_folder, 'commandline.json'), "r") as f:
        json_data = json.load(f)
    for key in json_data:
        if key != 'debug':
            setattr(args, key, json_data[key])
else:
    raise Exception("config_path is empty!")

# build  structure
os.makedirs(att_maps_path, exist_ok=True)
for key in TARGET2BRANCH.keys():
    os.makedirs(os.path.join(att_maps_path, key), exist_ok=True)
for index in range(N_VOLUMES):
    os.makedirs(os.path.join(att_maps_path, 'volumes', str(index)), exist_ok=True)

# get transformations
train_transforms, all_transforms = get_transforms('image', minmaxnormalization=True, data_augmentation=None)

# build target dataframe (useless in practice)
# stds, df_add_data = fetch_add_data(training_df)
raw_target_df = compute_target_df(study=study)
raw_target_df = pd.merge(raw_target_df, subjects[['participant_id', 'session_id']], on=['participant_id', 'session_id'])
raw_target_df.to_csv(os.path.join(output_path, 'raw_target_df.csv'), index=False)
means, stds = get_normalization_factors(training_df)
target_df = normalize_df(raw_target_df, means, stds)

# build data loader
# if args.dataset == 'train':
#     data_loader = MRIDatasetImage(caps_directory, training_df, df_add_data=df_add_data,
#                                   preprocessing=args.preprocessing, all_transformations=all_transforms)
# elif args.dataset == 'val':
#     data_loader = MRIDatasetImage(caps_directory, valid_df, df_add_data=df_add_data, preprocessing=args.preprocessing,
#                                   all_transformations=all_transforms)
# # elif args.dataset == 'test':
# #     data_loader =
# else:
#     raise Exception('No dataset.')
data_loader = MRIDatasetImage(caps_directory[study],
                              target_df[['participant_id', 'session_id', 'diagnosis']],
                              df_add_data=target_df,
                              preprocessing=args.preprocessing,
                              all_transformations=all_transforms)

# get sample
sample = data_loader[0]
# build model
model = Net(sample, args.convolutions, args.dropout, args.save_gradient_norm).cuda()
# load pretrained weights on validation set
saved_data = torch.load(os.path.join(args.model_folder, 'test_best_model.pt'))
model.load_state_dict(saved_data['model_state_dict'])

# initialize interpretability method
GC = GradCam(model)

for idx, data in tqdm(enumerate(data_loader)):
    # fetch participant and session ids
    participant_id = data['participant_id']
    session_id = data['session_id']
    # build name
    name = '_'.join((participant_id, session_id))
    # compute attribution maps
    for target in TARGET2BRANCH.keys():
        if target == 'volumes':
            for volume_index in range(N_VOLUMES):
                att_map = GC.generate_cam(data['image'],
                                          branch=TARGET2BRANCH[target],
                                          resize=False,
                                          to_cpu=True,
                                          volume_index=volume_index)
                np.save(os.path.join(att_maps_path, target, str(volume_index), name),
                        att_map)
                if args.debug:
                    # compute only one volume
                    break
        else:
            att_map = GC.generate_cam(data['image'],
                                      branch=TARGET2BRANCH[target],
                                      resize=False,
                                      to_cpu=True)
            np.save(os.path.join(att_maps_path, target, name), att_map)

    if args.debug:
        # process only one sample
        break
