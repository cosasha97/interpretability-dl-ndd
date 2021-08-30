import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import floor
import random
import time
import os
os.chdir("/network/lustre/dtlake01/aramis/users/sasha.collin/interpretability-dl-ndd")
import sys
sys.path.append("/network/lustre/dtlake01/aramis/users/sasha.collin/interpretability-dl-ndd")
import json
import re

# visualization
from scipy.ndimage import zoom
from tools.explanations.analysis.utils import *
from tools.settings import *
import nibabel as nib
from tools.data import *
import ast
import collections
import argparse
import multiprocessing

parser = argparse.ArgumentParser(description='Dice-scores generation')
parser.add_argument('--debug', action='store_true', default=False,
                    help="Launch debug model (use a small size dataset).")
parser.add_argument('--dataset', type=str, default='val', help="""Dataset. """)
parser.add_argument('--model_folder', type=str, default='results/models/model_85/')
args = parser.parse_args()

atlas_tsv, atlas_map = load_atlas(custom_tsv=True)


def DC_folder(data_path):
    files = [ file for file in os.listdir(data_path) if '.npy' in file ]
    volume_index = int(data_path.strip('/').split('/')[-1])
    print('Volume index:', volume_index)
    DC_scores = {}
    for k, file in enumerate(files):
        participant_id, session_id = file.strip('.npy').split('_')
        prediction = np.load(os.path.join(data_path, file))
        dc = dice_score_computation(prediction, volume_index, atlas_tsv, atlas_map, resize=True)
        DC_scores[file] = [participant_id, session_id] + list(dc)
        if args.debug and (k == 1):
            break
    cols = ['participant_id', 'session_id', 'DC', 'cDC']
    df= pd.DataFrame.from_dict(DC_scores, orient='index', columns=cols)
    df = df.reset_index()[cols]
    df.to_csv(os.path.join(data_path, 'dice_scores.csv'), index=False)


folders_path = os.path.join(args.model_folder, 'attribution_maps', 'GC', args.dataset, 'volumes')
folders = [ os.path.join(folders_path, folder) for folder in os.listdir(folders_path) ]

pool = multiprocessing.Pool(os.cpu_count())
_ = zip(*pool.map(DC_folder, folders))
