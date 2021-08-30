import torch
import torchvision
import numpy as np
from nilearn import plotting
import clinicadl
import pandas as pd
import matplotlib.pyplot as plt
from torchinfo import summary
from sklearn.metrics import log_loss
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from math import floor
import random
import time
import os
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


pipeline_name='t1-volume'
atlas_id='AAL2'

parser = argparse.ArgumentParser(description='Scores generation')
parser.add_argument('--target', type=str, default='age',
                    help="""Target. """)
args = parser.parse_args()

atlas_tsv, atlas_map = load_atlas(custom_tsv=True)

path = 'results/models/model_85/attribution_maps/GC/test/'
target = args.target
print(target)
output_path = path + target
raw_scores, normalized_scores = compute_scores(output_path, atlas_tsv, atlas_map)
raw_scores.to_csv(os.path.join(output_path, 'raw_scores.csv'), index=False)
normalized_scores.to_csv(os.path.join(output_path, 'normalized_scores.csv'), index=False)
