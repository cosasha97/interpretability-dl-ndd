from tools.settings import *
from tools.data import *
import os

parser = argparse.ArgumentParser(description='Predictions generation')
parser.add_argument('--model_path', type=str, default=None,
                    help="""Path to configuration (folder). """)
parser.add_argument('--dataset', type=str, default='val',
                    help="Dataset to make predictions on.")
parser.add_argument('--model_folder', type=str, default='results/models/model_85',
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
means, stds = get_normalization_factors(training_df)
target_df = normalize_df(raw_target_df, means, stds)
target_df = pd.merge(target_df, subjects[['participant_id', 'session_id']], on=['participant_id', 'session_id'])


# get transformations
train_transforms, all_transforms = get_transforms('image', minmaxnormalization=True, data_augmentation=None)

# build data loader
data_loader = MRIDatasetImage(caps_path[study], target_df[['participant_id', 'session_id']], df_add_data=target_df,
                              preprocessing=args.preprocessing, all_transformations=all_transforms)



