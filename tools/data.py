import pdb

import torch
import numpy as np
import pandas as pd
import os
from os import path
import re
from tools.settings import *
from tools.explanations.analysis.utils import *

# clinicaDL
from clinicadl.tools.deep_learning.data import generate_sampler, return_dataset, MRIDataset, MRIDatasetImage, \
    MRIDatasetSlice, get_transforms


# data generation
def get_subjects(study='aibl'):
    """
    Fetch subjects for a given study.
    :param study: string, AIBL or ADNI
    :return: pandas DataFrame
    """
    AIBL_CN = pd.read_csv('subjects/{}/CN.tsv'.format(study.upper()), sep='\t')
    AIBL_AD = pd.read_csv('subjects/{}/AD.tsv'.format(study.upper()), sep='\t')
    return AIBL_CN.append(AIBL_AD)


def compute_target_df(study='adni', pipeline_name='t1-volume', atlas_id='AAL2', nrows=None, verbose=0):
    """
    Compute dataframe containing all the raw targets.

    :param study: string, ADNI or AIBL
    :param pipeline_name: string
    :param atlas_id: string
    :param nrows: int, number of loaded rows
    :param verbose: int.
    :return: pandas DataFrame
    """
    # paths
    data_path = caps_path[study.lower()]
    summary_path = caps_summary_path[study.lower()]

    # fetch indexes
    df_summary = pd.read_csv(summary_path, sep='\t')
    df_summary = df_summary[(df_summary.pipeline_name == pipeline_name) & (df_summary.atlas_id == atlas_id)]
    df_summary['group_name'] = df_summary.group_id.apply(lambda x: re.findall('(?:adni|aibl)', x.lower())[0])
    df_summary = df_summary[(df_summary.group_name == study)]
    first_column_name = df_summary.first_column_name.item()
    last_column_name = df_summary.last_column_name.item()
    if verbose > 0:
        print('First column name: ', first_column_name)
        print('Last column name: ', last_column_name)
    df_data = pd.read_csv(data_path, sep='\t', nrows=1)
    first_column_index = df_data.columns.get_loc(first_column_name)
    last_column_index = df_data.columns.get_loc(last_column_name)

    # other data to fetch
    col_names = ['participant_id', 'session_id', 'sex', 'diagnosis', 'age']
    add_indexes = [df_data.columns.get_loc(col_name) for col_name in col_names]

    # compute dataframe
    # add 1 to first_column_index to ignore background volume
    used_columns = np.hstack([add_indexes, np.arange(first_column_index + 1, last_column_index + 1)]).flatten()
    raw_data = pd.read_csv(data_path, sep='\t', usecols=used_columns, nrows=nrows).dropna(axis=0, how='any')
    raw_data = raw_data.reset_index().drop(columns=['index'])
    if verbose > 0:
        print(raw_data.columns)

    return raw_data


def normalize_df(raw_dataframe, norm_means, norm_stds):
    """
    Normalize raw_dataframe scalar columns using norm_means and norm_stds

    :param raw_dataframe: pandas dataframe
    :param norm_means: array containing normalization means
    :param norm_stds: array containing normalization stds
    :return: normalized pandas DataFrame
    """
    scalar_cols = get_scalar_columns(raw_dataframe)
    raw_dataframe[scalar_cols] = (raw_dataframe[scalar_cols] - norm_means.reshape((1, -1))) / norm_stds.reshape((1, -1))
    return raw_dataframe


def get_scalar_columns(df):
    """
    Fetch names of scalar columns.

    :param df: pandas dataframe
    :return: list of strings
    """
    return [col for col in df.columns if col not in ['participant_id', 'session_id', 'diagnosis', 'sex']]


def get_normalization_factors(training_data, pipeline_name='t1-volume', atlas_id='AAL2', study='adni'):
    """
    Compute normalization factors for scalar features using training data.

    :param training_data: dataframe
    :param pipeline_name: string
    :param atlas_id: string
    :param study: string
    :return:
        - means: array
        - stds: array
    """
    # compute raw (unnormalized) dataframe
    df_add_data = compute_target_df(study, pipeline_name, atlas_id)

    # normalization using only statistics from training data
    temp_df = pd.merge(training_data[['participant_id', 'session_id']],
                       df_add_data, on=['participant_id', 'session_id'], how='left')
    # scalar_cols = temp_df.columns.difference(['participant_id', 'session_id', 'sex'])
    scalar_cols = get_scalar_columns(temp_df)
    # df_add_data[scalar_cols] contains only scalar columns with (patient, session) from training set
    means, stds = temp_df[scalar_cols].mean(), temp_df[scalar_cols].std()
    return means.to_numpy(), stds.to_numpy()


def fetch_add_data(training_data, pipeline_name='t1-volume', atlas_id='AAL2', study='adni'):
    """
    Fetch additional data: age, sex and volumes.
    Normalize scalar data.

    Args:
        training_data: DataFrame, with (at least) the following keys: participant_id, session_id, diagnosis, age, sex
        pipeline_name: string
        atlas_id: string, name of the atlas used to determine brain volumes
        study: string, adni or aibl

    Return:
        - stds: pandas.core.series.Series. Stds used to normalized scalar features.
        - df_add_data: DataFrame containing the normalized additional data
    """

    # compute raw (unnormalized) dataframe
    df_add_data = compute_target_df(study, pipeline_name, atlas_id)

    # normalization using only statistics from training data
    temp_df = pd.merge(training_data[['participant_id', 'session_id']],
                       df_add_data, on=['participant_id', 'session_id'], how='left')
    # scalar_cols = temp_df.columns.difference(['participant_id', 'session_id', 'sex'])
    scalar_cols = get_scalar_columns(temp_df)
    # df_add_data[scalar_cols] contains only scalar columns with (patient, session) from training set
    means, stds = temp_df[scalar_cols].mean(), temp_df[scalar_cols].std()
    df_add_data[scalar_cols] = (df_add_data[scalar_cols] - means) / stds
    print('stds keys: ', stds.keys())

    return stds, df_add_data


class MRIDatasetSlice(MRIDataset):

    def __init__(self, caps_directory, data_file, slice_index=None, preprocessing="t1-volume",  # t1-linear
                 train_transformations=None, mri_plane=0, prepare_dl=False,
                 discarded_slices=20, mixed=False, labels=True, all_transformations=None,
                 multi_cohort=False,
                 df_add_data=None):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string or DataFrame): Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing (string): Defines the path to the data in CAPS.
            slice_index (int, optional): If a value is given the same slice will be extracted for each image.
                else the dataset will load all the slices possible for one image.
            train_transformations (callable, optional): Optional transform to be applied only on training mode.
            prepare_dl (bool): If true pre-extracted patches will be loaded.
            mri_plane (int): Defines which mri plane is used for slice extraction.
            discarded_slices (int or list): number of slices discarded at the beginning and the end of the image.
                If one single value is given, the same amount is discarded at the beginning and at the end.
            mixed (bool): If True will look for a 'slice_id' column in the input DataFrame to load each slice
                independently.
            labels (bool): If True the diagnosis will be extracted from the given DataFrame.
            all_transformations (callable, options): Optional transform to be applied during training and evaluation.
            multi_cohort (bool): If True caps_directory is the path to a TSV file linking cohort names and paths.
            df_add_data (DataFrame): dataframe containing additional data to predict, such as volumes
        """
        # additional data
        self.df_add_data = df_add_data

        # Rename MRI plane
        if preprocessing == "shepplogan":
            raise ValueError("Slice mode is not available for preprocessing %s" % preprocessing)
        self.elem_index = slice_index
        self.mri_plane = mri_plane
        self.direction_list = ['sag', 'cor', 'axi']
        if self.mri_plane >= len(self.direction_list):
            raise ValueError(
                "mri_plane value %i > %i" %
                (self.mri_plane, len(
                    self.direction_list)))

        # Manage discarded_slices
        if isinstance(discarded_slices, int):
            discarded_slices = [discarded_slices, discarded_slices]
        if isinstance(discarded_slices, list) and len(discarded_slices) == 1:
            discarded_slices = discarded_slices * 2
        self.discarded_slices = discarded_slices

        if mixed:
            self.elem_index = "mixed"
        else:
            self.elem_index = None

        self.mode = "slice"
        self.prepare_dl = prepare_dl
        super().__init__(caps_directory, data_file, preprocessing,
                         augmentation_transformations=train_transformations, labels=labels,
                         transformations=all_transformations, multi_cohort=multi_cohort)

    def __getitem__(self, idx):
        participant, session, cohort, slice_idx, label = self._get_meta_data(idx)
        slice_idx = slice_idx + self.discarded_slices[0]

        if self.prepare_dl:
            # read the slices directly
            slice_path = path.join(self._get_path(participant, session, cohort, "slice")[0:-7]
                                   + '_axis-%s' % self.direction_list[self.mri_plane]
                                   + '_channel-rgb_slice-%i_T1w.pt' % slice_idx)
            image = torch.load(slice_path)
        else:
            image_path = self._get_path(participant, session, cohort, "image")
            full_image = torch.load(image_path)
            image = self.extract_slice_from_mri(full_image, slice_idx)

        if self.transformations:
            image = self.transformations(image)

        if self.augmentation_transformations and not self.eval_mode:
            image = self.augmentation_transformations(image)

        ## fetch additional data
        temp_df = self.df_add_data[(self.df_add_data.participant_id == participant) &
                                   (self.df_add_data.session_id == session)]
        sex = (temp_df.sex.to_numpy().item() == 'F') + 0.
        age = temp_df.age.to_numpy().item()
        volumes = temp_df.drop(columns=['participant_id', 'session_id', 'sex', 'age']).to_numpy().squeeze()

        sample = {'image': image, 'label': label,
                  'participant_id': participant, 'session_id': session,
                  'slice_id': slice_idx, 'age': age, 'sex': sex, 'volumes': volumes}

        return sample

    def num_elem_per_image(self):
        if self.elem_index is not None:
            return 1

        image = self._get_full_image()
        return image.size(self.mri_plane + 1) - \
               self.discarded_slices[0] - self.discarded_slices[1]

    def extract_slice_from_mri(self, image, index_slice):
        """
        This is a function to grab one slice in each view and create a rgb image for transferring learning: duplicate the slices into R, G, B channel
        :param image: (tensor)
        :param index_slice: (int) index of the wanted slice
        :return:
        To note, for each view:
        Axial_view = "[:, :, slice_i]"
        Coronal_view = "[:, slice_i, :]"
        Sagittal_view= "[slice_i, :, :]"
        """
        image = image.squeeze(0)
        simple_slice = image[(slice(None),) * self.mri_plane + (index_slice,)]
        triple_slice = torch.stack((simple_slice, simple_slice, simple_slice))

        return triple_slice


class MRIDatasetImage(MRIDataset):
    """Dataset of MRI organized in a CAPS folder."""

    def __init__(self, caps_directory, data_file,
                 preprocessing='t1-volume', train_transformations=None,  # 't1-linear'
                 labels=True, all_transformations=None, multi_cohort=False,
                 df_add_data=None):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string or DataFrame): Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing (string): Defines the path to the data in CAPS.
            train_transformations (callable, optional): Optional transform to be applied only on training mode.
            labels (bool): If True the diagnosis will be extracted from the given DataFrame.
            all_transformations (callable, options): Optional transform to be applied during training and evaluation.
            multi_cohort (bool): If True caps_directory is the path to a TSV file linking cohort names and paths.
            df_add_data (DataFrame): dataframe containing additional data to predict, such as volumes
        """
        self.df_add_data = df_add_data

        self.elem_index = None
        self.mode = "image"
        super().__init__(caps_directory, data_file, preprocessing,
                         augmentation_transformations=train_transformations, labels=labels,
                         transformations=all_transformations, multi_cohort=multi_cohort)

    def __getitem__(self, idx):
        participant, session, cohort, _, label = self._get_meta_data(idx)

        image_path = self._get_path(participant, session, cohort, "image")
        image = torch.load(image_path)

        if self.transformations:
            image = self.transformations(image)

        if self.augmentation_transformations and not self.eval_mode:
            image = self.augmentation_transformations(image)

        # fetch additional data
        temp_df = self.df_add_data[(self.df_add_data.participant_id == participant) &
                                   (self.df_add_data.session_id == session)]
        sex = (temp_df.sex.to_numpy().item() == 'F') + 0.
        age = temp_df.age.to_numpy().item()
        volumes = temp_df.drop(columns=['participant_id', 'session_id', 'sex', 'age', 'diagnosis']).to_numpy().squeeze()

        sample = {'image': image, 'label': label, 'participant_id': participant, 'session_id': session,
                  'image_path': image_path, 'age': age, 'sex': sex, 'volumes': volumes}

        return sample

    def num_elem_per_image(self):
        return 1


# data analysis
def fetch_results(dataset, model_folder='results/models/model_85/'):
    """
    Fetch results (predictions) as well as targets, format targets,
    and merge everything in a single dataframe.

    Params:
        - dataset: string ('val', 'test')
        - model_folder: string, path to folder of the model used to generate predictions
    Returns:
        - pandas DataFrame
    """
    prediction_path = os.path.join(model_folder, 'predictions/')
    # load target dataframe
    raw_target_df = pd.read_csv(os.path.join(prediction_path, dataset, 'raw_target_df.csv'))
    # format targets
    raw_target_df.diagnosis = (raw_target_df.diagnosis == 'AD').astype(int)
    raw_target_df.sex = (raw_target_df.sex == 'F').astype(int)
    # load predictions
    ID = np.load(os.path.join(prediction_path, dataset, 'id.npy'))
    disease_preds = np.load(os.path.join(prediction_path, dataset, 'disease.npy'))
    age_preds = np.load(os.path.join(prediction_path, dataset, 'age.npy'))
    sex_preds = np.load(os.path.join(prediction_path, dataset, 'sex.npy'))
    # store predictions in dataframe
    df_preds = pd.DataFrame({'participant_id': ID[:, 0],
                             'session_id': ID[:, 1],
                             'diagnosis': disease_preds.flatten(),
                             'age': age_preds.flatten(),
                             'sex': sex_preds.flatten()})
    # merge dataframes
    cols = ['participant_id', 'session_id', 'diagnosis', 'age', 'sex']
    data = raw_target_df[cols].merge(df_preds, on=['participant_id', 'session_id'], suffixes=('_true', '_pred'))

    return data


def classification(target, data, q=25):
    """
    Determine TP, TN, FP, FN for diagnosis and sex targets.
    For age, determine the 25% best (+1) and worst predictions (-1).
    Params:
        - target: string
        - data: pandas dataframe containing prediction target values
        - q: int, percentile (only useful for age)
    """
    if target in ['diagnosis', 'sex']:
        y_true = data[target + '_true']
        y_pred = (data[target + '_pred'] > 0.5).astype(int)
        data[target + '_class'] = np.nan
        data[target + '_class'][(y_true == y_pred) & (y_true == 1)] = 'TP'
        data[target + '_class'][(y_true == y_pred) & (y_true == 0)] = 'TN'
        data[target + '_class'][(y_true != y_pred) & (y_pred == 1)] = 'FP'
        data[target + '_class'][(y_true != y_pred) & (y_pred == 0)] = 'FN'
    elif target == 'age':
        age_delta = (data.age_true - data.age_pred).abs().to_numpy()
        q1 = np.percentile(age_delta, q=q)
        q3 = np.percentile(age_delta, q=100-q)
        data['age_class'] = 0
        data['age_class'][age_delta < q1] = 1
        data['age_class'][age_delta > q3] = -1
    else:
        raise Exception("Target unknown!")


def compute_classification_df(dataset, model_folder='results/models/model_85/', save=True):
    """
    Compute classification for all preds (diagnosis, age, sex).

    Params:
        - dataset: string ('val', 'test')
        - model_folder: string, path to folder of the model used to generate predictions
    Returns:
        - pandas DataFrame
    """
    data = fetch_results(dataset, model_folder)
    for target in ['sex', 'diagnosis', 'age']:
        classification(target, data)
    df_classification = data[['participant_id', 'session_id', 'diagnosis_class', 'sex_class', 'age_class']]
    if save:
        df_classification.to_csv(os.path.join(model_folder, 'predictions', dataset, 'df_classification.csv'),
                                 index=False)
    return df_classification


def results_analysis(dataset, model_folder='results/models/model_85/'):
    """
    Compute global analysis for age, sex and diagnosis.
    For classification task (i.e. sex and diagnosis), compute MIR for each of the following
    groups: 'TP', 'TN', 'FP', 'FN'.
    For regression task (i.e. age), compute MIR for each of the following groups: +1, -1.
    Save pandas DataFrame in the subfolder of model_folder 'results/'.
    """
    # load atlas
    atlas_tsv, _ = load_atlas(custom_tsv=True)
    # load classification
    df_classification = pd.read_csv(os.path.join(model_folder, 'predictions', dataset, 'df_classification.csv'))
    for target in ['disease', 'sex', 'age']:
        scores_path = os.path.join(model_folder, 'attribution_maps/GC', dataset, target, 'normalized_scores.csv')
        normalized_scores = pd.read_csv(scores_path)
        # merge df_classification with normalized_scores
        merge = normalized_scores.merge(df_classification, on=['participant_id', 'session_id'])
        d = {}
        # define classes
        if target in ['disease', 'sex']:
            classes = ['TP', 'TN', 'FP', 'FN']
        else:
            # age
            classes = [1, -1]
        # rename target
        if target == 'disease':
            target = 'diagnosis'
        # compute class analysis
        cols_class = ['diagnosis_class', 'sex_class', 'age_class']
        for class_ in classes:
            d[class_] = MIR(merge[merge[target + '_class'] == class_].drop(columns=cols_class), atlas_tsv, concat_df=True)
        # save analysis
        output_path = os.path.join(model_folder, 'analysis', dataset, 'summary_MIR_{}.csv'.format(target))
        final_df = pd.concat(d.values(), axis=1, keys=d.keys())
        final_df.to_csv(output_path, index=False)

