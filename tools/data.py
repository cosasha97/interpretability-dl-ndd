import torch
import numpy as np
import pandas as pd
from os import path

# clinicaDL
from clinicadl.tools.deep_learning.data import generate_sampler, return_dataset, MRIDataset, MRIDatasetImage, \
    MRIDatasetSlice, get_transforms


def fetch_add_data(training_data, pipeline_name='t1-volume', atlas_id='AAL2'):
    """
    Fetch additional data: age, sex and volumes.
    Normalize scalar data.

    Args:
        training_data: DataFrame, with (at least) the following keys: participant_id, session_id, diagnosis, age, sex
        pipeline_name: string
        atlas_id: string, name of the atlas used to determine brain volumes

    Return:
        - stds: pandas.core.series.Series. Stds used to normalized scalar features.
        - df_add_data: DataFrame containing the normalized additional data
    """
    # paths
    data_path = '/network/lustre/dtlake01/aramis/datasets/adni/caps/caps_v2021.tsv'
    summary_path = '/network/lustre/dtlake01/aramis/datasets/adni/caps/caps_v2021_summary.tsv'

    # fetch indexes
    df_summary = pd.read_csv(summary_path, sep='\t')
    df_summary = df_summary[(df_summary.pipeline_name == pipeline_name) & (df_summary.atlas_id == atlas_id)]
    first_column_name = df_summary.first_column_name.item()
    last_column_name = df_summary.last_column_name.item()
    print('First column name: ', first_column_name)
    print('Last column name: ', last_column_name)
    df_data = pd.read_csv(data_path, sep='\t', nrows=1)
    first_column_index = df_data.columns.get_loc(first_column_name)
    last_column_index = df_data.columns.get_loc(last_column_name)

    # other data to fetch
    col_names = ['participant_id', 'session_id', 'sex', 'age']
    add_indexes = [df_data.columns.get_loc(col_name) for col_name in col_names]

    # compute df_add_data
    # add 1 to first_column_index to ignore background
    used_columns = np.hstack([add_indexes, np.arange(first_column_index + 1, last_column_index + 1)]).flatten()
    df_add_data = pd.read_csv(data_path, sep='\t', usecols=used_columns).dropna(axis=0, how='any')
    print(df_add_data.head())

    # normalization using only statistics from training data
    temp_df = pd.merge(training_data[['participant_id', 'session_id']],
                       df_add_data, on=['participant_id', 'session_id'], how='left')
    scalar_cols = temp_df.columns.difference(['participant_id', 'session_id', 'sex'])
    # df_add_data[scalar_cols] contains only scalar columns with (patient, session) from training set
    means, stds = temp_df[scalar_cols].mean(), temp_df[scalar_cols].std()
    df_add_data[scalar_cols] = (df_add_data[scalar_cols] - means) / stds

    return stds, df_add_data


class MRIDatasetSlice(MRIDataset):

    def __init__(self, caps_directory, data_file, slice_index=None, preprocessing="t1-volume", # t1-linear
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
                 preprocessing='t1-volume', train_transformations=None, # 't1-linear'
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

        ## fetch additional data
        temp_df = self.df_add_data[(self.df_add_data.participant_id == participant) &
                                   (self.df_add_data.session_id == session)]
        sex = (temp_df.sex.to_numpy().item() == 'F') + 0.
        age = temp_df.age.to_numpy().item()
        volumes = temp_df.drop(columns=['participant_id', 'session_id', 'sex', 'age']).to_numpy().squeeze()

        sample = {'image': image, 'label': label, 'participant_id': participant, 'session_id': session,
                  'image_path': image_path, 'age': age, 'sex': sex, 'volumes': volumes}

        return sample

    def num_elem_per_image(self):
        return 1
