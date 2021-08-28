from scipy.ndimage import zoom
import nibabel as nib
import pandas as pd
import numpy as np
import re
from tools.settings import *
from tools.data import *
import ast
import collections


def load_atlas():
    """
    Load atlas (map & tsv).

    Returns:
        - atlas_tsv: pandas DataFrame with roi_name and roi_value columns
        - atlas_map: array
    """
    # code source de clinica clinica/resources/atlases
    # par exemple atlas-AAL2_dseg.nii.gz
    # atlas_path = '../clinica/clinica/resources/atlases/atlas-AAL2_dseg.nii.gz'

    # load atlas
    atlas_tsv = pd.read_csv(
        '/network/lustre/dtlake01/aramis/users/sasha.collin/clinica/clinica/resources/atlases/atlas-AAL2_dseg.tsv',
        sep='\t')
    atlas_path = 'atlas/atlas-AAL2_space-MNI152NLin2009cSym_dseg.nii.gz'
    atlas = nib.load(atlas_path)
    atlas_map = atlas.get_fdata()

    return atlas_tsv, atlas_map


def main_region_name(name):
    """
    Return main region name of area.

    Params:
        - name: string

    Return:
        - string
    """
    return re.findall(r"[a-zA-Z\_]+", name.split("_R")[0].split("_L")[0])[0].strip('_')


def update_atlas_tsv(atlas_tsv, study='adni'):
    """
    Update atlas_tsv (describing chosen atlas):
    - add main regions (e.g. Vermis replacing ['Vermis_1', ..., 'Vermis_10'])
    - format roi_name
    - apply various sanity checks and corrections

    Params:
        - atlas_tsv: pandas DataFrame describing the atlas

    Returns:
        - atlas_tsv: pandas DataFrame
    """
    # fixed variables
    caps_path = caps_path[study]
    summary_path = caps_summary_path[study]
    pipeline_name = 't1-volume'
    atlas_id = 'AAL2'

    ### fetch df_add_data ### => only to get column names
    # this part will be factorized in a function later

    df_summary = pd.read_csv(summary_path, sep='\t')
    df_summary = df_summary[(df_summary.pipeline_name == pipeline_name) & (df_summary.atlas_id == atlas_id)]
    first_column_name = df_summary.first_column_name.item()
    last_column_name = df_summary.last_column_name.item()
    # print('First column name: ', first_column_name)
    # print('Last column name: ', last_column_name)
    df_data = pd.read_csv(caps_path, sep='\t', nrows=1)
    first_column_index = df_data.columns.get_loc(first_column_name)
    last_column_index = df_data.columns.get_loc(last_column_name)

    # other data to fetch
    col_names = ['participant_id', 'session_id', 'sex', 'diagnosis', 'age']
    add_indexes = [df_data.columns.get_loc(col_name) for col_name in col_names]

    # compute df_add_data
    # add 1 to first_column_index to ignore background
    used_columns = np.hstack([add_indexes, np.arange(first_column_index + 1, last_column_index + 1)]).flatten()
    df_add_data = pd.read_csv(caps_path, sep='\t', usecols=used_columns, nrows=1)

    ### updating atlas_tsv ###

    # drop line of tsv corresponding to background
    atlas_tsv = atlas_tsv[atlas_tsv.roi_name != 'Background']
    atlas_tsv['roi_name'] = atlas_tsv['roi_name'].apply(lambda x: x.strip())
    atlas_tsv['roi_name_data'] = [d.split('ROI-')[1].split('_intensity')[0].strip('_|-')
                                  for d in df_add_data.columns[len(col_names):]]
    # reset index
    atlas_tsv.reset_index(inplace=True)

    # sanity check
    if (atlas_tsv['roi_name'] != atlas_tsv['roi_name_data']).sum() != 0:
        print("Warning: there is a mismatch for", atlas_tsv[atlas_tsv['roi_name'] != atlas_tsv['roi_name_data']])
    else:
        atlas_tsv = atlas_tsv[['roi_value', 'roi_name']]

    # correction for left amygdala
    idx = atlas_tsv[atlas_tsv.roi_name == 'Amygdala_L'].index
    atlas_tsv.roi_value[idx] = 4201
    # print(atlas_tsv[(atlas_tsv.roi_value == 4202) | (atlas_tsv.roi_value == 4201)])

    # generate main regions after removing left/right and index numbers
    # main regions are for instance Vermis replacing ['Vermis_1', ..., 'Vermis_10']
    atlas_tsv['main_roi_name'] = atlas_tsv['roi_name'].apply(lambda x: main_region_name(x))
    main_regions = atlas_tsv[['roi_value', 'main_roi_name']].groupby('main_roi_name')['roi_value'].apply(
        lambda x: list(x))
    # append main regions to tsv and reindex it
    atlas_tsv = atlas_tsv.append(pd.DataFrame({'roi_name': main_regions.keys(), 'roi_value': main_regions.to_numpy()}))
    atlas_tsv = atlas_tsv.reset_index().drop(columns=['index', 'main_roi_name'])

    return atlas_tsv


def threshold_att_map(att_map):
    """
    Values of attention maps are supposed to be between 0 and 1.
    Values below 0 are set to 0, and those above 1 to 1.

    Params:
        - att_map: 3D array, attention map
    Returns:
        - (thresholded) att_map: 3D array
    """
    att_map = np.where(att_map > 0, att_map, 0)
    att_map = np.where(att_map < 1, att_map, 1)
    return att_map


def region_scores(data, atlas_tsv, atlas_map, resize=True):
    """
    Compute attention scores for each region.

    Params:
        - data: array, attention map
        - atlas_tsv: pandas DataFrame with roi_name and roi_value
        - atlas_map: array, index of the regions
        - resize: bool. If True, resize data to atlas_map.shape

    Returns:
        - scores: array of attention scores
    """
    # check for NaN
    if np.isnan(data).sum() > 0:
        print('NaN was found')
        return None, None

    if resize:
        resized_data = threshold_att_map(zoom(data, atlas_map.shape / np.array(data.shape)))
    else:
        resized_data = data

    # regional score computation
    raw_scores = np.zeros(len(atlas_tsv))
    normalized_scores = np.zeros(len(atlas_tsv))
    for k, roi_val in enumerate(atlas_tsv['roi_value']):
        if type(roi_val) == int:
            # simple region
            raw_scores[k] = (resized_data[atlas_map == roi_val]).sum()
            normalized_scores[k] = raw_scores[k] / (atlas_map == roi_val).sum()
        else:
            # main region
            temp_score = 0
            temp_area = 0
            for val in roi_val:
                temp_score += (resized_data[atlas_map == val]).sum()
                temp_area += (atlas_map == val).sum()
            raw_scores[k] = temp_score
            normalized_scores[k] = temp_score / temp_area

    return raw_scores, normalized_scores


def compute_scores(data_path, atlas_tsv, atlas_map):
    """
    Compute scores for all files in given data_path.
    Scores: scores of attention for each region in atlas_tsv.

    Params:
        - data_path: string, path to folder containing attention maps
        - atlas_tsv: tsv containing roi_name and roi_value as columns
        - atlas_map: 3D array filled with roi values

    Returns:
        - raw_scores: dataframe containing raw scores for each (subject, session)
        - normalized_scores: dataframe containing scores for each (subject, session)
            normalized by region areas.
    """
    cols = ['participant_id', 'session_id'] + atlas_tsv.roi_name.to_list()
    raw_scores = pd.DataFrame(columns=cols)
    normalized_scores = pd.DataFrame(columns=cols)
    # fetch file names
    files = [file for file in os.listdir(data_path) if os.path.splitext(file)[1] == '.npy']
    for k, file in tqdm(enumerate(files)):
        filename = file.split('.npy')[0]
        participant_id, session_id = file.strip('.npy').split('_')
        # load attention map
        att_map = np.load(os.path.join(data_path, file))
        # compute region scores
        raw_score, normalized_score = region_scores(att_map, atlas_tsv, atlas_map, resize=True)
        if raw_score is not None:
            raw_scores.loc[k, :2] = [participant_id, session_id]
            raw_scores.loc[k, 2:] = raw_score
            normalized_scores.loc[k, :2] = [participant_id, session_id]
            normalized_scores.loc[k, 2:] = normalized_score
    return raw_scores, normalized_scores


def MAR(names, scores, N_regions=10):
    """
    Determine Regions with the Most Attention.
    Params:
        - names: array of strings, names of regions (order matters)
        - scores: float array, attention score for each region (order matters)
        - N_regions: int, number of regions to output
    Returns:
        - list of strings: list of MA regions
    """
    return list(names[np.argsort(-scores)][:N_regions])


def MOR(region_list, subjects_nb, N_regions=10):
    """
    Most Occurent Regions.
    Params:
        - region_list: list of region names
        - N_regions: int, number of regions to output
    Returns:
        - pandas DataFrame with most occuring regions (col='region')
            and their frequency (col='freq')
    """
    occurences = collections.Counter(region_list)
    df = pd.DataFrame.from_dict(occurences, orient='index').reset_index().rename(columns={'index': 'region', 0: 'freq'})
    df['freq'] = df['freq'] / subjects_nb
    df = df.iloc[np.argsort(- df['freq'].to_numpy())[:N_regions]]
    return df.reset_index().drop(columns=['index'])


def MIR(df_scores, atlas_tsv, N_regions=10):
    """
    Determine most important regions given a dataframe of scores df_scores.
    Params:
        - df_scores: pandas DataFrame, scores for each volume.
            It may include participant_id and session_id, these columns
            will be dropped.
        - atlas_tsv: pandas DataFrame containing at least the column 'roi_name'
        - N_regions: int, number of most important regions to determine.
    """
    # total number of samples
    samples_nb = len(df_scores)
    # drop useless columns
    scores = df_scores.drop(columns=['participant_id', 'session_id'], errors='ignore')
    # region & main region names
    region_names = atlas_tsv.roi_name[:N_VOLUMES].to_numpy()
    main_region_names = atlas_tsv.roi_name[N_VOLUMES:].to_numpy()
    # record most import regions
    regions = []
    main_regions = []
    for k in range(samples_nb):
        regions += MAR(region_names, scores.iloc[k, :N_VOLUMES], N_regions)
        main_regions += MAR(main_region_names, scores.iloc[k, N_VOLUMES:], N_regions)

    # compute most occuring regions
    df_regions = MOR(regions, samples_nb, N_regions)
    df_main_regions = MOR(main_regions, samples_nb, N_regions)

    return df_regions, df_main_regions
