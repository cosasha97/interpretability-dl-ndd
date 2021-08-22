from scipy.ndimage import zoom
import nibabel as nib
import pandas as pd
import numpy as np
import re


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


def update_atlas_tsv(atlas_tsv):
    """
    Update atlas_tsv:
    - add main regions
    - format roi_name
    - apply various sanity checks and corrections

    Params:
        - atlas_tsv: pandas DataFrame

    Returns:
        - atlas_tsv: pandas DataFrame
    """
    # fixed variables
    caps_path = '/network/lustre/dtlake01/aramis/datasets/adni/caps/caps_v2021.tsv'
    summary_path = '/network/lustre/dtlake01/aramis/datasets/adni/caps/caps_v2021_summary.tsv'
    pipeline_name = 't1-volume'
    atlas_id = 'AAL2'

    ### fetch df_add_data ###
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
    col_names = ['participant_id', 'session_id', 'sex', 'age']
    add_indexes = [df_data.columns.get_loc(col_name) for col_name in col_names]

    # compute df_add_data
    # add 1 to first_column_index to ignore background
    used_columns = np.hstack([add_indexes, np.arange(first_column_index + 1, last_column_index + 1)]).flatten()
    df_add_data = pd.read_csv(caps_path, sep='\t', usecols=used_columns, nrows=1)

    ### updating atlas_tsv ###

    # drop background line
    atlas_tsv = atlas_tsv[atlas_tsv.roi_name != 'Background']
    atlas_tsv['roi_name'] = atlas_tsv['roi_name'].apply(lambda x: x.strip())
    atlas_tsv['roi_name_data'] = [d.split('ROI-')[1].split('_intensity')[0].strip('_') for d in df_add_data.columns[4:]]
    # reset index
    atlas_tsv.reset_index(inplace=True)

    # sanity check
    if (atlas_tsv['roi_name'] != atlas_tsv['roi_name_data']).sum() != 0:
        raise Exception('No match')
    else:
        atlas_tsv = atlas_tsv[['roi_value', 'roi_name']]

    # correction for left amygdala
    idx = atlas_tsv[atlas_tsv.roi_name == 'Amygdala_L'].index
    atlas_tsv.roi_value[idx] = 4201
    # print(atlas_tsv[(atlas_tsv.roi_value == 4202) | (atlas_tsv.roi_value == 4201)])

    # generate main regions after removing left/right and index numbers
    atlas_tsv['main_roi_name'] = atlas_tsv['roi_name'].apply(lambda x: main_region_name(x))
    main_regions = atlas_tsv[['roi_value', 'main_roi_name']].groupby('main_roi_name')['roi_value'].apply(
        lambda x: list(x))
    # append main regions to tsv and reindex it
    atlas_tsv = atlas_tsv.append(pd.DataFrame({'roi_name': main_regions.keys(), 'roi_value': main_regions.to_numpy()}))
    atlas_tsv = atlas_tsv.reset_index().drop(columns=['index', 'main_roi_name'])

    return atlas_tsv


def region_scores(data, atlas_map, atlas_tsv, resize=True):
    """
    Compute attention scores for each region.

    Params:
        - data: array, attention map
        - atlas_map: array, index of the regions
        - atlas_tsv: pandas DataFrame with roi_name and roi_value
        - resize: bool. If True, resize data to atlas_map.shape

    Returns:
        - scores: array of attention scores
    """
    if resize:
        resized_data = zoom(data, atlas_map.shape / np.array(data.shape))

    # regional score computation
    scores = np.zeros(len(atlas_tsv))
    for k, roi_val in enumerate(atlas_tsv['roi_value']):
        if type(roi_val) == int:
            # simple region
            scores[k] = resized_data[atlas_map == roi_val].sum() / (atlas_map == roi_val).sum()
        else:
            # main region
            temp_score = 0
            temp_area = 0
            for val in roi_val:
                temp_score += resized_data[atlas_map == val].sum()
                temp_area += (atlas_map == val).sum()
            scores[k] = temp_score / temp_area

    return scores
