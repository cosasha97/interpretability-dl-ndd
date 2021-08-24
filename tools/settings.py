# global variables

# branch to target
BRANCH2TARGET = {
    'branch1': 'disease',
    'branch2': 'volumes',
    'branch3': 'age',
    'branch4': 'sex'
}

# target to branch
TARGET2BRANCH = {v: k for k, v in BRANCH2TARGET.items()}

N_VOLUMES = 120

# global paths
caps_path = '/network/lustre/dtlake01/aramis/datasets/{}/caps/caps_v2021/'
