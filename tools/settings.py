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
