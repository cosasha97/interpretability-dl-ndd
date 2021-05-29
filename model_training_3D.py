import torch.optim as optim

# clinicaDL
from clinicadl.tools.tsv.data_split import create_split
from clinicadl.tools.deep_learning.data import generate_sampler, return_dataset, MRIDataset, get_transforms
from torch.utils.data import DataLoader
from clinicadl.tools.deep_learning import EarlyStopping

# own imports
from tools.models.CN5_FC3_3D import *
from tools.callbacks import *
from tools.data import *
from train.train_CNN import *

print("Beginning of the script - TRAINING")

# global parameters
caps_directory = '/network/lustre/dtlake01/aramis/datasets/adni/caps/caps_v2021/'
batch_size = 4
num_workers = 1
loss = 'default'
optimizer_name = 'Adam'
learning_rate = 1e-4
weight_decay = 1e-4

# fold iterator
fold_iterator = range(2)

# load dataframes
AD = pd.read_csv('subjects/AD.tsv',sep='\t')
CN = pd.read_csv('subjects/CN.tsv',sep='\t')

# remove samples with NaN
AD.drop(AD[AD.isna().sum(axis=1) > 0].index, inplace=True)
CN.drop(CN[CN.isna().sum(axis=1) > 0].index, inplace=True)

# split data between training and validation sets
training_df, valid_df = create_split('AD', AD, 'diagnosis',0.2)
df_CN = create_split('CN', CN, 'diagnosis',0.2)
training_df = training_df.append(df_CN[0]).reset_index()#.iloc[np.array([0,1,2,-1,-2,-3])]
valid_df = valid_df.append(df_CN[1]).reset_index()#.iloc[np.array([0,1,2,-1,-2,-3])]

# drop index column
training_df.drop(columns = ['index'], inplace=True)
valid_df.drop(columns = ['index'], inplace=True)

train_transforms, all_transforms = get_transforms('image', minmaxnormalization=True, data_augmentation=None )
# fetch volumetric data
df_add_data = fetch_add_data(training_df)

# all_transforms = torchvision.transforms.Compose([])

data_train = MRIDatasetImage(caps_directory, training_df, df_add_data=df_add_data,all_transformations=all_transforms) #train_transformations=all_transforms
data_valid = MRIDatasetImage(caps_directory, valid_df, df_add_data=df_add_data, all_transformations=all_transforms) #train_transformations=all_transforms,


# sampler
train_sampler = generate_sampler(data_train)
valid_sampler = generate_sampler(data_valid)
# loaders
train_loader = DataLoader(data_train,
                         batch_size=batch_size,
                         sampler=train_sampler,
                         num_workers=num_workers,
                         pin_memory=True)

valid_loader = DataLoader(data_valid,
                         batch_size=batch_size,
                         sampler=valid_sampler,
                         num_workers=num_workers,
                         pin_memory=True)

# get sample
sample = data_train[0]
# build model
model = Net(sample, [8, 16, 32, 64, 128])
if torch.cuda.is_available():
    print("To cuda")
    model.cuda()
model.summary()

nb_epochs = 20
# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# device
cuda = torch.device('cuda')
# record losses
train_losses = {
    'classification': np.zeros(nb_epochs),
    'volumes': np.zeros(nb_epochs),
    'age': np.zeros(nb_epochs),
    'sex': np.zeros(nb_epochs),
    'train': np.zeros(nb_epochs)
}
test_losses = {
    'classification': np.zeros(nb_epochs),
    'volumes': np.zeros(nb_epochs),
    'age': np.zeros(nb_epochs),
    'sex': np.zeros(nb_epochs),
    'test': np.zeros(nb_epochs)
}

# callbacks
ES = EarlyStopping(patience=5)
MC = ModelCheckpoint()

print("Beginning of the training")

# training
for epoch in range(nb_epochs):
    update_dict(train_losses, train(epoch, model, optimizer, cuda, train_loader), epoch)
    update_dict(test_losses, test(model, cuda, valid_loader), epoch)
    if ES.step(train_losses['train'][epoch]):
        break
    MC.step(train_losses['train'][epoch], epoch, model, optimizer, 'model_3D')  # path


# save training curves
def save_loss(loss, name="loss"):
    df = pd.DataFrame.from_dict(loss)
    df[df.any(axis=1)].to_csv(name + '.csv', index=False)


save_loss(train_losses, 'train_losses_3D')
save_loss(test_losses, 'val_losses_3D')