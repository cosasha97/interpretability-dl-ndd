import numpy as np
from math import floor
import torch
from torchinfo import summary
from torchmetrics import Accuracy, Recall, Precision, MetricCollection, MeanSquaredError, R2Score, AUROC, F1

# torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, sample, convolutions, dropout=0.5):
        super().__init__()
        self.image_size = sample['image'].shape
        self.features = nn.Sequential()
        in_channels = self.image_size[0]
        d_h_w = self.image_size[1:]

        # recorded gradient norms
        self.gradient_norms = []

        # default parameters
        kernel_size = 3
        self.dropout = dropout

        for index, nb_conv in enumerate(convolutions[:-1]):
            in_channels, d_h_w = self.add_conv_unit(self.features, in_channels,
                                                    nb_conv, kernel_size, d_h_w,
                                                    index)

        # default nodes of dense layers
        self.dense_features = [32, 16]

        ## disease
        prefix = 'b1-'
        self.branch1 = nn.Sequential()
        # add convolutional unit
        _, d_h_w = self.add_conv_unit(self.branch1, in_channels, convolutions[-1],
                                      kernel_size, d_h_w, prefix=prefix)
        # compute number of output features
        self.features_output_size = np.prod(d_h_w) * convolutions[-1]
        # add dense unit
        self.add_dense_unit(self.branch1, self.features_output_size,
                            self.dense_features + [1], prefix=prefix)
        self.branch1.add_module(prefix + "sigmoid", nn.Sigmoid())

        ## volumes
        prefix = 'b2-'
        self.n_volumes = np.prod(sample['volumes'].shape).item()
        self.branch2 = nn.Sequential()
        # add convolutional unit
        _ = self.add_conv_unit(self.branch2, in_channels, convolutions[-1],
                               kernel_size, d_h_w, prefix=prefix)
        # add dense unit
        dense_features = [4 * self.n_volumes, 2 * self.n_volumes, self.n_volumes]
        self.add_dense_unit(self.branch2, self.features_output_size,
                            dense_features, prefix=prefix)

        ## age
        prefix = 'b3-'
        self.branch3 = nn.Sequential()
        # add convolutional unit
        _, d_h_w = self.add_conv_unit(self.branch3, in_channels, convolutions[-1],
                                      kernel_size, d_h_w, prefix=prefix)
        # add dense unit
        self.add_dense_unit(self.branch3, self.features_output_size,
                            self.dense_features + [1], prefix=prefix)

        ## sex
        prefix = 'b4-'
        self.branch4 = nn.Sequential()
        # add convolutional unit
        _, d_h_w = self.add_conv_unit(self.branch4, in_channels, convolutions[-1],
                                      kernel_size, d_h_w, prefix=prefix)
        # add dense unit
        self.add_dense_unit(self.branch4, self.features_output_size,
                            self.dense_features + [1], prefix=prefix)
        self.branch4.add_module(prefix + "sigmoid", nn.Sigmoid())

        # metrics
        self.b1_metrics = MetricCollection([Accuracy(), F1(), AUROC()])
        self.b2_metrics = MetricCollection([MeanSquaredError(squared=False),
                                            R2Score(num_outputs=self.n_volumes, multioutput='uniform_average')])
        # self.b3_metrics = MetricCollection([R2Score(num_outputs=self.n_volumes, multioutput='uniform_average')])
        self.b3_metrics = MetricCollection([MeanSquaredError(squared=False), R2Score()])
        self.b4_metrics = MetricCollection([Accuracy(), F1(), AUROC()])

    @staticmethod
    def conv_output_shape(d_h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size, kernel_size)
        new_d_h_w = []
        for k in range(len(d_h_w)):
            new_d_h_w.append(floor(((d_h_w[k] + (2 * pad) - (dilation * (kernel_size[k] - 1)) - 1) / stride) + 1))
        return new_d_h_w

    def add_conv_unit(self, sequential_block, in_channels, nb_conv, kernel_size, d_h_w, index='', prefix=''):
        """
        Build a single convolutional unit.
        Args:
            sequential_block: nn.Sequential()
            in_channels: int, number of channels in input
            kernel_size: int
            d_h_w: list of ints: depth, height and width of the input feature map
            index: (optional) int or string, index of the convolutional block
            prefix: (optional) string, prefix to add in layer names
        Return:
            nb_conv: number of output channels
            d_h_w: shape of the output feature maps (depth, height, width)
        """
        sequential_block.add_module(prefix + 'conv' + str(index), nn.Conv3d(in_channels, nb_conv, kernel_size))
        d_h_w = self.conv_output_shape(d_h_w, 3)
        sequential_block.add_module(prefix + 'relu' + str(index), nn.ReLU())
        sequential_block.add_module(prefix + 'bnn' + str(index), nn.BatchNorm3d(nb_conv))
        sequential_block.add_module(prefix + 'pool' + str(index), nn.MaxPool3d(2, 2))
        d_h_w = self.conv_output_shape(d_h_w, 2, stride=2)
        return nb_conv, d_h_w

    def add_dense_unit(self, sequential_block, in_features, dense_features, prefix=''):
        """
        Add dense unit to sequential block.
        Args:
            sequential_block: nn.Sequential()
            in_features: int, input features
            dense_features: list of ints, number of nodes for each linear layer
            prefix: string, prefix to add in layer names
        """
        sequential_block.add_module(prefix + 'flatten', nn.Flatten())
        for k, df in enumerate(dense_features[:-1]):
            sequential_block.add_module(prefix + 'linear' + str(k), nn.Linear(in_features, df))
            in_features = df
            sequential_block.add_module(prefix + 'relu' + str(k), nn.ReLU())
            sequential_block.add_module(prefix + 'batchnorm' + str(k), nn.BatchNorm1d(df))
            sequential_block.add_module(prefix + 'dropout' + str(k), nn.Dropout(p=self.dropout))
        sequential_block.add_module(prefix + 'linear' + str(len(dense_features) - 1),
                                    nn.Linear(in_features, dense_features[-1]))

    @staticmethod
    def reshape_input(x):
        if len(x.shape) == 5:
            return x
        elif len(x.shape) == 4:
            return x[None, ...]
        else:
            raise Exception("Bad input shape.")

    def format_input(self, x):
        if type(x) == dict:
            return self.reshape_input(x['image'])
        elif type(x) == torch.Tensor:
            return self.reshape_input(x)
        else:
            raise Exception('Bad input type.')

    def record_gradients(self, grad_norm):
        """
        Record norm of gradients at the end of self.features.
        """
        print('GRADIENT ', torch.linalg.norm(grad_norm).item())
        self.gradient_norms.append(grad_norm)

    def forward(self, data, compute_metrics=False, rescaling=None):
        """
        Forward pass. Compute metrics if compute_metrics is True.

        :param data: tensor, input features
        :param compute_metrics: bool,
        :param rescaling: parameters used to rescale normalized values (for regression). In practice, it
        may be used for age and volumes.

        :return : loss for each branch
        """
        x = self.features(self.format_input(data))
#         if x.requires_grad:
#             x.register_hook(self.record_gradients)
        disease = self.branch1(x)
        volumes = self.branch2(x)
        age = self.branch3(x)
        sex = self.branch4(x)

        if compute_metrics:
            # rescaling parameters
            if rescaling is not None:
                # volumes
                columns = rescaling[rescaling.keys().difference(['age'])]
                b2_scale = rescaling[columns].reshape((1, self.n_volumes))
                # age
                b3_scale = rescaling['age'].reshape((1, 1))
            else:
                # volumes
                b2_scale = torch.ones((1, self.n_volumes))
                # age
                b3_scale = torch.ones((1, 1))
            # compute metrics
            self.b1_metrics.update(disease.squeeze().detach(), data['label'].type(torch.int8).detach())
            self.b2_metrics.update(b2_scale*volumes.detach(), b2_scale*data['volumes'].detach())
            self.b3_metrics.update(b3_scale*age.squeeze().detach(), b3_scale*data['age'].detach())
            self.b4_metrics.update(sex.squeeze().detach(), data['sex'].type(torch.int8).detach())

        return disease, volumes, age, sex

    def compute_metrics(self):
        """
        Compute global metrics (across all patches seen) for all branches.
        Return:
            dictionary: metric_names -> values
        """
        all_metrics = dict()
        for branch in range(1, 5):
            # fetch metrics for each branch and add them to all_metrics dictionary
            # add prefix to each metric according to the corresponding branch
            all_metrics.update(
                {f'b{branch}_{k}': v.item() for k, v in getattr(self, 'b' + str(branch) + '_metrics').compute().items()})
        return all_metrics

    def reset_metrics(self):
        """
        Reset all metrics.
        """
        for branch in range(1, 5):
            getattr(self, 'b' + str(branch) + '_metrics').reset()

    def summary(self, batch_size=1):
        """
        Print a summary of the model.
        """
        summary(self, (batch_size,) + self.image_size, verbose=1)