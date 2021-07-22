import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from tools.settings import *


class CamExtractor():
    """
    Extracts cam features from the model
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x, branch=None):
        """
        Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        if self.target_layer is not None:
            for module_pos, module in self.model.features._modules.items():
                x = module(x)  # Forward
                if module_pos == self.target_layer:
                    x.register_hook(self.save_gradient)
                    conv_output = x  # Save the convolution output on that layer
        else:
            if branch is None:
                raise Exception("Either target_layer or branch must not be None.")
            x = self.model.features(x)
            module_pos, module = next(iter(getattr(self.model, branch)._modules.items()))
            x = module(x)
            x.register_hook(self.save_gradient)
            conv_output = x

        return conv_output, x

    def forward_pass(self, x, branch=None):
        """
        Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x, branch)
        if self.target_layer is not None:
            # x = x.view(x.size(0), -1)  # Flatten
            # Forward pass on the classifier
            x = getattr(self.model, branch)(x)
        else:
            for k, (module_pos, module) in enumerate(getattr(self.model, branch)._modules.items()):
                # do not apply again last convolution
                if k != 0:
                    x = module(x)  # Forward
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)
        if torch.cuda.is_available():
            self.null_scalar = torch.cuda.FloatTensor((1,)).fill_(0.)
        else:
            self.null_scalar = torch.FloatTensor((1,)).fill_(0.)

    def generate_cam(self,
                     input_image,
                     branch: str = None,
                     target: str = None,
                     resize=False,
                     to_cpu=False,
                     volume_index: int = None):
        """
        Args:
            input_image: array
            branch: string, name of the branch
            target: string, target name
            resize: bool. if True, resize attention maps to input_image shape
            to_cpu: bool. If True, moves cam to cpu
            volume_index: int. Index of the volume of interest
        """
        if branch is None:
            if target is None:
                raise Exception("branch and target can not be both None")
            else:
                branch = TARGET2BRANCH[target]

        while len(input_image.shape) < 5:
            input_image = input_image[None, ...]
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        if torch.cuda.is_available():
            input_image = input_image.cuda()
        conv_output, model_output = self.extractor.forward_pass(input_image, branch)
        # Zero grads
        self.model.features.zero_grad()
        getattr(self.model, branch).zero_grad()
        # Backward pass with specified target
        if BRANCH2TARGET[branch] == 'volumes' and volume_index is not None:
            # apply mask to target a specific volume
            mask_volumes = torch.zeros_like(model_output)
            mask_volumes[:, volume_index] = 1
            (model_output * mask_volumes).sum().backward(retain_graph=True)
        else:
            model_output.sum().backward(retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data[0]
        # Get convolution outputs
        target = conv_output.data[0]
        # Get weights from gradients
        weights = guided_gradients.mean(axis=(1, 2, 3))  # Take averages for each gradient
        # Multiply each weight with its conv output and then, sum
        cam = (weights.view(weights.shape + (1, 1, 1)) * target).sum(axis=0)
        cam = torch.where(cam > self.null_scalar, cam, self.null_scalar)
        cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize between 0-1
        # resize to shape of input image
        if resize:
            cam = zoom(cam.cpu(), input_image.shape[-3:] / np.array(cam.shape))
        if to_cpu and not resize:
            cam = cam.cpu()
        return cam

    def get_explanations(self,
                         input_image,
                         resize=False,
                         to_cpu=False,
                         volume_index: int = None):
        """
        Generate Grad-CAM attention maps for all branches.
        Args:
            input_image: 3D-tensor. 4D-tensor (channel included) and 5D-tensor
                (batch size dimension included) with dim 1 are also accepted.
            resize: bool, if True, resize attention maps to input_image shape
            to_cpu: bool, if True, explanation maps are moved to cpu
            volume_index: int. Index of the volume of interest
        """
        if torch.cuda.is_available():
            input_image = input_image.cuda()
        # grad-cam attention maps
        cams = {}
        for branch in BRANCH2TARGET.keys():
            cams[branch] = self.generate_cam(input_image,
                                             branch=branch,
                                             resize=resize,
                                             to_cpu=to_cpu,
                                             volume_index=volume_index)
        return cams
