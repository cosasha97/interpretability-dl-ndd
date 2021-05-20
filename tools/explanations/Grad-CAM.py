import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class CamExtractor():
    """
    Extracts cam features from the model
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
        Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if module_pos == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x, branch):
        """
        Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = getattr(self.model, branch)(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)
        # grad-cam attention maps
        self.cams = {}

    def generate_cam(self, input_image, branch='branch1'):
        """
        Args:
            input_image: array
            branch: string, name of the branch
        """
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
        if torch.cuda.is_available():
            model_output.sum().backward(retain_graph=True)
        else:
            model_output.sum().backward(retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.cpu().data.numpy()[0]
        # Get convolution outputs
        target = conv_output.cpu().data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[3],
                                                    input_image.shape[2]), Image.ANTIALIAS)) / 255
        return cam

    def generate_all_cams(self, input_image):
        """
        Generate Grad-CAM attention maps for all branches.
        """
        branches = ['branch' + str(k) for k in range(1, 5)]
        for branch in branches:
            self.cams[branch] = self.generate_cam(input_image, branch=branch)
        return self.cams

    def visualize_cams(self, img, plot_img=True):
        """
        Visualize cams for all branches.
        Args:
            img: array or tensor
            plot_img: bool. If True, plot attention map on top of input image.
        """
        if not self.cams:
            _ = self.generate_all_cams(img)

        ## format image for visualization
        # rgb dimension set to last dimension
        formatted_img = np.transpose(img, (0, 2, 3, 1))[0, ...]
        formatted_img = (formatted_img - formatted_img.min()) / formatted_img.max()

        branch2target = {
            'branch1': 'classification',
            'branch2': 'volumes',
            'branch3': 'age',
            'branch4': 'sex'
        }

        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        indexes = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for k, branch in enumerate(branch2target):
            ax[indexes[k]].set_title(branch2target[branch])
            if plot_img: ax[indexes[k]].imshow(formatted_img)
            ax[indexes[k]].imshow(self.cams[branch], cmap='bwr', alpha=0.3)
