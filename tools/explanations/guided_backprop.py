"""
Inspired from https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/guided_backprop.py
"""

import torch
from torch.nn import ReLU
from torch.autograd import Variable
from tools.settings import *


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        self.forward_relu_outputs_per_branch = []
        self.forward_relu_outputs_index = -1
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        # self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """

        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = torch.clone(self.forward_relu_outputs[self.forward_relu_outputs_index])
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            # del self.forward_relu_outputs[-1]  # Remove last forward output
            self.forward_relu_outputs_index -= 1
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)
        for branch in BRANCH2TARGET.keys():
            module = list(getattr(self.model, branch)._modules.items())[1][1]
            if isinstance(module, ReLU):
                # print('register hook on ' + branch)
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def split_relu_outputs(self):
        """
        Separate relu outputs which are specific to a branch to the ones in the convolutional backbone.
        """
        # branch specific relu outputs are stored in forward_relu_outputs_per_branch
        self.forward_relu_outputs, self.forward_relu_outputs_per_branch = \
            self.forward_relu_outputs[:-len(BRANCH2TARGET)], self.forward_relu_outputs[-len(BRANCH2TARGET):]
        self.forward_relu_outputs.append(None)

    def generate_gradients(self, input_image):
        while len(input_image.shape) < 5:
            input_image = input_image[None, ...]
        # add grad to input
        input_image = Variable(input_image, requires_grad=True)
        # Forward pass
        model_output = self.model(input_image)
        self.split_relu_outputs()

        attention_maps = dict()
        torch.autograd.set_detect_anomaly(True)
        for k, branch in enumerate(BRANCH2TARGET):
            print(branch)
            # Zero gradients
            self.model.zero_grad()
            input_image.grad = None
            # reset index
            self.forward_relu_outputs_index = -1
            # update forward_relu_outputs
            self.forward_relu_outputs[-1] = self.forward_relu_outputs_per_branch[k]
            # Backward pass
            if k < len(BRANCH2TARGET) - 1:
                model_output[k].sum().backward(retain_graph=True)
            else:
                model_output[k].sum().backward()
            # Convert Pytorch variable to numpy array
            # [0] to get rid of the first channel (1, ...)
            attention_maps[branch] = input_image.grad.cpu().data.numpy().squeeze()
            # Normalize between 0-1
            min_ = attention_maps[branch].min()
            max_ = attention_maps[branch].max()
            attention_maps[branch] = (attention_maps[branch] - min_) / (max_ - min_)
        # delete saved outputs
        for output in self.forward_relu_outputs:
            # del self.forward_relu_outputs[-1]  # Remove last forward output
            del output
        return attention_maps
