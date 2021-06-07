import matplotlib.pyplot as plt
from ipywidgets import interact
from tools.settings import *


# %matplotlib inline

def rotate270(array_2D):
    """
    Rotation of 90 counterclockwise
    :param array_2D: 2D array
    :return: rotated 2D array
    """
    return array_2D[:, ::-1].T


def get_slice(img, layer, channel=0):
    """
    Return appropriate slice of 3D image
    """
    d, h, w = img.shape
    if channel == 0:
        return rotate270(img[min(layer, d - 1), :, :])
    elif channel == 1:
        return rotate270(img[:, min(layer, h - 1), :])
    else:
        # channel == 2
        return rotate270(img[:, :, min(layer, w - 1)])


def visualize_explanations(img, explanations, plot_img=True, targets=None):
    """
    Visualize explanations for all branches.
    Args:
        img: array or tensor
        explanations: dict of arrays, explanations for each branch
        plot_img: bool. If True, plot attention map on top of input image.
        targets: string or list of strings. Targets to visualize, i.e. disease, volumes, age or sex.
    """
    img = img.squeeze()
    print("Image shape:", *img.shape)

    # format image for visualization
    img = (img - img.min()) / img.max()

    branch2target = BRANCH2TARGET.copy()

    if targets is not None:
        if type(targets) != list:
            targets = [targets]
        for key in list(branch2target.keys()):
            if branch2target[key] not in targets:
                branch2target.pop(key)

    # Define a function to interactively visualize the data
    def explore_3dimage(layer, channel=0):
        # nb images to plot
        n_images = len(branch2target)
        n_images = max(n_images, 2)

        fig, ax = plt.subplots(1, n_images, figsize=(n_images * 4, 4))

        for k, branch in enumerate(branch2target):
            # display image slice
            ax[k].imshow(get_slice(img.numpy(), layer, channel), cmap='gray')
            # display explanation slice
            ax[k].imshow(get_slice(explanations[branch], layer, channel), cmap='bwr', alpha=0.5)
            # set subplot title
            ax[k].set_title(branch2target[branch])

        fig.suptitle('Explore Attention of Brain MRI', fontsize=20)
        plt.axis('off')

    # Run the ipywidgets interact() function to explore the data
    interact(explore_3dimage, layer=(0, max(img.shape) - 1), channel=(0, 2))
