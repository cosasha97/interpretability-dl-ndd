import matplotlib.pyplot as plt
from ipywidgets import interact
#%matplotlib inline


def get_slice(img, layer, channel=0):
    """
    Return appropriate slice of 3D image
    """
    d, h, w = img.shape
    if channel == 0:
        return img[min(layer,d-1), :, :]
    elif channel == 1:
        return img[:, min(layer,h-1), :]
    else:
        # channel == 2
        return img[:, :, min(layer,w-1)]


def visualize_explanations(img, explanations, plot_img=True):
    """
    Visualize explanations for all branches.
    Args:
        img: array or tensor
        explanations: dict of arrays, explanations for each branch
        plot_img: bool. If True, plot attention map on top of input image.
    """
    img = img.squeeze()
    print("Image shape:", *img.shape)

    # format image for visualization
    img = (img - img.min()) / img.max()

    branch2target = {
        'branch1': 'disease',
        'branch2': 'volumes',
        'branch3': 'age',
        'branch4': 'sex'
    }

    # Define a function to interactively visualize the data
    def explore_3dimage(layer, channel=0):
        plt.figure(figsize=(10, 5))
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))

        for k in range(0, 4):
            # display image slice
            ax[k].imshow(get_slice(img, layer, channel), cmap='gray')
            # display explanation slice
            ax[k].imshow(get_slice(explanations['branch' + str(k + 1)], layer, channel), cmap='bwr', alpha=0.5)
            # set subplot title
            ax[k].set_title(branch2target['branch' + str(k + 1)])

        fig.suptitle('Explore Attention of Brain MRI', fontsize=20)
        plt.axis('off')

    # Run the ipywidgets interact() function to explore the data
    interact(explore_3dimage, layer=(0, max(img.shape) - 1), channel=(0, 2))