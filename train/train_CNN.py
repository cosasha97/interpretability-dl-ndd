import torch
from tqdm import tqdm

# torch
import torch.nn as nn


def compute_loss(output_x, true_x, output_v, true_v, output_age, true_age, output_sex, true_sex):
    """
    Compute loss L

    """
    # print(output_x.flatten(), output_v.flatten(), output_age.flatten(), output_sex.flatten())
    # L1 = nn.L1Loss(reduction='mean')
    # L_disease = L1(output_x.flatten(), true_x)
    BCE = nn.BCELoss()
    L_disease = BCE(output_x.flatten(), true_x)
    MSE = nn.MSELoss(reduction='mean')
    L_vol = MSE(output_v, true_v)
    L_age = MSE(output_age.flatten(), true_age)
    # L_sex = L1(output_sex, true_sex)
    L_sex = BCE(output_sex.flatten(), true_sex)
    # print(L_disease, L_mse, L_age, L_sex, L_disease + L_mse + L_age + L_sex)
    return L_disease, L_vol, L_age, L_sex, L_disease + L_vol + L_age + L_sex


def train(epoch, model, optimizer_, loader, to_cuda=True):
    """
    Training of multi-branch model

    Args:
        epoch: int, epoch index
        model: pytorch model
        optimizer_: pytorch optimizer
        loader: data loader
        to_cuda: bool. If True, moves data to gpu.
    """

    model.reset_metrics()
    model.train()
    train_loss = 0
    L_disease, L_vol, L_age, L_sex = 0, 0, 0, 0
    # accuracy
    correct = 0
    for batch_idx, data in tqdm(enumerate(loader)):
        if torch.cuda.is_available() and to_cuda:
            for key in data.keys():
                try:
                    data[key] = data[key].cuda()
                except AttributeError:
                    pass
        # zero the parameter gradients
        optimizer_.zero_grad()
        # forward
        disease, volumes, age, sex = model(data, compute_metrics=True)
        losses = compute_loss(disease.float(), data['label'].float(),
                              volumes.float(), data['volumes'].float(),
                              age.float(), data['age'].float(),
                              sex.float(), data['sex'].float())

        # update current losses
        train_loss += losses[-1].item()
        L_disease += losses[0].item()
        L_vol += losses[1].item()
        L_age += losses[2].item()
        L_sex += losses[3].item()

        # compute other metrics
        correct += ((disease.float() > 0.5).flatten() == data['label']).float().sum().item()

        # backward + optimize
        losses[-1].backward()
        optimizer_.step()

    # scale losses
    losses = {'disease': L_disease,
              'volumes': L_vol,
              'age': L_age,
              'sex': L_sex,
              'train': train_loss}

    for key in losses:
        losses[key] = losses[key] / len(loader.dataset)

    # other metrics
    accuracy = 100 * correct / len(loader.dataset)

    print('Epoch: {}, Average loss: {:.4f}, Accuracy: {:.2f}'.format(epoch, train_loss, accuracy))

    # scale metrics and add them to dictionary
    losses.update(model.compute_metrics())

    return losses


def test(model, loader, to_cuda=True):
    """
    Test a trained model

    Args:
        model: pytorch model
        loader: data loader
        to_cuda: bool. If True, moves data to gpu.
    """
    model.reset_metrics()
    model.eval()
    test_loss = 0
    L_disease, L_vol, L_age, L_sex = 0, 0, 0, 0
    correct = 0.
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if torch.cuda.is_available() and to_cuda:
                for key in data.keys():
                    try:
                        data[key] = data[key].cuda()
                    except AttributeError:
                        pass
            disease, volumes, age, sex = model(data, compute_metrics=True)
            losses = compute_loss(disease.float(), data['label'].float(),
                                  volumes.float(), data['volumes'].float(),
                                  age.float(), data['age'].float(),
                                  sex.float(), data['sex'].float())

            # update current losses
            test_loss += losses[-1].item()
            L_disease += losses[0].item()
            L_vol += losses[1].item()
            L_age += losses[2].item()
            L_sex += losses[3].item()

            # compute other metrics
            correct += ((disease.float() > 0.5).flatten() == data['label']).float().sum().item()

    # scale losses
    losses = {'disease': L_disease,
              'volumes': L_vol,
              'age': L_age,
              'sex': L_sex,
              'test': test_loss}

    for key in losses:
        losses[key] = losses[key] / len(loader.dataset)

    # other metrics
    accuracy = 100 * correct / len(loader.dataset)

    print('Test set loss: {:.4f}, Accuracy: {:.2f}'.format(test_loss, accuracy))

    # scale metrics and add them to dictionary
    losses.update(model.compute_metrics())

    return losses


def update_dict(global_dict, values):
    """
    Update global_dict with values.

    Args:
        global_dict: dictionary
        values: dictionary
    """
    if global_dict == {}:
        for key in values:
            global_dict[key] = [values[key]]
    else:
        for key in values.keys():
            global_dict[key].append(values[key])
