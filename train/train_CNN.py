import numpy as np
from tqdm import tqdm
import pdb

# torch
import torch
import torch.nn as nn


def set_seed(seed):
    """
    Fix seed.
    :param seed: int.
    """
    import torch
    import random
    import numpy as np
    torch.manual_seed(seed)  # pytorch
    random.seed(seed)
    np.random.seed(seed)


def compute_loss(output_x, true_x, output_v, true_v, output_age, true_age, output_sex, true_sex,
                 loss_weights=(1., 1., 1., 1.)):
    """
    Compute loss L
    Args:
        output_x: tensor, model prediction for the disease (label)
        true_x: tensor, true label
        output_v: tensor, model prediction for the volumes
        true_v: tensor, true volumes
        output_age: tensor, model prediction for age
        true_age: tensor, true age
        output_sex: tensor, model prediction for sex
        true_sex: tensor, true sex
        loss_weights: list of float, weights to assign to each loss
    """
    # criteria
    disease_criterion = nn.BCELoss()
    sex_criterion = nn.BCELoss()
    volumes_criterion = nn.MSELoss(reduction='mean')
    age_criterion = nn.MSELoss(reduction='mean')
    # computation
    L_disease = disease_criterion(output_x.flatten(), true_x)
    L_vol = volumes_criterion(output_v, true_v)
    L_age = age_criterion(output_age.flatten(), true_age)
    L_sex = sex_criterion(output_sex.flatten(), true_sex)
    L_total = L_disease * loss_weights[0] + L_vol * loss_weights[1] + L_age * loss_weights[2] + L_sex * loss_weights[3]
    return L_disease, L_vol, L_age, L_sex, L_total


def train(epoch,
          model,
          optimizer_,
          loader,
          loss_weights=(1., 1., 1., 1.),
          to_cuda=True,
          compute_metrics=True,
          rescaling=None):
    """
    Training of multi-branch model

    Args:
        epoch: int, epoch index
        model: pytorch model
        optimizer_: pytorch optimizer
        loader: data loader
        loss_weights: list of float, weights to assign to each loss
        to_cuda: bool. If True, moves data to gpu.
        compute_metrics: bool. If True, compute evaluation metrics for each branch.
        rescaling: pandas Series containing scaling factor (useful for metrics computation)
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
        # print(data['image'].shape)
        disease, volumes, age, sex = model(data,
                                           compute_metrics=compute_metrics,
                                           rescaling=rescaling)
        losses = compute_loss(disease.float(), data['label'].float(),
                              volumes.float(), data['volumes'].float(),
                              age.float(), data['age'].float(),
                              sex.float(), data['sex'].float(),
                              loss_weights)

        # update current losses
        train_loss += losses[4].item()
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

    print('Epoch: {}, Average loss: {:.4f}, Accuracy: {:.2f}'.format(epoch, losses['train'], accuracy))

    # scale metrics and add them to dictionary
    if compute_metrics:
        losses.update(model.compute_metrics())

    return losses


def test(model,
         loader,
         loss_weights=(1., 1., 1., 1.),
         to_cuda=True,
         rescaling=None,
         compute_metrics=True,
         eval_mode=True,
         save_predictions=False):
    """
    Test a trained model

    Args:
        model: pytorch model
        loader: data loader
        to_cuda: bool. If True, moves data to gpu.
        compute_metrics: bool. If True, compute evaluation metrics for each branch.
        save_predictions: bool. If True, save predictions for each branch over the whole dataset.
    """
    model.reset_metrics()
    if eval_mode:
        model.eval()
        print("Eval mode: activated.")
    else:
        print("Eval mode: not activated.")
    test_loss = 0
    L_disease, L_vol, L_age, L_sex = 0, 0, 0, 0
    correct = 0.

    # predictions
    if save_predictions:
        participant_id = []
        session_id = []
        pred_disease = None
        pred_volumes = None
        pred_age = None
        pred_sex = None

        def update_predictions(preds, new_preds):
            if preds is None:
                return new_preds
            else:
                return np.concatenate((preds, new_preds), axis=0)

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if torch.cuda.is_available() and to_cuda:
                for key in data.keys():
                    try:
                        data[key] = data[key].cuda()
                    except AttributeError:
                        pass

            if eval_mode:
                disease, volumes, age, sex = model(data,
                                                   compute_metrics=compute_metrics,
                                                   rescaling=rescaling)
            else:
                model.train()
                _ = model(data, compute_metrics=compute_metrics, rescaling=rescaling)
                model.eval()
                disease, volumes, age, sex = model(data,
                                                   compute_metrics=compute_metrics,
                                                   rescaling=rescaling)

            # saving predictions
            if save_predictions:
                participant_id = participant_id + data['participant_id']
                session_id = session_id + data['session_id']
                pred_disease = update_predictions(pred_disease, disease.cpu().numpy())
                pred_volumes = update_predictions(pred_volumes, volumes.cpu().numpy())
                pred_age = update_predictions(pred_age, age.cpu().numpy())
                pred_sex = update_predictions(pred_sex, sex.cpu().numpy())

            losses = compute_loss(disease.float(), data['label'].float(),
                                  volumes.float(), data['volumes'].float(),
                                  age.float(), data['age'].float(),
                                  sex.float(), data['sex'].float(),
                                  loss_weights)

            # update current losses
            test_loss += losses[4].item()
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

    print('Test set loss: {:.4f}, Accuracy: {:.2f}'.format(losses['test'], accuracy))

    # scale metrics and add them to dictionary
    if compute_metrics:
        losses.update(model.compute_metrics())

    if save_predictions:
        return {'id': np.array([participant_id, session_id]).T,
                'disease': pred_disease,
                'volumes': pred_volumes,
                'age': pred_age,
                'sex': pred_sex}, \
               losses

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
