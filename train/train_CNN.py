
def compute_loss(output_x, true_x, output_v, true_v, output_age, true_age, output_sex, true_sex):
    """
    Compute loss L

    """
    # print(output_x.flatten(), output_v.flatten(), output_age.flatten(), output_sex.flatten())
    # L1 = nn.L1Loss(reduction='mean')
    # L_class = L1(output_x.flatten(), true_x)
    BCE = nn.BCELoss()
    L_class = BCE(output_x.flatten(), true_x)
    MSE = nn.MSELoss(reduction='mean')
    L_vol = MSE(output_v, true_v)
    L_age = MSE(output_age.flatten(), true_age)
    # L_sex = L1(output_sex, true_sex)
    L_sex = BCE(output_sex.flatten(), true_sex)
    # print(L_class, L_mse, L_age, L_sex, L_class + L_mse + L_age + L_sex)
    return L_class, L_vol, L_age, L_sex, L_class + L_vol + L_age + L_sex


def train(epoch, model, optimizer_, device, loader):
    """
    Training of a vae model
    :param epoch: int, epoch index
    :param model: pytorch model
    :param optimizer_: pytorch optimizer
    :param device: cuda device
    :param loader: data loader
    """

    model.train()
    train_loss = 0
    L_class, L_vol, L_age, L_sex = 0, 0, 0, 0
    # accuracy
    correct = 0
    for batch_idx, data in tqdm(enumerate(loader)):
        if torch.cuda.is_available():
            for key in data.keys():
                try:
                    data[key] = data[key].to(device)
                except AttributeError:
                    pass
        # zero the parameter gradients
        optimizer_.zero_grad()
        # forward
        classification, volumes, age, sex = model(data['image'])
        losses = compute_loss(classification.float(), data['label'].float(),
                              volumes.float(), data['volumes'].float(),
                              age.float(), data['age'].float(),
                              sex.float(), data['sex'].float())

        # update current losses
        train_loss += losses[-1].item()
        L_class += losses[0].item()
        L_vol += losses[1].item()
        L_age += losses[2].item()
        L_sex += losses[3].item()

        # compute other metrics
        correct += ((classification.float() > 0.5).flatten() == data['label']).float().sum().item()

        # backward + optimize
        losses[-1].backward()
        optimizer_.step()

    # scale losses
    for l in [train_loss, L_class, L_vol, L_age, L_sex]:
        l = l / len(loader.dataset)
    # other metrics
    accuracy = 100 * correct / len(loader.dataset)

    print('Epoch: {}, Average loss: {:.4f}, Accuracy: {:.2f}'.format(epoch, train_loss, accuracy))

    return {'classification': L_class,
            'volumes': L_vol,
            'age': L_age,
            'sex': L_sex,
            'train': train_loss}


def test(model, device, loader):
    """
    Test a trained vae model
    :param model: pytorch model
    """
    model.eval()
    test_loss = 0
    L_class, L_vol, L_age, L_sex = 0, 0, 0, 0
    correct = 0.
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            for key in data.keys():
                try:
                    data[key] = data[key].to(device)
                except AttributeError:
                    pass
            classification, volumes, age, sex = model(data['image'])
            losses = compute_loss(classification.float(), data['label'].float(),
                                  volumes.float(), data['volumes'].float(),
                                  age.float(), data['age'].float(),
                                  sex.float(), data['sex'].float())
            # update current losses
            test_loss += losses[-1].item()
            L_class += losses[0].item()
            L_vol += losses[1].item()
            L_age += losses[2].item()
            L_sex += losses[3].item()

            # compute other metrics
            correct += ((classification.float() > 0.5).flatten() == data['label']).float().sum().item()

    for l in [test_loss, L_class, L_vol, L_age, L_sex]:
        l = l / len(loader.dataset)
    # other metrics
    accuracy = 100 * correct / len(loader.dataset)

    print('Test set loss: {:.4f}, Accuracy: {:.2f}'.format(test_loss, accuracy))

    return {'classification': L_class,
            'volumes': L_vol,
            'age': L_age,
            'sex': L_sex,
            'test': test_loss}


def update_dict(dictionary, values, epoch):
    """
    Update dictionary using values.
    """
    for key in values.keys():
        dictionary[key][epoch] = values[key]