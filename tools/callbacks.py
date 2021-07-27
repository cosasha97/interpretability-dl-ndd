import torch
import os
import numpy as np


class ModelCheckpoint(object):
    def __init__(self, mode='min', min_delta=0, save_last_model=False, name="best_model.pt"):
        """
        Save checkpoint of the model state, optimizer parameters, metrics, loss and epoch values.

        :param mode: string
        :param min_delta: float
        :param save_last_model: bool. If True, automatically save model at each iteration under 'last_model'.
        :param name: string, name of the model saved according to a given metric (!= last model)
        """
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.is_better = None
        self.save_last_model = save_last_model
        self._init_is_better(mode, min_delta)
        self.name = name

    def step(self, metric, epoch, model, optimizer, train_metrics=None, val_metrics=None, path=""):
        """
        Monitor progress and save data if necessary.

        :param metric: float, value of the metric used to monitor progress
        :param epoch: int, current epoch
        :param model: pytorch model
        :param optimizer: pytorch optimizer
        :param train_metrics: dict of lists (evolution of the metrics on training set during model training)
        :param val_metrics: dict of lists (evolution of the metrics on validation set during model training)
        :param path: string, directory path where data will be stored
        """
        # default: save model in last_model.pt
        if self.save_last_model:
            self.save_model(metric, epoch, model, optimizer, train_metrics, val_metrics, path, "last_model.pt")

        if self.best is None:
            self.best = metric
            self.save_model(metric, epoch, model, optimizer, train_metrics, val_metrics, path, self.name)
            return True

        if np.isnan(metric):
            raise Exception("Metric is NaN")

        if self.is_better(metric, self.best):
            self.best = metric
            self.save_model(metric, epoch, model, optimizer, train_metrics, val_metrics, path, self.name)
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')

        if mode == 'min':
            self.is_better = lambda a, best: a < best - best * min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + best * min_delta

    def save_model(self, metric, epoch, model, optimizer, train_metrics, val_metrics, path="", name="best_model.pt"):
        print('### ModelCheckpoint: saving model {} ###'.format(name))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': metric,
            'best_loss': self.best,  # same as loss when best_model is saved
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }, os.path.join(path, name))
