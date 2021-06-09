import torch
import os


class ModelCheckpoint(object):
    def __init__(self, mode='min', min_delta=0):
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.is_better = None
        self._init_is_better(mode, min_delta)

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
        import numpy as np

        if self.best is None:
            self.best = metric
            self.save_model(metric, epoch, model, optimizer, train_metrics, val_metrics, path)
            return True

        if np.isnan(metric):
            raise Exception("Metric is NaN")

        if self.is_better(metric, self.best):
            self.best = metric
            self.save_model(metric, epoch, model, optimizer, path)
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')

        if mode == 'min':
            self.is_better = lambda a, best: a < best - best * min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + best * min_delta

    @staticmethod
    def save_model(metric, epoch, model, optimizer, train_metrics, val_metrics, path=""):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': metric,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }, os.path.join(path, "best_model.pt"))
