
class ModelCheckpoint(object):
    def __init__(self, mode='min', min_delta=0):
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.is_better = None
        self._init_is_better(mode, min_delta)

    def step(self, metrics, epoch, model, optimizer, path="model_0.pt"):
        import numpy as np

        if self.best is None:
            self.best = metrics
            self.save_model(metrics, epoch, model, optimizer, path)
            return True

        if np.isnan(metrics):
            raise Exception("Metric is NaN")

        if self.is_better(metrics, self.best):
            self.best = metrics
            self.save_model(metrics, epoch, model, optimizer, path)
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')

        if mode == 'min':
            self.is_better = lambda a, best: a < best - best * min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + best * min_delta

    def save_model(self, metrics, epoch, model, optimizer, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': metrics,
        }, path)