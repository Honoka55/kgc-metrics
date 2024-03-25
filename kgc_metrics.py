import torch
from wilds.common.metrics.metric import Metric
from wilds.common.utils import minimum


class HitsAtK(Metric):
    def __init__(self, name=None, k=None):
        self.k = k
        if name is None:
            name = f'Hits@{self.k}'
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        if self.k is None:
            raise NotImplementedError

        hits = 0
        for i in range(self.k):
            hits += (y_pred[i] == y_true).float().mean()

        return hits

    def worst(self, metrics):
        return minimum(metrics)


class HitsAt1(HitsAtK):
    def __init__(self, name=None):
        super().__init__(name=name, k=1)


class HitsAt3(HitsAtK):
    def __init__(self, name=None):
        super().__init__(name=name, k=3)


class HitsAt10(HitsAtK):
    def __init__(self, name=None):
        super().__init__(name=name, k=10)


class MRR(Metric):
    def __init__(self, name=None):
        if name is None:
            name = 'MRR'
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        y_pred = y_pred.t()
        r_ranks = 0
        for i in range(len(y_pred)):
            if y_true[i] in y_pred[i]:
                r_ranks += 1 / (y_pred[i].tolist().index(y_true[i]) + 1)

        return torch.tensor(r_ranks / len(y_pred))

    def worst(self, metrics):
        return minimum(metrics)
