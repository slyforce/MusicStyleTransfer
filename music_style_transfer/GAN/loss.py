import mxnet as mx
from mxnet.gluon.loss import Loss

class BinaryCrossEntropy(Loss):
    def __init__(self, label_smoothing: float=0.0):
        super().__init__(1.0, 0)
        self.label_smoothing = label_smoothing

    def hybrid_forward(self, F, preds, labels):
        preds = F.squeeze(preds)
        labels = F.squeeze(labels)
        if self.label_smoothing > 0.:
            labels = self.label_smoothing * 0.5 + (1. - self.label_smoothing) * labels
        return -(F.log(preds+1e-12)*labels + F.log(1.-preds+1e-12)*(1.-labels))