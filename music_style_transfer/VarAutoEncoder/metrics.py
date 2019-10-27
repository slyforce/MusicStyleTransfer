import mxnet as mx
import numpy as np
from mxnet.metric import check_label_shapes

class TopKAccuracy(mx.metric.TopKAccuracy):
    def __init__(self, ignore_label, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_label = ignore_label

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        #labels, preds = check_label_shapes(labels, preds, True)

        for label, pred_label in zip(labels, preds):
            if len(pred_label.shape) > 2:
                pred_label = mx.nd.reshape(pred_label, shape=[-1, pred_label.shape[-1]])
                label =  mx.nd.reshape(pred_label, shape=[-1])

            # Using argpartition here instead of argsort is safe because
            # we do not care about the order of top k elements. It is
            # much faster, which is important since that computation is
            # single-threaded due to Python GIL.
            pred_label = np.argpartition(pred_label.asnumpy().astype('float32'), -self.top_k)
            label = label.asnumpy().astype('int32')
            check_label_shapes(label, pred_label)
            num_dims = len(pred_label.shape)
            mask = (label != self.ignore_label).astype(np.int32)
            num_samples = mask.sum()

            num_classes = pred_label.shape[1]
            top_k = min(num_classes, self.top_k)
            for j in range(top_k):
                num_correct = ((pred_label[:, num_classes - 1 - j].flat == label.flat) * mask).sum()
                self.sum_metric += num_correct
                self.global_sum_metric += num_correct

            self.num_inst += num_samples
            self.global_num_inst += num_samples

class Accuracy(mx.metric.Accuracy):
    def __init__(self, ignore_label, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_label = ignore_label

    def update(self, labels, preds):
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = mx.nd.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')
            # flatten before checking shapes to avoid shape miss match
            label = label.flat
            pred_label = pred_label.flat

            check_label_shapes(label, pred_label)

            mask = (label != self.ignore_label).astype(np.int32)
            num_correct = ((pred_label == label) * mask).sum()

            self.sum_metric += num_correct
            self.global_sum_metric += num_correct
            self.num_inst += np.sum(mask)
            self.global_num_inst += np.sum(mask)