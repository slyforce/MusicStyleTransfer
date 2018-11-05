import mxnet as mx

class VariationalKLLoss(mx.gluon.loss.Loss):
    def __init__(self):
        super().__init__(weight=1.0, batch_axis=0)

    def hybrid_forward(self, F, z_means, z_vars):
        kl_loss = 0.5 * (z_vars * z_vars +  z_means * z_means - 1 - F.log(z_vars * z_vars))
        kl_loss_dimension_sum = F.sum(kl_loss, axis=2)
        kl_loss_sequence_mean = F.mean(kl_loss_dimension_sum, axis=1)
        return kl_loss_sequence_mean


class BinaryCrossEntropy(mx.gluon.loss.Loss):
    def __init__(self, from_sigmoid=False, label_smoothing=0.0, negative_label_downweighting=True):
        super().__init__(weight=1.0, batch_axis=0)
        self._from_sigmoid = from_sigmoid
        self.label_smoothing = label_smoothing
        self.negative_label_downweighting = negative_label_downweighting

    def _apply_label_smoothing(self, label):
        # label smoothing with binary labels -> each has a 50% probability
        return (1.-self.label_smoothing) * label + self.label_smoothing * 0.5

    def hybrid_forward(self, F, pred, label):

        if not self._from_sigmoid:
            # apply sigmoid if necessary
            pred = F.sigmoid(pred)

        s_label = self._apply_label_smoothing(label)

        # binary cross entropy with the smoothed labels
        # epsilon normalization term to not take the log of 0
        bce = -1 * (s_label * F.log(1e-12 + pred) + (1-s_label) * F.log(1e-12 + (1. - pred)))

        if self.negative_label_downweighting:
            # scale down the loss values for negative labels
            bce = F.where(label == 0.,
                          F.broadcast_mul(self._calculate_batchwise_upweighting(F, label), bce) * bce,
                          bce)

        return F.mean(bce, axis=0, exclude=True)

    def _calculate_batchwise_upweighting(self, F, label):

        # determine values of labels
        positive_labels = F.where(label == 1.,
                                  F.ones_like(label),
                                  F.zeros_like(label))
        negative_labels = F.where(label == 1.,
                                  F.zeros_like(label),
                                  F.ones_like(label))

        # sum up values batch-wise
        n_positives = positive_labels.sum(axis=0, exclude=True)
        n_negatives = negative_labels.sum(axis=0, exclude=True)

        # calculate the ratio of negative to positive samples
        # it's possible for there to be no positive samples therefore add an epsilon term
        downweight = n_positives / (n_negatives + 1e-12)

        # broadcast to 3D
        downweight = F.expand_dims(downweight, axis=1)
        downweight = F.expand_dims(downweight, axis=2)
        downweight = F.broadcast_like(downweight, label)

        return downweight