import mxnet as mx

class VariationalKLLoss(mx.gluon.loss.Loss):
    def __init__(self):
        super().__init__(weight=1.0, batch_axis=0)

    def hybrid_forward(self, F, z_means, z_vars):
        kl_loss = 0.5 * (z_vars * z_vars +  z_means * z_means - 1 - F.log(z_vars * z_vars))
        kl_loss_dimension_sum = F.sum(kl_loss, axis=2)
        kl_loss_sequence_mean = F.mean(kl_loss_dimension_sum, axis=1)
        return kl_loss_sequence_mean
