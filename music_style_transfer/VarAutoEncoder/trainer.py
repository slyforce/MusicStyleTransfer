from time import time
from mxboard import *

from . import model, loss, utils
from music_style_transfer.MIDIUtil.MelodyWriter import MelodyWriter

import mxnet as mx
import numpy as np
from music_style_transfer.GAN import data
from mxnet import autograd
from mxnet import gluon

class OptimizerConfig:
    def __init__(self,
                 optimizer: str,
                 optimizer_params: str,
                 learning_rate: float):
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.learning_rate = learning_rate

    def params_to_dict(self):
        """
         Turn strings in format key1:value1,key2:value2 to a dictionary.
         Key-Value pairs with more than one delimiter are ignored.
        """
        out = {}
        for key_val in self.optimizer_params.strip().split(','):
            key_val = key_val.split(':')
            if len(key_val) != 2.:
                continue
            out[str(key_val[0])] = float(key_val[1])

        return out


class TrainConfig:
    def __init__(self,
                 batch_size: int,
                 sampling_frequency: int,
                 checkpoint_frequency: int,
                 num_checkpoints_not_improved: int,
                 optimizer: OptimizerConfig,
                 kl_loss: float):
        self.batch_size = batch_size
        self.sampling_frequency = sampling_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.num_checkpoints_not_improved = num_checkpoints_not_improved
        self.optimizer = optimizer
        self.kl_loss_weight = kl_loss

class TrainingState:
    def __init__(self):
        self.n_checkpoints = 0
        self.n_batches = 0
        self.num_checkpoints_not_improved = 0
        self.best_resconstruction_loss = np.inf


class Trainer:
    def __init__(self,
                 config: TrainConfig,
                 context: mx.Context,
                 model: model.EncoderDecoder):
        self.config = config
        self.context = context
        self.model = model

        self._initialize_model()
        self._initialize_optimizers()
        self._initialize_metrics()
        self._initialize_losses()

        self.summary_writer = SummaryWriter(logdir='/tmp/out', flush_secs=5)

    def _initialize_losses(self):
        self.token_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(batch_axis=0, from_sigmoid=False)
        self.token_loss.hybridize()

        self.art_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(batch_axis=0, from_sigmoid=False)
        self.art_loss.hybridize()

        self.kl_loss = loss.VariationalKLLoss()
        self.kl_loss.hybridize()

    def _initialize_optimizers(self):
        params = self.model.collect_params()
        optimizer_params = {'learning_rate': self.config.optimizer.learning_rate}
        optimizer_params.update(self.config.optimizer.params_to_dict())

        self.optimizer = gluon.Trainer(params,
                                       self.config.optimizer.optimizer,
                                       optimizer_params)

    def _initialize_model(self):
        self.model.initialize(mx.init.Xavier(), ctx=self.context)
        #self.model.hybridize()

    def _initialize_metrics(self):
        def mean(_, pred):
            return pred.sum(), pred.size

        def accuracy(labels, pred):
            return ((pred > 0.) == labels).sum(), labels.size

        self.tokens_metric_bce = mx.metric.CustomMetric(mean, name='tokens_bce')
        self.tokens_metric_acc =  mx.metric.CustomMetric(accuracy, name='tokens_acc')

        self.arti_metric_bce = mx.metric.CustomMetric(mean, name='tokens_bce')
        self.arti_metric_acc =  mx.metric.CustomMetric(accuracy, name='tokens_acc')

        self.kl_metric = mx.metric.CustomMetric(mean, name='kl_loss')
        self.main_metric = mx.metric.CustomMetric(mean, name='total_loss')

        self.metrics = [self.tokens_metric_bce, self.tokens_metric_acc,
                        self.arti_metric_bce, self.arti_metric_acc,
                        self.kl_metric, self.main_metric]

    def fit(self,
            dataset: data.Dataset,
            validation_dataset: data.Dataset,
            model_folder: str,
            epochs: int,
            samples_output_path: str = None):

        start_time = time()
        self.train_state = TrainingState()

        print("Starting training")
        for epoch in range(epochs):
            for batch in dataset:
                self._step(batch)

                self.train_state.n_batches += 1
                if self.train_state.n_batches % 50 == 0:
                    self._periodic_log(epoch, start_time)
                    #if samples_output_path is not None and self.config.sampling_frequency > 0 and self.train_state.n_batches % self.config.sampling_frequency == 0 :
                    #    raise NotImplemented

                if self.train_state.n_batches % self.config.checkpoint_frequency == 0:
                    self._checkpoint(model_folder, validation_dataset)

                    if self.train_state.num_checkpoints_not_improved == self.config.num_checkpoints_not_improved:
                        print("Maximum checkpoints not improved reached. Stopping training.")
                        return

                    print("Checkpoint [{}]  {}\n".format(self.train_state.n_checkpoints, self._metric_to_string_output(self.train_state.n_batches)))

    def _step(self, batch):
        [tokens, articulations, classes] = [x.as_in_context(self.context) for x in batch.data]
        (batch_size, seq_len, _) = tokens.shape

        noise = self._generate_var_ae_noise(batch_size, seq_len)

        with autograd.record():
            loss, sep_losses, outputs = self._forward_pass_with_loss_computation(articulations, classes, noise, tokens)

        # backprop step
        loss.backward()

        # update step with metric logging
        self.optimizer.step(batch_size)
        self._update_metrics(loss, *sep_losses,
                             tokens, articulations,
                             *outputs)

    def _forward_pass_with_loss_computation(self, articulations, classes, noise, tokens):
        # forward pass through the network
        tokens_out, articulations_out, z_means, z_stddev = self.model(tokens,
                                                                      articulations,
                                                                      classes,
                                                                      classes,
                                                                      noise)
        # compute loss of
        # (1) tokens
        # (2) articulations
        # (3) KL-divergence to unit gaussian
        tk_loss = self.token_loss(tokens_out, tokens)
        art_loss = self.art_loss(articulations_out, articulations)
        kl_loss = self.config.kl_loss_weight * self.kl_loss(z_means, z_stddev)
        loss = kl_loss + art_loss + tk_loss
        return loss, (tk_loss, art_loss, kl_loss), (tokens_out, articulations_out)

    def _update_metrics(self, loss, tk_loss, art_loss, kl_loss,
                        tokens, articulations,
                        tokens_out, articulations_out):
        self.tokens_metric_bce.update(mx.nd.ones_like(tk_loss), tk_loss)
        self.tokens_metric_acc.update(tokens, tokens_out)

        self.arti_metric_acc.update(articulations, articulations_out)
        self.arti_metric_bce.update(mx.nd.ones_like(art_loss), art_loss)

        self.kl_metric.update(mx.nd.ones_like(kl_loss), kl_loss)
        self.main_metric.update(mx.nd.ones_like(loss), loss)

    def _generate_var_ae_noise(self, batch_size, seq_len):
        return mx.nd.random_normal(loc=0.,
                                   scale=1.,
                                   shape=(batch_size, seq_len, self.model.config.latent_dimension),
                                   ctx=self.context)

    def _checkpoint(self, model_folder, validation_dataset):
        self.train_state.n_checkpoints += 1
        print("\nCheckpoint {} reached.".format(self.train_state.n_checkpoints))

        utils.save_model(self.model, model_folder + '/params.{}'.format(self.train_state.n_checkpoints))

        for m in self.metrics:
            m.reset()

        for batch in validation_dataset:
            [tokens, articulations, classes] = [x.as_in_context(self.context) for x in batch.data]
            (batch_size, seq_len, _) = tokens.shape

            loss, sep_losses, outputs = self._forward_pass_with_loss_computation(articulations,
                                                                                 classes,
                                                                                 self._generate_var_ae_noise(batch_size, seq_len),
                                                                                 tokens)
            self._update_metrics(loss, *sep_losses,
                                 tokens, articulations,
                                 *outputs)

        r_loss = self.main_metric.get()[1]
        if r_loss < self.train_state.best_resconstruction_loss:
            print("Loss improved from {} to {}.".format(self.train_state.best_resconstruction_loss,
                                                        r_loss))
            self.train_state.best_resconstruction_loss = r_loss
        else:
            self.train_state.num_checkpoints_not_improved += 1
            print("Loss did not improve. {} out {} unsucessful checkpoints".format(
                self.train_state.num_checkpoints_not_improved,
                self.config.num_checkpoints_not_improved))
            print("Best loss thus far: {}".format(self.train_state.best_resconstruction_loss))

    def _metric_to_string_output(self, n_batches):
        out = ''
        for metric in self.metrics:
            for metric_name, val in metric.get_name_value():
                self.summary_writer.add_scalar(tag="{}".format(metric_name),
                                               value=val, global_step=n_batches)
                out += '{}={:.3f} '.format(metric_name, val)

            metric.reset()
        return out

    def _periodic_log(self, epoch, start_time):
        print("Epoch [{}] Batch [{}] updates/sec: {:.2f} {}".format(epoch,
                                                                    self.train_state.n_batches,
                                                                    self.train_state.n_batches / (time() - start_time),
                                                                    self._metric_to_string_output(self.train_state.n_batches)))

        self._log_gradients()


    def _log_gradients(self):
        # logging the gradients of parameters for checking convergence
        average_gradient_norm = 0.
        n_valid_gradients = 0
        for i, (name, grad) in enumerate(self.model.collect_params().items()):
            if grad.grad_req == 'null':
                continue
            self.summary_writer.add_scalar(tag=name, value=grad.grad().norm().asscalar(), global_step=self.train_state.n_batches)
            average_gradient_norm += grad.grad().norm().asscalar()
            n_valid_gradients += 1

        self.summary_writer.add_scalar(tag='global_grad',
                                       value=average_gradient_norm / n_valid_gradients,
                                       global_step=self.train_state.n_batches)