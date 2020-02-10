from time import time

import mxnet as mx
import numpy as np
from VarAutoEncoder import data
from mxboard import *
from mxnet import autograd
from mxnet import gluon
from . import model, loss, utils, sampler, metrics

import os


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
                 kl_loss: float,
                 label_smoothing: float,
                 negative_label_downscaling: bool,
                 verbose: bool):
        self.batch_size = batch_size
        self.sampling_frequency = sampling_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.num_checkpoints_not_improved = num_checkpoints_not_improved
        self.optimizer = optimizer
        self.kl_loss_weight = kl_loss
        self.label_smoothing = label_smoothing
        self.negative_label_downscaling = negative_label_downscaling
        self.verbose = verbose


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
                 model: model.Model,
                 sampler: sampler.SamplerBase):
        self.config = config
        self.context = context
        self.model = model
        self.sampler = sampler

        self._initialize_model()
        self._initialize_optimizers()
        self._initialize_metrics()
        self._initialize_losses()

        self.summary_writer = SummaryWriter(logdir='/tmp/out', flush_secs=5)

    def _initialize_losses(self):
        self.token_loss = loss.SoftmaxCrossEntropy(axis=-1,
                                                   batch_axis=0)
        self.token_loss.hybridize()

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

        self.tokens_metric_ppl = mx.metric.Perplexity(name='ppl', ignore_label=0)
        self.tokens_metric_acc = metrics.Accuracy(name="acc", axis=2, ignore_label=0)
        self.tokens_metric_topk = metrics.TopKAccuracy(name="topk", top_k=5, ignore_label=0)

        self.kl_metric = mx.metric.CustomMetric(mean, name='kl_loss')
        self.main_metric = mx.metric.CustomMetric(mean, name='total_loss')

        self.metrics = [self.tokens_metric_ppl, self.tokens_metric_acc, self.tokens_metric_topk,
                        self.kl_metric, self.main_metric]
        self._reset_metrics()

    def fit(self,
            dataset: data.Dataset,
            model_folder: str,
            epochs: int,
            validation_dataset: data.Dataset = None):

        start_time = time()
        self.train_state = TrainingState()

        # try to resume from checkpoint if it exists
        self._load_latest_checkpoint(model_folder)

        for epoch in range(epochs):
            for batch in dataset:
                self._step(batch)

                self.train_state.n_batches += 1
                if self.train_state.n_batches % 50 == 0:
                    self._periodic_log(epoch, start_time)

                if self.train_state.n_batches % self.config.checkpoint_frequency == 0:
                    self._checkpoint(model_folder, validation_dataset)

                    if self.train_state.num_checkpoints_not_improved == self.config.num_checkpoints_not_improved:
                        print("Maximum checkpoints not improved reached. Stopping training.")
                        return

                if self.sampler is not None and self.config.sampling_frequency > 0 and self.train_state.n_batches % self.config.sampling_frequency == 0:
                    self.sampler.update_parameters(self.model)
                    self.sampler.process_batch(batch,
                                               os.path.join(model_folder, 'samples/step-{}'.format(self.train_state.n_batches)),
                                               dataset.num_classes())

    def _step(self, batch, is_train=True):
        [tokens, seq_lens, classes] = [x.as_in_context(self.context) for x in batch.data]
        [labels] = [x.as_in_context(self.context) for x in batch.label]
        batch_size, seq_len = tokens.shape

        if self.config.verbose:
            print("Step {}".format(self.train_state.n_batches))
            print("tokens:  {}, {}".format(tokens.shape, tokens))
            print("classes: {}, {}".format(classes.shape, classes))
            print("labels:  {}, {}".format(labels.shape, labels))

        # forward, todo: remove autograd in inference mode
        with autograd.record():
            probs, z_means, z_vars = self.model(tokens, seq_lens, classes)

            ce_loss = self.token_loss(probs, labels)
            kl_loss = self.kl_loss(z_means, z_vars)
            loss = ce_loss + self.config.kl_loss_weight * kl_loss

        # backward + update
        if is_train:
            loss.backward()
            self.optimizer.step(batch_size)

        self._update_metrics(kl_loss, labels, loss, probs)

    def _update_metrics(self, kl_loss, labels, loss, probs):
        self.tokens_metric_ppl.update(labels, probs)
        self.tokens_metric_acc.update(labels, probs)
        self.tokens_metric_topk.update(labels, probs)
        self.kl_metric.update(mx.nd.ones_like(kl_loss), kl_loss)
        self.main_metric.update(mx.nd.ones_like(loss), loss)

    def _load_latest_checkpoint(self, model_folder):
        print("Looking into folder {} for a valid training.".format(model_folder))
        try:
            latest_checkpoint = utils.get_latest_checkpoint_index(model_folder)
        except:
            print("No checkpoint was found. Starting training from scratch")
            return

        print("Checkpoint {} found. Resuming training.".format(latest_checkpoint))
        utils.load_model_parameters(self.model,
                                    os.path.join(model_folder, "params.{}".format(latest_checkpoint)),
                                    self.context)
        self.train_state = utils.load_object(os.path.join(model_folder, "train_state.pkl"))

    def _checkpoint(self, model_folder, validation_dataset):
        self.train_state.n_checkpoints += 1
        print("\nCheckpoint {} reached.".format(self.train_state.n_checkpoints))

        # save model parameters and train state
        utils.save_model(self.model, os.path.join(model_folder, 'params.{}'.format(self.train_state.n_checkpoints)))
        utils.save_object(self.train_state, os.path.join(model_folder, "train_state.pkl"))

        # reset metrics
        self._reset_metrics()

        # return early if no validation is required
        if validation_dataset is None:
            return

        for batch in validation_dataset:
            self._step(batch, is_train=False)

        reconstruction_loss = self.main_metric.get()[1]
        if reconstruction_loss < self.train_state.best_resconstruction_loss:
            print("Loss improved from {} to {}.".format(self.train_state.best_resconstruction_loss,
                                                        reconstruction_loss))
            self.train_state.best_resconstruction_loss = reconstruction_loss
        else:
            self.train_state.num_checkpoints_not_improved += 1
            print("Loss did not improve. {} out {} unsucessful checkpoints".format(
                self.train_state.num_checkpoints_not_improved,
                self.config.num_checkpoints_not_improved))
            print("Best loss thus far: {}".format(self.train_state.best_resconstruction_loss))
        print("Checkpoint [{}]  {}\n".format(self.train_state.n_checkpoints,
                                             self._metric_to_string_output(self.train_state.n_batches)))
        self._reset_metrics()

    def _reset_metrics(self):
        for m in self.metrics:
            m.reset()

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
