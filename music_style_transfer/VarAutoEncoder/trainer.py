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
        self.kl_loss = kl_loss


def generate_melodies(original_melody_nd: mx.nd.NDArray,
                      original_melody_class: mx.nd.NDArray,
                      num_classes: int,
                      output_path: str,
                      decoder: model.Decoder,
                      encoder: model.Encoder,
                      context: mx.Context,
                      add_sample_noise: bool=True):
    """
    Transform a melody into a range of classes.
    :param original_melody_nd: Pitches in mx.nd.NDArray format. Shape (1, seq_len)
    :param original_melody_class: Class in mx.nd.NDArray format. Shape (1,)
    :param num_classes: Iterate from 0 to this number (exclusive) and generate a melody for each class
    :param output_path: Output path for the .mid files
    :param encoder: Generator model
    :param decoder: Decoder model
    :param context: Mxnet context
    """

    # create a writer object
    writer = MelodyWriter()
    [original_melody_notes, original_articulation] = mx.nd.split(original_melody_nd, num_outputs=2, squeeze_axis=True, axis=2)
    z_mean, z_var = encoder(original_melody_nd, original_melody_class)
    pitches_per_class, articulations_per_class = [], []
    for c in range(0, num_classes):
        if add_sample_noise:
            sampled_eps = mx.nd.random_normal(0, scale=1.0, shape=z_mean.shape, ctx=context)
            z = z_mean + z_var * sampled_eps
        else:
            z = z_mean
        generated, generated_articulation = decoder(z, mx.nd.array([c], ctx=context))

        # take the maximum over all classes
        pitches_per_class += [mx.nd.argmax(generated, axis=2).asnumpy().ravel()]
        articulations_per_class += [(generated_articulation > 0.5).asnumpy().ravel()]

    original_melody = utils.construct_melody_from_integers([int(x) for x in original_melody_notes.asnumpy().ravel()],
                                                           [bool(x) for x in original_articulation.asnumpy().ravel()])
    print("Original melody: {}".format(original_melody.notes))
    writer.write_to_file(output_path + '_original.mid', original_melody)

    for class_index, (pitches, articulations) in enumerate(zip(pitches_per_class, articulations_per_class)):
        melody = utils.construct_melody_from_integers([int(x) for x in pitches],
                                                      [bool(x) for x in articulations])

        print("Melody for class {}: {}".format(class_index, melody.notes))
        writer.write_to_file(output_path + '_{}.mid'.format(class_index), melody)


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
                 encoder: model.Encoder,
                 decoder: model.Decoder):
        self.config = config
        self.context = context
        self.encoder = encoder
        self.decoder = decoder

        self._initialize_models()
        self._initialize_optimizers()
        self._initialize_metrics()
        self._initialize_losses()

        self.summary_writer = SummaryWriter(logdir='/tmp/out', flush_secs=5)

    def _initialize_losses(self):
        self.ce_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss(batch_axis=0, from_logits=False)
        self.ce_loss.hybridize()

        self.art_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(batch_axis=0, from_sigmoid=True)
        self.art_loss.hybridize()

        self.kl_loss = loss.VariationalKLLoss()
        self.kl_loss.hybridize()

    def _initialize_optimizers(self):
        params = self.decoder.collect_params()
        params.update(self.encoder.collect_params())
        optimizer_params = {'learning_rate': self.config.optimizer.learning_rate}
        optimizer_params.update(self.config.optimizer.params_to_dict())

        self.optimizer = gluon.Trainer(params,
                                       self.config.optimizer.optimizer,
                                       optimizer_params)

    def _initialize_models(self):
        self.decoder.initialize(mx.init.Xavier(), ctx=self.context)
        self.decoder.hybridize()

        self.encoder.initialize(mx.init.Xavier(), ctx=self.context)
        self.encoder.hybridize()

    def _initialize_metrics(self):
        def mean(_, pred):
            return pred.sum(), pred.size

        def accuracy(labels, pred):
            labels = np.ndarray.astype(labels, dtype=np.int32)
            pred = pred.argmax(axis=2)
            correct_labels = (pred == labels).sum()
            return correct_labels, labels.size

        self.r_metric = mx.metric.CompositeEvalMetric(
            [mx.metric.CustomMetric(mean)],
            name='reconstruction_loss'
        )

        self.r_acc_metric = mx.metric.CompositeEvalMetric(
            [mx.metric.CustomMetric(accuracy)],
            name='reconstruction_acc'
        )

        self.kl_metric = mx.metric.CompositeEvalMetric(
            [mx.metric.CustomMetric(mean)],
            name='kl_loss'
        )

        self.r_kl_metric = mx.metric.CompositeEvalMetric(
            [mx.metric.CustomMetric(mean)],
            name='total_loss'
        )

        self.metrics = [self.r_metric, self.r_acc_metric, self.kl_metric, self.r_kl_metric]

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
                batch_size = batch.data[0].shape[0]
                [tokens, classes] = batch.data[0].as_in_context(self.context), batch.data[1].as_in_context(self.context)

                self._update_step(batch_size, classes, tokens)

                self.train_state.n_batches += 1
                if self.train_state.n_batches % 50 == 0:
                    self._periodic_log(epoch, start_time)
                    if samples_output_path is not None and self.config.sampling_frequency > 0 and self.train_state.n_batches % self.config.sampling_frequency == 0 :

                        print("Generating samples from a melody of original class {}".format(int(classes[0].asscalar())))
                        generate_melodies(mx.nd.expand_dims(tokens[0, :], axis=0),
                                          classes[0],
                                          dataset.num_classes(),
                                          output_path='{}/generated-step-{}'.format(samples_output_path, self.train_state.n_batches),
                                          decoder=self.decoder,
                                          encoder=self.encoder,
                                          context=self.context)

                        print("Generating samples from a melody of original class {} without noise".format(int(classes[0].asscalar())))
                        generate_melodies(mx.nd.expand_dims(tokens[0, :], axis=0),
                                          classes[0],
                                          dataset.num_classes(),
                                          output_path='{}/generated-wo-noise-step-{}'.format(samples_output_path, self.train_state.n_batches),
                                          decoder=self.decoder,
                                          encoder=self.encoder,
                                          context=self.context,
                                          add_sample_noise=False)

                if self.train_state.n_batches % self.config.checkpoint_frequency == 0:
                    self._checkpoint(model_folder, validation_dataset)

                    if self.train_state.num_checkpoints_not_improved == self.config.num_checkpoints_not_improved:
                        print("Maximum checkpoints not improved reached. Stopping training.")
                        return

                    print("Checkpoint [{}]  {}\n".format(self.train_state.n_checkpoints, self._metric_to_string_output(self.train_state.n_batches)))

    def _checkpoint(self, model_folder, validation_dataset):
        self.train_state.n_checkpoints += 1
        print("\nCheckpoint {} reached.".format(self.train_state.n_checkpoints))

        utils.save_model(self.decoder, model_folder + '/decoder/params.{}'.format(self.train_state.n_checkpoints))
        utils.save_model(self.encoder, model_folder + '/encoder/params.{}'.format(self.train_state.n_checkpoints))

        for m in self.metrics:
            m.reset()

        for batch in validation_dataset:
            [tokens, classes] = batch.data[0].as_in_context(self.context), batch.data[1].as_in_context(self.context)
            kl_loss, reconstruction_loss, logits = self._forward_pass(classes, tokens)
            self.r_metric.update([mx.nd.ones_like(reconstruction_loss)], [reconstruction_loss])
            self.r_acc_metric.update([tokens[:,:,0]], [logits])
            self.kl_metric.update([mx.nd.ones_like(kl_loss)], [kl_loss])
            self.r_kl_metric.update([mx.nd.ones_like(kl_loss)], [kl_loss + reconstruction_loss])

        r_loss = self.r_kl_metric.get_metric(0).get()[1]
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
            loss_name = metric.name
            for metric_name, val in metric.get_name_value():
                self.summary_writer.add_scalar(tag="{}_{}".format(loss_name, metric_name),
                                               value=val, global_step=n_batches)
                out += '{}_{}={:.3f} '.format(loss_name, metric_name, val)

            metric.reset()
        return out

    def _update_step(self, batch_size, classes, tokens):
        with autograd.record():
            kl_loss, reconstruction_loss, logits = self._forward_pass(classes, tokens)
            loss = reconstruction_loss + kl_loss
        loss.backward()
        self.optimizer.step(batch_size)
        self.r_metric.update([mx.nd.ones_like(reconstruction_loss)], [reconstruction_loss])
        self.r_acc_metric.update([tokens[:,:,0], ], [logits, ])
        self.kl_metric.update([mx.nd.ones_like(kl_loss)], [kl_loss])
        self.r_kl_metric.update([mx.nd.ones_like(kl_loss)], [kl_loss + reconstruction_loss])

    def _forward_pass(self, classes, tokens_articulation):
        [tokens, articulation] = mx.nd.split(tokens_articulation, num_outputs=2, axis=2, squeeze_axis=True)

        z_means, z_vars = self.encoder(tokens_articulation, classes)
        sampled_eps = mx.nd.random_normal(0, scale=1.0, shape=z_means.shape, ctx=self.context)
        z = z_means + z_vars * sampled_eps
        logits, articulation_decoder = self.decoder(z, classes)
        # print("logits: ", logits.asnumpy())
        reconstruction_loss = self.ce_loss(logits, tokens) + self.art_loss(articulation, articulation_decoder)

        #print(logits, real_tokens, reconstruction_loss, mx.nd.softmax(logits).max(axis=2))
        kl_loss = self.config.kl_loss * self.kl_loss(z_means, z_vars)
        return kl_loss, reconstruction_loss, logits

    def _periodic_log(self, epoch, start_time):
        print("Epoch [{}] Batch [{}] updates/sec: {:.2f} {}".format(epoch,
                                                                    self.train_state.n_batches,
                                                                    self.train_state.n_batches / (time() - start_time),
                                                                    self._metric_to_string_output(self.train_state.n_batches)))

        self._log_gradients(self.decoder, 'decoder')
        self._log_gradients(self.encoder, 'encoder')


    def _log_gradients(self, model, output_prefix: str):
        # logging the gradients of parameters for checking convergence
        average_gradient_norm = 0.
        n_valid_gradients = 0
        for i, (name, grad) in enumerate(model.collect_params().items()):
            if grad.grad_req == 'null':
                continue
            self.summary_writer.add_scalar(tag=name, value=grad.grad().norm().asscalar(), global_step=self.train_state.n_batches)
            average_gradient_norm += grad.grad().norm().asscalar()
            n_valid_gradients += 1

        self.summary_writer.add_scalar(tag=output_prefix + '_global_grad',
                                       value=average_gradient_norm / n_valid_gradients,
                                       global_step=self.train_state.n_batches)