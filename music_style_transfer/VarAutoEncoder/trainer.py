from mxboard import *

from music_style_transfer.GAN import data

from music_style_transfer.MIDIUtil.MelodyWriter import MelodyWriter
from music_style_transfer.MIDIUtil.Melody import Melody
from music_style_transfer.MIDIUtil.Note import Note

import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from . import model, loss

import numpy as np

from time import time
from typing import List

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
                 optimizer: OptimizerConfig):
        self.batch_size = batch_size
        self.sampling_frequency = sampling_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.num_checkpoints_not_improved = num_checkpoints_not_improved
        self.optimizer = optimizer


def construct_melody_from_integers(notes: List[int],
                                   mask_offset: int=1):
    """
    Create a melody from a list of integers corresponding to midi pitches.
    """
    melody = Melody()
    melody.notes = [Note(midi_pitch=max(val-mask_offset, 0)) for val in notes]
    return melody


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

    z_mean, z_var = encoder(original_melody_nd, original_melody_class)
    pitches_per_class = []
    for c in range(0, num_classes):
        if add_sample_noise:
            sampled_eps = mx.nd.random_normal(0, scale=1.0, shape=z_mean.shape, ctx=context)
            z = z_mean + z_var * sampled_eps
        else:
            z = z_mean
        generated = decoder(z, mx.nd.array([c], ctx=context))

        # take the maximum over all classes
        pitches_per_class += [mx.nd.argmax(generated, axis=2).asnumpy().ravel()]

    original_melody = construct_melody_from_integers([int(x) for x in original_melody_nd.asnumpy().ravel()])
    print("Original melody: {}".format(original_melody.notes))
    writer.write_to_file(output_path + '_original.mid', original_melody)

    for class_index, pitches in enumerate(pitches_per_class):
        melody = construct_melody_from_integers([int(x) for x in pitches])

        print("Melody for class {}: {}".format(class_index, melody.notes))
        writer.write_to_file(output_path + '_{}.mid'.format(class_index), melody)


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
            pred = np.argmax(pred, axis=2)
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

        self.metrics = [self.r_metric, self.r_acc_metric, self.kl_metric]

    def fit(self,
            dataset: data.Dataset,
            validation_dataset: data.Dataset,
            epochs: int,
            samples_output_path: str = None):

        n_batches = 0
        n_checkpoints = 0
        checkpoint_not_improved = 0
        best_r_loss = np.inf

        start_time = time()

        print("Starting training")
        for epoch in range(epochs):
            for batch in dataset:
                batch_size = batch.data[0].shape[0]
                [tokens, classes] = batch.data[0].as_in_context(self.context), batch.data[1].as_in_context(self.context)

                self._update_step(batch_size, classes, tokens)

                n_batches += 1
                if n_batches % 50 == 0:
                    self._periodic_log(epoch, n_batches, start_time)
                    if samples_output_path is not None and self.config.sampling_frequency > 0 and n_batches % self.config.sampling_frequency == 0 :

                        print("Generating samples from a melody of original class {}".format(int(classes[0].asscalar())))
                        generate_melodies(mx.nd.expand_dims(tokens[0, :], axis=0),
                                          classes[0],
                                          dataset.num_classes(),
                                          output_path='{}/generated-step-{}'.format(samples_output_path, n_batches),
                                          decoder=self.decoder,
                                          encoder=self.encoder,
                                          context=self.context)

                        print("Generating samples from a melody of original class {} without noise".format(int(classes[0].asscalar())))
                        generate_melodies(mx.nd.expand_dims(tokens[0, :], axis=0),
                                          classes[0],
                                          dataset.num_classes(),
                                          output_path='{}/generated-wo-noise-step-{}'.format(samples_output_path, n_batches),
                                          decoder=self.decoder,
                                          encoder=self.encoder,
                                          context=self.context,
                                          add_sample_noise=False)

                if n_batches % self.config.checkpoint_frequency == 0:
                    n_checkpoints += 1
                    print("\nCheckpoint {} reached.".format(n_checkpoints))

                    for m in self.metrics:
                        m.reset()

                    for batch in validation_dataset:
                        [tokens, classes] = batch.data[0].as_in_context(self.context), batch.data[1].as_in_context(self.context)
                        kl_loss, reconstruction_loss, logits = self._forward_pass(classes, tokens)
                        self.r_metric.update([mx.nd.ones_like(reconstruction_loss)], [reconstruction_loss])
                        self.r_acc_metric.update([tokens], [logits])
                        self.kl_metric.update([mx.nd.ones_like(kl_loss)], [kl_loss])

                    r_loss = self.r_metric.get_metric(0).get()[1]
                    if r_loss < best_r_loss:
                        print("Reconstruction loss improved from {} to {}.".format(best_r_loss, r_loss))
                        best_r_loss = r_loss
                    else:
                        checkpoint_not_improved += 1
                        print("Reconstruction loss did not improve. {} out {} unsucessful checkpoints".format(checkpoint_not_improved, self.config.num_checkpoints_not_improved))
                        print("Best loss thus far: {}".format(best_r_loss))

                    if checkpoint_not_improved == self.config.num_checkpoints_not_improved:
                        return

                    print("Checkpoint [{}]  {}\n".format(n_checkpoints, self._metric_to_string_output(n_batches)))

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

    def _update_step(self, batch_size, real_classes, real_tokens):
        with autograd.record():
            kl_loss, reconstruction_loss, logits = self._forward_pass(real_classes, real_tokens)
            loss = reconstruction_loss + kl_loss
        loss.backward()
        self.optimizer.step(batch_size)
        self.r_metric.update([mx.nd.ones_like(reconstruction_loss)], [reconstruction_loss])
        self.r_acc_metric.update([real_tokens, ], [logits, ])
        self.kl_metric.update([mx.nd.ones_like(kl_loss)], [kl_loss])

    def _forward_pass(self, real_classes, real_tokens):
        z_means, z_vars = self.encoder(real_tokens, real_classes)
        sampled_eps = mx.nd.random_normal(0, scale=1.0, shape=z_means.shape, ctx=self.context)
        z = z_means + z_vars * sampled_eps
        logits = self.decoder(z, real_classes)
        # print("logits: ", logits.asnumpy())
        reconstruction_loss = self.ce_loss(logits, real_tokens)
        #print(logits, real_tokens, reconstruction_loss, mx.nd.softmax(logits).max(axis=2))
        kl_loss = self.kl_loss(z_means, z_vars)
        return kl_loss, reconstruction_loss, logits

    def _periodic_log(self, epoch, n_batches, start_time):
        print("Epoch [{}] Batch [{}] updates/sec: {:.2f} {}".format(epoch,
                                                                    n_batches,
                                                                    n_batches / (time() - start_time),
                                                                    self._metric_to_string_output(n_batches)))

        self._log_gradients(self.decoder, n_batches, 'decoder')
        self._log_gradients(self.encoder, n_batches, 'encoder')


    def _log_gradients(self, model, n_batches, output_prefix: str):
        # logging the gradients of parameters for checking convergence
        average_gradient_norm = 0.
        n_valid_gradients = 0
        for i, (name, grad) in enumerate(model.collect_params().items()):
            if grad.grad_req == 'null':
                continue
            self.summary_writer.add_scalar(tag=name, value=grad.grad().norm().asscalar(), global_step=n_batches)
            average_gradient_norm += grad.grad().norm().asscalar()
            n_valid_gradients += 1

        self.summary_writer.add_scalar(tag=output_prefix + '_global_grad',
                                       value=average_gradient_norm / n_valid_gradients,
                                       global_step=n_batches)