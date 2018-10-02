from mxboard import *

from music_style_transfer.MIDIUtil.MelodyWriter import MelodyWriter
from music_style_transfer.MIDIUtil.Melody import Melody
from music_style_transfer.MIDIUtil.Note import Note

import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from . import model
from . import data

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
                 discriminator_update_steps: int,
                 sampling_frequency: int,
                 d_label_smoothing: float,
                 g_optimizer: OptimizerConfig,
                 d_optimizer: OptimizerConfig):
        self.batch_size = batch_size
        self.discriminator_update_steps = discriminator_update_steps
        self.sampling_frequency = sampling_frequency
        self.d_label_smoothing = d_label_smoothing
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer


def construct_melody_from_integers(notes: List[int],
                                   mask_offset: int=1):
    """
    Create a melody from a list of integers corresponding to midi pitches.
    """
    melody = Melody()
    melody.notes = [Note(midi_pitch=max(val-mask_offset, 0)) for val in notes]
    return melody


def generate_melodies(original_melody_nd: mx.nd.NDArray,
                      num_classes: int,
                      output_path: str,
                      generator: model.Generator,
                      context: mx.Context):
    """
    Transform a melody into a range of classes.
    :param original_melody_nd: Pitches in mx.nd.NDArray format. Shape (1, seq_len)
    :param num_classes: Iterate from 0 to this number (exclusive) and generate a melody for each class
    :param output_path: Output path for the .mid files
    :param generator: Generator model
    :param context: Mxnet context
    """

    # create a writer object
    writer = MelodyWriter()

    pitches_per_class = []
    for c in range(0, num_classes):
        noise = generator.create_noise(original_melody_nd.shape).as_in_context(context)
        generated = generator.forward(original_melody_nd, mx.nd.array([c], ctx=context), noise)

        # take the maximum over all classes
        print(mx.nd.max(generated, axis=2))
        pitches_per_class += [mx.nd.argmax(generated, axis=2).asnumpy().ravel()]

    original_melody = construct_melody_from_integers([int(x) for x in original_melody_nd.asnumpy().ravel()])

    print("Original melody: {}".format(original_melody.notes))
    for class_index, pitches in enumerate(pitches_per_class):
        melody = construct_melody_from_integers([int(x) for x in pitches])

        print("Melody for class {}: {}".format(class_index, melody.notes))
        writer.write_to_file(output_path + '_{}.mid'.format(class_index), melody)


class Trainer:
    def __init__(self,
                 config: TrainConfig,
                 context: mx.Context,
                 generator: model.Generator,
                 discriminator: model.Discriminator):
        self.config = config
        self.context = context
        self.generator = generator
        self.discriminator = discriminator

        self._initialize_models()
        self._initialize_optimizers()
        self._initialize_metrics()

        self.summary_writer = SummaryWriter(logdir='/tmp/out', flush_secs=5)

    def _initialize_optimizers(self):
        # trainer for the generator and the discriminator
        self.g_optimizer = gluon.Trainer(self.generator.collect_params(),
                                         self.config.g_optimizer.optimizer,
                                         {'learning_rate': self.config.g_optimizer.learning_rate}.update(self.config.d_optimizer.params_to_dict()))
        self.d_optimizer = gluon.Trainer(self.discriminator.collect_params(),
                                         self.config.d_optimizer.optimizer,
                                         {'learning_rate': self.config.d_optimizer.learning_rate}.update(self.config.d_optimizer.params_to_dict()))

    def _initialize_models(self):
        self.generator.initialize(mx.init.Xavier(), ctx=self.context)
        #self.generator.hybridize()

        self.discriminator.initialize(mx.init.Xavier(), ctx=self.context)
        #self.discriminator.hybridize()

    def _initialize_metrics(self):
        def distance(_, pred):
            return pred.sum(), pred.size

        self.d_acc = mx.metric.CompositeEvalMetric(
            [mx.metric.CustomMetric(distance)],
            name='Negative_EM_distance'
        )

    def fit(self,
            dataset: data.Dataset,
            epochs: int,
            samples_output_path: str = None):

        n_batches = 0
        start_time = time()

        print("Starting training")
        for epoch in range(epochs):
            for batch in dataset:
                batch_size = batch.data[0].shape[0]
                [real_tokens, real_classes] = batch.data[0].as_in_context(self.context), batch.data[1].as_in_context(self.context)
                seq_lens = mx.nd.where(real_tokens != 0,
                                       mx.nd.ones_like(real_tokens),
                                       mx.nd.zeros_like(real_tokens)).sum(axis=1)

                if n_batches % (self.config.discriminator_update_steps+1) == 0:
                    self._generator_step(batch_size, real_tokens, seq_lens)
                else:
                    self._discriminator_step(batch_size, real_tokens, real_classes, seq_lens)

                n_batches += 1
                if n_batches % 50 == 0:
                    self._periodic_log(epoch, n_batches, start_time)

                    if samples_output_path is not None and self.config.sampling_frequency > 0 and n_batches % self.config.sampling_frequency == 0 :

                        print("Generating samples from a melody of original class {}".format(int(real_classes[0].asscalar())))
                        generate_melodies(mx.nd.expand_dims(real_tokens[0, :], axis=0),
                                          dataset.num_classes(),
                                          output_path='{}/generated-step-{}'.format(samples_output_path, n_batches),
                                          generator=self.generator,
                                          context=self.context)

    def _periodic_log(self, epoch, n_batches, start_time):

        out = ''
        for metric in [self.d_acc]:
            loss_name = metric.name
            for metric_name, val in metric.get_name_value():
                self.summary_writer.add_scalar(tag="{}_{}".format(loss_name, metric_name),
                                               value=val, global_step=n_batches)
                out += '{}_{}={:.3f} '.format(loss_name, metric_name, val)

            metric.reset()

        print("Epoch [{}] Batch [{}] updates/sec: {:.2f} {}".format(epoch,
                                                                    n_batches,
                                                                    n_batches / (time() - start_time),
                                                                    out))

        self._log_gradients(self.discriminator, n_batches, 'discriminator')
        self._log_gradients(self.generator, n_batches, 'generator')


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


    def _generator_step(self, batch_size, tokens, seq_lens):

        classes = self._get_fake_classes(batch_size)
        with autograd.record():
            noise = self.generator.create_noise(tokens.shape).as_in_context(self.context)
            fake_tokens = self.generator.forward(tokens, classes, noise)
            _, fake_classes = self.discriminator.convert_to_one_hot(tokens, classes)

            loss = -1 * self.discriminator.forward(fake_tokens, fake_classes, seq_lens)
            loss.backward()

        self.g_optimizer.step(batch_size)

    def _discriminator_step(self, batch_size,
                            real_tokens, real_classes, seq_lens):

        classes = self._get_fake_classes(batch_size)

        with autograd.record():
            real_tokens_oh, real_classes_oh = self.discriminator.convert_to_one_hot(real_tokens, real_classes)
            loss_real = self.discriminator.forward(real_tokens_oh, real_classes_oh, seq_lens)

            noise = self.generator.create_noise(real_tokens.shape).as_in_context(self.context)
            fake_tokens = self.generator.forward(real_tokens, classes, noise)
            _, fake_classes = self.discriminator.convert_to_one_hot(real_tokens, classes)
            loss_fake = self.discriminator.forward(fake_tokens, fake_classes, seq_lens)

            loss = loss_fake - loss_real
            loss.backward()

        self.d_optimizer.step(batch_size)
        self.d_acc.update([mx.nd.ones_like(loss), ], [loss, ])


    def _get_fake_classes(self, batch_size):
        classes = mx.nd.array(np.random.randint(low=0, high=self.generator.config.conditional_class_config.input_dim,
                                                size=batch_size), ctx=self.context)
        return classes