import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from . import model
from . import loss

import numpy as np

from time import time

class OptimizerConfig:
    def __init__(self,
                 optimizer: str,
                 learning_rate: float):
        self.optimizer = optimizer
        self.learning_rate = learning_rate


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


from music_style_transfer.MIDIUtil.MelodyWriter import MelodyWriter
from music_style_transfer.MIDIUtil.Melody import Melody
from music_style_transfer.MIDIUtil.Note import Note


def reconstruct_melody_from_notes(notes, mask_offset=1):
    melody = Melody()
    melody.notes = [Note(midi_pitch=val - mask_offset) for val in notes]
    return melody


def generate_melodies(original_melody,
                      num_classes,
                      output_path,
                      generator,
                      context):
    writer = MelodyWriter()

    mel_notes = []
    for c in range(1, num_classes):
        generated = generator(original_melody, mx.nd.array([c], ctx=context))[0]
        mel_notes += [mx.nd.argmax(generated, axis=2).asnumpy().ravel()]

    original_melody_object = reconstruct_melody_from_notes([int(x) for x in original_melody.asnumpy().ravel()])
    print("Original melody: {}".format(original_melody_object.notes))

    for i, notes in enumerate(mel_notes):
        melody = reconstruct_melody_from_notes([int(x) for x in notes])
        print("Melody for class {}: {}".format(i, melody.notes))
        writer.write_to_file(output_path + '_{}.mid'.format(i+1),
                             melody)


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
        self._initialize_losses()

    def _initialize_losses(self):
        self.g_loss = loss.BinaryCrossEntropy()
        self.g_loss.hybridize()

        self.d_loss = loss.BinaryCrossEntropy(label_smoothing=self.config.d_label_smoothing)
        self.d_loss.hybridize()

    def _initialize_optimizers(self):
        # trainer for the generator and the discriminator
        self.g_optimizer = gluon.Trainer(self.generator.collect_params(),
                                         self.config.g_optimizer.optimizer,
                                         {'learning_rate': self.config.g_optimizer.learning_rate})
        self.d_optimizer = gluon.Trainer(self.discriminator.collect_params(),
                                         self.config.d_optimizer.optimizer,
                                         {'learning_rate': self.config.d_optimizer.learning_rate})

    def _initialize_models(self):
        # initialize the generator and the discriminator
        self.generator.initialize(mx.init.Xavier(), ctx=self.context)
        self.generator.hybridize()

        self.discriminator.initialize(mx.init.Xavier(), ctx=self.context)
        self.discriminator.hybridize()

    def _initialize_metrics(self):
        def stable_log(x):
            return np.log( np.maximum(np.full(x.shape, fill_value=1e-10), x) )

        def bce(label, pred):
            return -np.sum(stable_log(pred) * label + stable_log(1. - pred) * (1. - label)), label.size

        self.g_acc = mx.metric.CompositeEvalMetric(
            [mx.metric.Accuracy(), mx.metric.CustomMetric(bce)],
            name='Generator'
        )

        self.d_acc = mx.metric.CompositeEvalMetric(
            [mx.metric.Accuracy(), mx.metric.CustomMetric(bce)],
            name='Discriminator'
        )

    def fit(self,
            dataset,
            epochs=10,
            samples_output_path: str = None):

        n_batches = 0
        start_time = time()

        print("Starting training")
        for epoch in range(epochs):
            for batch in dataset:
                batch_size = batch.data[0].shape[0]
                [real_tokens, real_classes] = batch.data[0].as_in_context(self.context), batch.data[1].as_in_context(self.context)
                ones = mx.nd.ones((batch_size,), ctx=self.context)
                zeros = mx.nd.zeros((batch_size,), ctx=self.context)
                seq_lens = mx.nd.where(real_tokens != 0,
                                       mx.nd.ones_like(real_tokens),
                                       mx.nd.zeros_like(real_tokens)).sum(axis=1)

                if n_batches % (self.config.discriminator_update_steps+1) == 0:
                    self._generator_step(batch_size, ones, real_tokens, real_classes, seq_lens)
                else:
                    self._discriminator_step(batch_size, ones, zeros, real_tokens, real_classes, seq_lens)

                n_batches += 1
                if n_batches % 50 == 0:
                    self._periodic_log(epoch, n_batches, start_time)

                    if samples_output_path is not None and self.config.sampling_frequency > 0 and n_batches % self.config.sampling_frequency == 0 :

                        print("Generating samples from a melody of original class {}".format(real_classes[0].asscalar()))
                        generate_melodies(mx.nd.expand_dims(real_tokens[0, :], axis=0),
                                          dataset.num_classes(),
                                          output_path='{}/generated-step-{}'.format(samples_output_path, n_batches),
                                          generator=self.generator,
                                          context=self.context)

    def _periodic_log(self, epoch, n_batches, start_time):
        generator_loss = 'Generator '
        for name, val in zip(*self.g_acc.get()):
            generator_loss += '{}={:.3f} '.format(name, val)
        discriminator_loss = 'Discriminator '
        for name, val in zip(*self.d_acc.get()):
            discriminator_loss += '{}={:.3f} '.format(name, val)
        print("Epoch [{}] Batch [{}] updates/sec: {:.2f} {} {}".format(epoch,
                                                                       n_batches,
                                                                       n_batches / (time() - start_time),
                                                                       generator_loss,
                                                                       discriminator_loss))
        self.d_acc.reset()
        self.g_acc.reset()

    def _generator_step(self, batch_size, ones, tokens, classes, seq_lens):
        with autograd.record():
            print(tokens, classes)
            fake_tokens, fake_classes = self.generator.forward(tokens, classes)
            print(fake_tokens.argmax(axis=2), fake_classes.argmax(axis=1))

            output = self.discriminator.forward(fake_tokens, fake_classes, seq_lens)
            loss = self.g_loss(output, ones)
            loss.backward()
        self.g_acc.update([ones, ], [output, ])
        self.g_optimizer.step(batch_size)

    def _discriminator_step(self, batch_size, ones, zeros,
                            real_tokens, real_classes, seq_lens):
        with autograd.record():
            real = self.discriminator.forward(*self.discriminator.convert_to_one_hot(real_tokens, real_classes), seq_lens)
            loss_real = self.d_loss(real, ones)

            fake_tokens, fake_classes = self.generator.forward(real_tokens, real_classes)
            # detach s.t. no gradients are given to the generator
            fake = self.discriminator.forward(fake_tokens.detach(), fake_classes.detach(), seq_lens)
            loss_fake = self.d_loss(fake, zeros)

            loss = loss_fake + loss_real
            loss.backward()

        self.d_optimizer.step(batch_size)
        self.d_acc.update([ones, ], [real, ])
        self.d_acc.update([zeros, ], [fake, ])
