import tensorflow as tf

from . import model


class OptimizerConfig:
    def __init__(self,
                 optimizer: str,
                 learning_rate: float):
        self.optimizer = optimizer
        self.learning_rate = learning_rate


class TrainConfig:
    def __init__(self,
                 batch_size: int,
                 g_optimizer: OptimizerConfig,
                 d_optimizer: OptimizerConfig):
        self.batch_size = batch_size
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer


class Trainer:
    def __init__(self,
                 config: TrainConfig,
                 generator: model.Generator,
                 discriminator: model.Discriminator):
        self.config = config
        self.generator = generator
        self.discriminator = discriminator

    def _initialize_optimizers(self):
        if self.config.g_optimizer.optimizer == 'adam':
            self.g_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.g_optimizer.learning_rate)
        else:
            raise NotImplementedError

        if self.config.d_optimizer.optimizer == 'adam':
            self.d_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.d_optimizer.learning_rate)
        else:
            raise NotImplementedError

    def fit(self, dataset, epochs=10):

        n_batches = 0
        for epoch in range(epochs):
            for tokens, conditional_class in dataset:
                sequence_lengths = tf.reduce_sum(tokens != 0., axis=1)
                real_inputs = [tokens, sequence_lengths, conditional_class]

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_inputs = self.generator(
                        real_inputs, training=True)

                    real_output = self.discriminator(
                        real_inputs, training=True)
                    generated_output = self.discriminator(
                        generated_inputs, training=True)

                    gen_loss = model.generator_loss(generated_output)
                    disc_loss = model.discriminator_loss(
                        real_output, generated_output)

                gradients_of_generator = gen_tape.gradient(
                    gen_loss, self.generator.variables)
                gradients_of_discriminator = disc_tape.gradient(
                    disc_loss, self.discriminator.variables)

                self.g_optimizer.apply_gradients(
                    zip(gradients_of_generator, self.generator.variables))
                self.d_optimizer.apply_gradients(
                    zip(gradients_of_discriminator, self.discriminator.variables))

                n_batches += 1
                print("G/D loss {} / {}".format(gen_loss, disc_loss))
