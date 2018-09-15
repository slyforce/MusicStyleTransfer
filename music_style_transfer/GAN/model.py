import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
tf.enable_eager_execution()
tfe = tf.contrib.eager


class TransformerConfig:
    def __init__(self,
                 n_layers: int,
                 model_dim: int,
                 hidden_dim: int):
        self.n_layers = n_layers
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim


class EmbeddingConfig:
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 mask_zero: bool):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mask_zero = mask_zero


class OutputLayerConfig:
    def __init__(self,
                 output_dim: int,
                 softmax: bool=True):
        self.output_dim = output_dim
        self.softmax = softmax


class ModelConfig:
    def __init__(self,
                 encoder_config: TransformerConfig,
                 embedding_config: EmbeddingConfig,
                 conditional_class_config: EmbeddingConfig,
                 output_layer_config: OutputLayerConfig,
                 class_output_layer_config: OutputLayerConfig):
        self.encoder_config = encoder_config
        self.embedding_config = embedding_config
        self.conditional_class_config = conditional_class_config
        self.output_layer_config = output_layer_config
        self.class_output_layer_config = class_output_layer_config


class Generator(tf.keras.Model):
    def __init__(self,
                 config: ModelConfig):
        super(Generator, self).__init__()
        self.config = config

        self.encoder = LSTM(
            units=config.encoder_config.hidden_dim,
            return_sequences=True
        )

        self.embeddings = Embedding(
            input_dim=self.config.embedding_config.input_dim,
            output_dim=self.config.embedding_config.hidden_dim,
            mask_zero=self.config.embedding_config.mask_zero
        )

        self.class_embeddings = Embedding(
            input_dim=self.config.conditional_class_config.input_dim,
            output_dim=self.config.conditional_class_config.hidden_dim,
            mask_zero=self.config.conditional_class_config.mask_zero
        )

        self.output_layer = Dense(
            units=self.config.output_layer_config.output_dim
        )

        self.class_output_layer = Dense(
            units=self.config.class_output_layer_config.output_dim
        )

    def call(self, inputs, training=True, mask=None):
        """
        inputs is a list of values
         - source tokens: the tokengss being fed into the encoder (batch_size, max_sequence_len)
         - sequence length: length of the input sequence (batch_size)
         - conditional class: start token for the decoder sequence (batch_size)
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        [source_tokens, conditional_class] = inputs

        # get the embeddings for the source tokens and conditional class
        # shape: (batch_size, seq_len, h_dim)
        source_emb = self.embeddings(source_tokens)
        # shape: (batch_size, h_dim)
        class_emb = self.class_embeddings(conditional_class)

        # shape: (batch_size, 1, h_dim)
        class_emb = tf.expand_dims(class_emb, axis=1)

        # place the class embedding in the first position
        encoder_input = tf.concat([source_emb, class_emb], 1)

        # call encoder on the sequence to generate a sequence of encoded tokens
        # of the same length
        encoder_output = self.encoder(encoder_input)

        # separate class token and embeddings
        class_emb = tf.slice(encoder_output, [0, 0, 0], [-1, 1, -1])
        token_emb = tf.slice(encoder_output, [0, 1, 0], [-1, -1, -1])

        # project the encoder outputs to the respective class vocabulary sizes
        token_output = tf.nn.softmax(self.output_layer(token_emb))
        class_output = tf.nn.softmax(tf.squeeze(self.class_output_layer(class_emb), axis=1))

        return token_output, class_output


class Discriminator(tf.keras.Model):
    def __init__(self,
                 config: ModelConfig):
        super(Discriminator, self).__init__()
        self.config = config

        self.encoder = LSTM(
            units=config.encoder_config.hidden_dim,
            return_sequences=True
        )

        self.embeddings = Dense(
            units=self.config.embedding_config.hidden_dim
        )

        self.class_embeddings = Dense(
            units=self.config.conditional_class_config.hidden_dim
        )

        self.output_layer = Dense(
            units=self.config.output_layer_config.output_dim
        )

    def call(self, inputs, training=True, mask=None):
        """
        inputs is a list of values
         - source tokens: the tokengss being fed into the encoder (batch_size, max_sequence_len)
         - sequence length: length of the input sequence (batch_size)
         - conditional class: start token for the decoder sequence (batch_size)
        :param inputs:
        :param training:
        :param mask:
        :return:
        """

        [source_tokens, conditional_class] = inputs
        conditional_class, source_tokens = self._convert_input_to_one_hot(conditional_class, source_tokens)

        # get the embeddings for the source tokens and conditional class
        # shape: (batch_size, seq_len, h_dim)
        source_emb = self.embeddings(source_tokens)
        # shape: (batch_size, 1, h_dim)
        class_emb = tf.expand_dims(self.class_embeddings(conditional_class), axis=1)

        # place the class embedding in the first position
        encoder_input = tf.concat([class_emb, source_emb], axis=1)

        # call encoder on the sequence to generate a sequence of encoded tokens
        # of the same length
        encoder_output = self.encoder(encoder_input)

        # take the maximum values over the time axis
        # shape (batch_size, hidden_dim)
        encoder_output = tf.reduce_max(encoder_output, axis=1)

        # project to a scalar
        # shape (batch_size, 1)
        encoder_output = self.output_layer(encoder_output)

        return encoder_output

    def _convert_input_to_one_hot(self, conditional_class, source_tokens):
        if tf.shape(source_tokens).shape == 2:
            source_tokens = tf.one_hot(source_tokens,
                                       depth=self.config.embedding_config.input_dim,
                                       on_value=1.,
                                       off_value=0.)
        if tf.shape(conditional_class).shape == 1:
            conditional_class = tf.one_hot(conditional_class,
                                           depth=self.config.conditional_class_config.input_dim,
                                           on_value=1.,
                                           off_value=0.)
        return conditional_class, source_tokens


def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want
    # our generated examples to look like it
    real_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(real_output), logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros_like(generated_output),
        logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss


def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(
        tf.ones_like(generated_output), generated_output)
