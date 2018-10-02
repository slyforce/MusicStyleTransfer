import mxnet as mx
from mxnet.gluon.rnn import LSTM
from mxnet.gluon.nn import Dense, Embedding

from typing import Tuple

class EncoderConfig:
    def __init__(self,
                 n_layers: int,
                 hidden_dim: int):
        self.n_layers = n_layers
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
                 softmax: bool = True):
        self.output_dim = output_dim
        self.softmax = softmax


class NoiseConfig:
    def __init__(self,
                 noise_dim: int,
                 variance=0.01):
        self.noise_dim = noise_dim
        self.variance = variance


class ModelConfig:
    def __init__(self,
                 encoder_config: EncoderConfig,
                 embedding_config: EmbeddingConfig,
                 conditional_class_config: EmbeddingConfig,
                 output_layer_config: OutputLayerConfig,
                 class_output_layer_config: OutputLayerConfig,
                 noise_config: NoiseConfig):
        self.encoder_config = encoder_config
        self.embedding_config = embedding_config
        self.conditional_class_config = conditional_class_config
        self.output_layer_config = output_layer_config
        self.class_output_layer_config = class_output_layer_config
        self.noise_config = noise_config


class Generator(mx.gluon.HybridBlock):
    def __init__(self,
                 config: ModelConfig,
                 **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.config = config

        with self.name_scope():
            self.encoder = LSTM(config.encoder_config.hidden_dim,
                                config.encoder_config.n_layers,
                                dropout=0.2,
                                input_size=self.config.embedding_config.hidden_dim + self.config.noise_config.noise_dim,
                                layout='NTC')

            self.embeddings = Embedding(
                input_dim=self.config.embedding_config.input_dim,
                output_dim=self.config.embedding_config.hidden_dim,
            )

            self.class_embeddings = Embedding(
                input_dim=self.config.conditional_class_config.input_dim,
                output_dim=self.config.conditional_class_config.hidden_dim,
            )

            self.output_layer = Dense(
                in_units=config.encoder_config.hidden_dim,
                units=self.config.output_layer_config.output_dim,
                flatten=False
            )


    def hybrid_forward(self, F, tokens: mx.nd.NDArray, classes: mx.nd.NDArray, noise: mx.nd.NDArray):
        """
         tokens: the tokens being fed into the encoder  (batch_size, max_sequence_len)
         classes: start token for the decoder sequence (batch_size)
        """

        # get the embeddings for the source tokens and conditional class
        # shape: (batch_size, seq_len, h_dim)
        source_emb = self.embeddings.forward(tokens)
        # shape: (batch_size, h_dim)
        class_emb = self.class_embeddings.forward(classes)

        # shape: (batch_size, 1, h_dim)
        class_emb = F.expand_dims(class_emb, axis=1)

        # place the class embedding in the first position
        # shape: (batch_size, seq_len + 1, h_dim)
        encoder_input = F.concat(source_emb, class_emb, dim=1)

        # apply noise to all the input embeddings
        # shape: (batch_size, seq_len + 1, h_dim + noise_dim)
        encoder_input = F.concat(encoder_input, noise, dim=2)

        # call encoder on the sequence to generate a sequence of encoded tokens
        # of the same length
        # shape: (batch_size, seq_len + 1, h_dim)
        encoder_output = self.encoder.forward(encoder_input)

        # separate class token and embeddings
        # shape: (batch_size, seq_len, h_dim)
        token_emb = F.slice_axis(encoder_output, axis=1, begin=1, end=None)

        # project the encoder outputs to the respective class vocabulary sizes
        # shape: (batch_size, seq_len, num_token_types)
        token_output = F.softmax(self.output_layer.forward(token_emb))

        return token_output

    def create_noise(self, shape: Tuple[int, int]):
        # noise shape (batch_size, seq_len + 1, noise_dim)
        # +1 to sequence length because of initial token
        return mx.nd.random_normal(loc=0, scale=self.config.noise_config.variance,
                                   shape=(shape[0], shape[1]+1, self.config.noise_config.noise_dim))


class Discriminator(mx.gluon.HybridBlock):
    def __init__(self,
                 config: ModelConfig):
        super(Discriminator, self).__init__()
        self.config = config
        with self.name_scope():
            self.encoder = LSTM(config.encoder_config.hidden_dim,
                                config.encoder_config.n_layers,
                                dropout=0.2,
                                input_size=self.config.embedding_config.hidden_dim,
                                layout='NTC')

            self.embeddings = Dense(
                in_units=self.config.embedding_config.input_dim,
                units=self.config.embedding_config.hidden_dim,
                flatten=False
            )

            self.class_embeddings = Dense(
                in_units=self.config.conditional_class_config.input_dim,
                units=self.config.conditional_class_config.hidden_dim,
                flatten=False
            )

            self.output_layer = Dense(
                in_units=config.encoder_config.hidden_dim,
                units=self.config.output_layer_config.output_dim,
                flatten=False,
            )


    def hybrid_forward(self, F, tokens: mx.nd.NDArray, classes: mx.nd.NDArray, seq_lens: mx.nd.NDArray):
        """
        inputs is a list of values
         - tokens: the tokens being fed into the encoder in one hot encoding (batch_size, max_sequence_len, num_token_types)
         - classes: start token for the decoder sequence in one hot encoding (batch_size, num_class_types)
        """

        # get the embeddings for the source tokens and conditional class
        # shape: (batch_size, seq_len, h_dim)
        source_emb = self.embeddings.forward(tokens)
        # shape: (batch_size, 1, h_dim)
        class_emb = F.expand_dims(self.class_embeddings.forward(classes), axis=1)

        # place the class embedding in the first position
        encoder_input = F.concat(class_emb, source_emb, dim=1)

        # call encoder on the sequence to generate a sequence of encoded tokens
        encoder_output = self.encoder.forward(encoder_input)

        # take the last value of the sequence
        # +1 because of the concatenation of start token
        encoder_output = F.SequenceLast(encoder_output,
                                        axis=1,
                                        sequence_length=seq_lens + 1,
                                        use_sequence_length=True)

        # project to a scalar
        # shape (batch_size, 1)
        encoder_output = F.squeeze(self.output_layer.forward(encoder_output))
        return encoder_output

    def convert_to_one_hot(self, source_tokens: mx.nd.NDArray, conditional_class: mx.nd.NDArray):
        """
        source_tokens: (batch_size, max_sequence_len)
        conditional_class: (batch_size)

        Converts tokens and classes to one-hot representations
        """
        source_tokens = mx.nd.one_hot(
            mx.nd.cast(source_tokens, 'int32'),
            depth=self.config.embedding_config.input_dim,
            on_value=1.,
            off_value=0.)

        conditional_class = mx.nd.one_hot(
            mx.nd.cast(conditional_class, 'int32'),
            depth=self.config.conditional_class_config.input_dim,
            on_value=1.,
            off_value=0.)

        return source_tokens, conditional_class

