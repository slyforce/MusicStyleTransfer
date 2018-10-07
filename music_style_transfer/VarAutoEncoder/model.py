import mxnet as mx
from mxnet.gluon.rnn import LSTM
from mxnet.gluon.nn import Dense, Embedding

from typing import Tuple

class LSTMConfig:
    def __init__(self,
                 n_layers: int,
                 hidden_dim: int,
                 dropout: float):
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout


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


class EncoderConfig:
    def __init__(self,
                 encoder_config: LSTMConfig,
                 embedding_config: EmbeddingConfig,
                 latent_dimension: int,
                 input_classes: int):
        self.encoder_config = encoder_config
        self.latent_dimension = latent_dimension
        self.embedding_config = embedding_config
        self.input_classes = input_classes

class Encoder(mx.gluon.HybridBlock):

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        with self.name_scope():
            self.embeddings = Embedding(
                input_dim=self.config.embedding_config.input_dim,
                output_dim=self.config.embedding_config.hidden_dim
            )

            self.encoder = LSTM(config.encoder_config.hidden_dim,
                                config.encoder_config.n_layers,
                                dropout=config.encoder_config.dropout,
                                input_size=self.config.embedding_config.hidden_dim + self.config.input_classes,
                                layout='NTC')

            self.enc2latent = Dense(units=2 * self.config.latent_dimension,
                                    flatten=False,
                                    in_units=config.encoder_config.hidden_dim)

    def one_hot(self, F, input):
        return  F.one_hot(
                F.cast(input, 'int32'),
                depth=self.config.input_classes,
                on_value=1.,
                off_value=0.)

    def hybrid_forward(self, F, sequences, classes):

        classes = F.expand_dims(classes, axis=1)
        classes = F.broadcast_like(classes, sequences, lhs_axes=(1,), rhs_axes=(1,))

        # shape: (batch_size, seq_len, emb_dim)
        embeddings = self.embeddings(sequences)

        # shape: (batch_size, seq_len, emb_dim + num_classes)
        embeddings = F.concat(*[embeddings, self.one_hot(F, classes)], dim=2)

        # shape: (batch_size, seq_len, h_dim)
        encoder_output = self.encoder(embeddings)

        # shape: (batch_size, seq_len, 2 * z_dim)
        z = self.enc2latent(encoder_output)

        # shape of each: (batch_size, seq_len, z_dim)
        [z_means, z_vars] = F.split(z, num_outputs=2, axis=2)
        return z_means, z_vars

class DecoderConfig:
    def __init__(self,
                 encoder_config: LSTMConfig,
                 output_layer_config: OutputLayerConfig,
                 latent_dimension: int,
                 input_classes: int):
        self.encoder_config = encoder_config
        self.latent_dimension = latent_dimension
        self.output_layer_config = output_layer_config
        self.input_classes = input_classes

class Decoder(mx.gluon.HybridBlock):

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        with self.name_scope():
            self.encoder = LSTM(config.encoder_config.hidden_dim,
                                config.encoder_config.n_layers,
                                dropout=config.encoder_config.dropout,
                                input_size=self.config.latent_dimension + self.config.input_classes,
                                layout='NTC')

            self.output_layer = Dense(units=self.config.output_layer_config.output_dim,
                                      flatten=False,
                                      in_units=config.encoder_config.hidden_dim)

    def one_hot(self, F, input):
        return F.one_hot(
            F.cast(input, 'int32'),
            depth=self.config.input_classes,
            on_value=1.,
            off_value=0.)

    def hybrid_forward(self, F, z_sequences, classes):
        classes = F.expand_dims(classes, axis=1)
        classes = F.broadcast_like(classes, z_sequences.sum(axis=2), lhs_axes=(1,), rhs_axes=(1,))

        # shape: (batch_size, seq_len, z_dim + num_classes)
        enc_input = F.concat(*[z_sequences, self.one_hot(F, classes)], dim=2)

        # shape: (batch_size, seq_len, h_dim)
        enc_output = self.encoder(enc_input)

        # shape: (batch_size, seq_len, num_tokens)
        logits = self.output_layer(enc_output)
        return logits
