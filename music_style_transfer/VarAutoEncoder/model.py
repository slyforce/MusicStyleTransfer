import mxnet as mx
from mxnet.gluon.rnn import LSTM
from mxnet.gluon.nn import Dense, Embedding

from typing import Tuple

from .config import Config

class LSTMConfig(Config):
    def __init__(self,
                 n_layers: int,
                 hidden_dim: int,
                 dropout: float):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout


class EncoderDecoderConfig(Config):
    def __init__(self,
                 latent_dimension: int,
                 feature_dimension: int,
                 input_classes: int,
                 encoder_config: LSTMConfig,
                 decoder_config: LSTMConfig):
        super().__init__()
        self.latent_dimension = latent_dimension
        self.feature_dimension = feature_dimension
        self.input_classes = input_classes
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config


class EncoderDecoder(mx.gluon.HybridBlock):
    def __init__(self, config: EncoderDecoderConfig):
        super().__init__()
        self.config = config
        with self.name_scope():

            self.initial_conv = mx.gluon.nn.Conv2D(channels=32,
                                                   kernel_size=(1, self.config.feature_dimension),
                                                   strides=(1, self.config.feature_dimension),
                                                   padding=(0, 0),
                                                   layout='NCHW')
            self.encoder = mx.gluon.nn.HybridSequential()
            for i in range(self.config.encoder_config.n_layers):
                input_size = 32 + self.config.input_classes if i == 0 else config.encoder_config.hidden_dim
                self.encoder.add(LSTM(config.encoder_config.hidden_dim // 2,
                                 1,
                                 bidirectional=True,
                                 dropout=config.encoder_config.dropout,
                                 input_size=input_size,
                                 layout='NTC'))
                self.encoder.add(mx.gluon.nn.LayerNorm())
            self.encoder.add(Dense(self.config.latent_dimension * 2,
                                   flatten=False,
                                   in_units=self.config.encoder_config.hidden_dim))

            self.decoder = mx.gluon.nn.HybridSequential()
            for i in range(self.config.encoder_config.n_layers):
                input_size = self.config.latent_dimension + self.config.input_classes if i == 0 else config.encoder_config.hidden_dim
                self.decoder.add(LSTM(config.decoder_config.hidden_dim // 2,
                                 1,
                                 bidirectional=True,
                                 dropout=config.decoder_config.dropout,
                                 input_size=input_size,
                                 layout='NTC'))
                self.decoder.add(mx.gluon.nn.LayerNorm())
            self.decoder.add(Dense(self.config.feature_dimension * 2,
                                   flatten=False,
                                   in_units=self.config.decoder_config.hidden_dim))

            pass

    def hybrid_forward(self, F, tokens, articulations, enc_classes, dec_classes, noise):
        # melodies: shape: (batch_size, seq_len, feature_dim)
        # articulations: shape: (batch_size, seq_len, feature_dim)
        # enc_classes: shape: (batch_size)
        # dec_classes: shape: (batch_size)
        # noise: shape: (batch_size, seq_len, noise_dim)

        dec_classes, enc_classes = self._preprocess_classes(F, dec_classes, enc_classes, tokens)
        tokens, x = self._preprocess_inputs(F, articulations, tokens)

        # shape: (batch_size, hidden_dim, seq_len)
        x = F.squeeze(self.initial_conv(x), axis=3)

        # shape: (batch_size, seq_len, hidden_dim)
        x = F.swapaxes(x, dim1=1, dim2=2)

        # shape: (batch_size, seq_len, hidden_dim + input_classes)
        x = F.concat(x, self.one_hot(F, enc_classes), dim=2)

        z_values = self.encoder(x)
        [z_means, z_stddev] = F.split(z_values, num_outputs=2, axis=2)

        # shape: (batch_size, seq_len, noise_dim)
        z = z_means + z_stddev * noise
        z = F.concat(z, self.one_hot(F, dec_classes), dim=2)

        # shape: (batch_size, seq_len, feature_dim + 1)
        z = self.decoder(z)

        [out_pitches, out_articulations] = F.split(z, num_outputs=2, axis=2)
        return out_pitches, out_articulations, z_means, z_stddev

    def _preprocess_classes(self, F, dec_classes, enc_classes, tokens):
        # shape: (batch_size, 1)
        enc_classes = F.expand_dims(enc_classes, axis=1)
        # shape: (batch_size, seq_len)
        enc_classes = F.broadcast_like(enc_classes, tokens.sum(axis=2), lhs_axes=(1,), rhs_axes=(1,))
        # shape: (batch_size, 1)
        dec_classes = F.expand_dims(dec_classes, axis=1)
        # shape: (batch_size, seq_len)
        dec_classes = F.broadcast_like(dec_classes, tokens.sum(axis=2), lhs_axes=(1,), rhs_axes=(1,))
        return dec_classes, enc_classes

    def _preprocess_inputs(self, F, articulations, melodies):
        # shape: (batch_size, seq_len, feature_dim)
        articulations = F.expand_dims(articulations, axis=3)
        melodies = F.expand_dims(melodies, axis=3)
        # shape: (batch_size, seq_len, feature_dim, 2)
        melodies = F.concat(melodies, articulations, dim=3)
        # shape: (batch_size, 2, feature_dim, seq_len)
        x = F.swapaxes(melodies, dim1=1, dim2=3)
        # shape: (batch_size, 2, seq_len, feature_dim)
        x = F.swapaxes(x, dim1=2, dim2=3)
        return melodies, x

    def one_hot(self, F, input):
        return F.one_hot(
            F.cast(input, 'int32'),
            depth=self.config.input_classes,
            on_value=1.,
            off_value=0.)




