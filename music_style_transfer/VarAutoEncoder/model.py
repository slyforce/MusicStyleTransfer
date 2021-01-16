import mxnet as mx
from mxnet.gluon.rnn import LSTM
from mxnet.gluon.nn import Dense, Embedding
import sys

from .config import Config
from .transformer import TransformerEncoder, TransformerConfig, TransformerDecoder
from music_style_transfer.MIDIUtil.defaults import *


class LSTMConfig(Config):
    def __init__(self,
                 n_layers: int,
                 hidden_dim: int,
                 dropout: float):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout


class DecoderConfig(Config):
    def __init__(self,
                 transformer_config: TransformerConfig,
                 latent_dim: int,
                 num_classes: int,
                 output_dim: int):
        super().__init__()
        self.transformer_config = transformer_config
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.output_dim = output_dim


class EncoderConfig(Config):
    def __init__(self,
                 transformer_config: TransformerConfig,
                 latent_dim: int,
                 num_classes: int,
                 input_dim: int):
        super().__init__()
        self.transformer_config = transformer_config
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.input_dim = input_dim


class ModelConfig(Config):
    def __init__(self,
                 encoder_config: EncoderConfig,
                 decoder_config: DecoderConfig):
        super().__init__()
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config


class Encoder(mx.gluon.HybridBlock):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        with self.name_scope():
            self.class2hid = Embedding(input_dim=self.config.num_classes,
                                       output_dim=self.config.transformer_config.model_size)

            self.encoder_embedding = Embedding(input_dim=self.config.input_dim,
                                               output_dim=self.config.transformer_config.model_size)

            self.encoder = TransformerEncoder(self.config.transformer_config)

            self.latent_proj = Dense(in_units=self.config.transformer_config.model_size,
                                     units=self.config.latent_dim * 2)

    def hybrid_forward(self, F, tokens, seq_length, classes):
        """
        :param tokens: (batch_size, max_seq_len)
        :param seq_length: (batch_size,)
        :param classes: (batch_size,)
        :return:
        """

        mask = mx.nd.where(tokens != 0,
                           mx.nd.ones_like(tokens),
                           mx.nd.zeros_like(tokens))

        # shape: (batch_size, max_seq_len, hidden_dim)
        token_emb = self.encoder_embedding(tokens)

        # shape: (batch_size, hidden_dim)
        class_emb = self.class2hid(classes)

        encoder_input = F.broadcast_add(F.expand_dims(class_emb, axis=1), token_emb)

        # shape: (batch_size, max_seq_len, hidden_dim)
        encoder_output = self.encoder(encoder_input, mask)

        # shape: (batch_size, hidden_dim)
        last_states = encoder_output[:,0,:]

        # shape: (batch_size, 2 * latent_dim)
        latent_state = self.latent_proj(last_states)

        # shape: (batch_size, latent_dim)
        [means, stddevs] = F.split(latent_state, num_outputs=2, axis=1)
        return means, stddevs


class DecoderState:
    """
    To be used in inference to keep track of decoder states.
    The token sequence or Pre-computed attention values e.g.
    """
    def __init__(self,
                 batch_size: int,
                 num_cache_layers: int,
                 initial_state: mx.nd.NDArray):
        self.reset(batch_size, num_cache_layers)
        self.initial_state = initial_state

    def advance_state(self, tokens):
        self.tokens = mx.nd.concat(self.tokens, tokens, dim=1)
        self.t += 1

    def reset(self,
              batch_size: int,
              num_cache_layers: int):
        self.tokens = mx.nd.full(shape=(batch_size, 1), val=SOS_ID)
        self.t = 1
        self.caches = [{} for _ in range(num_cache_layers)]


class LSTMDecoder(mx.gluon.HybridBlock):
    def __init__(self, config: DecoderConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self._define_parameters()

    def _define_parameters(self):
        with self.name_scope():
            self.latent2hid = Dense(in_units=self.config.latent_dim,
                                    units=self.config.lstm_config.hidden_dim * 2)

            self.class2hid = Embedding(input_dim=self.config.num_classes,
                                       output_dim=self.config.lstm_config.hidden_dim * 2)

            self.embedding = Embedding(input_dim=self.config.output_dim,
                                       output_dim=self.config.lstm_config.hidden_dim)

            self.decoder = LSTM(self.config.lstm_config.hidden_dim,  # // 2,
                                self.config.lstm_config.n_layers,
                                bidirectional=False,
                                dropout=self.config.lstm_config.dropout,
                                input_size=self.config.lstm_config.hidden_dim,
                                layout='NTC')

            self.output_layer = Dense(in_units=self.config.lstm_config.hidden_dim,
                                      units=self.config.output_dim,
                                      flatten=False)

    def get_initial_state(self, F, classes, hidden_state):
        transform = self.latent2hid(hidden_state) + self.class2hid(classes)
        transform = F.repeat(mx.nd.expand_dims(transform, axis=0),
                             axis=0,
                             repeats=self.config.lstm_config.n_layers)
        init_states = F.split(transform, axis=2, squeeze_axis=False, num_outputs=2)

        # shape (in each) (num_layers, batch_size, lstm_dim)
        return init_states

    def hybrid_forward(self, F,  tokens, seq_length, hidden_states, classes):
        return self.forward_train(F, tokens, seq_length, hidden_states, classes)

    def forward_train(self, F, tokens, seq_length, hidden_states, classes):
        init_state = self.get_initial_state(F, classes, hidden_states)

        # shape: (batch_size, seq_len, feature_dim)
        token_embeddings = self.embedding(tokens)

        # shape: (batch_size, seq_len, feature_dim)
        lstm_outputs = self.decoder(token_embeddings, init_state)[0]

        # shape: (batch_size, seq_len, output_dim)
        probs = F.softmax(self.output_layer(lstm_outputs), axis=-1)
        return probs

    def forward_inference(self, tokens, previous_state, classes, step):
        # performs one step of the recurrency

        # shape: (batch-size, 1)
        tokens = mx.nd.expand_dims(tokens, axis=1)

        # shape: (batch_size, 1, feature_dim)
        token_embeddings = self.embedding(tokens)

        # shape: (batch_size, 1, feature_dim)
        lstm_outputs, next_states = self.decoder(token_embeddings, previous_state)

        # shape: (batch_size, 1, output_dim)
        probs = mx.nd.softmax(self.output_layer(lstm_outputs), axis=-1)

        # shape: (batch_size, 1, output_dim)
        probs = mx.nd.reshape(probs, (0, -1))

        return probs, next_states


class Decoder(mx.gluon.HybridBlock):
    def __init__(self, config: DecoderConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self._define_parameters()

    def _define_parameters(self):
        with self.name_scope():
            self.latent2hid = Dense(in_units=self.config.latent_dim,
                                    units=self.config.transformer_config.model_size)

            self.class2hid = Embedding(input_dim=self.config.num_classes,
                                       output_dim=self.config.transformer_config.model_size)

            self.embedding = Embedding(input_dim=self.config.output_dim,
                                       output_dim=self.config.transformer_config.model_size)

            self.decoder = TransformerDecoder(self.config.transformer_config)

            self.output_layer = Dense(in_units=self.config.transformer_config.model_size,
                                      units=self.config.output_dim,
                                      flatten=False)

    def get_initial_state(self, F, classes, hidden_state):
        # shape [B, 1, D]
        res = self.latent2hid(hidden_state) + self.class2hid(classes)
        return mx.nd.expand_dims(res, axis=1)

    def hybrid_forward(self, F,  tokens, seq_length, hidden_states, classes):
        return self.forward_train(F, tokens, seq_length, hidden_states, classes)

    def forward_train(self, F, tokens: mx.nd.NDArray, seq_length, hidden_states, classes):
        batch_size, seq_len = tokens.shape

        # shape: (batch_size, seq_len, feature_dim)
        token_embeddings = self.embedding(tokens)

        # shape: (batch_size, seq_len + 1, feature_dim)
        input_states = mx.nd.concat(self.get_initial_state(F, classes, hidden_states),
                                    token_embeddings, dim=1)
        decoder_mask = mx.nd.SequenceMask(mx.nd.ones(shape=(batch_size, seq_len+1), dtype="float32"),
                                          use_sequence_length=True, sequence_length=seq_length+1, axis=1)

        # shape: (batch_size, seq_len + 1, feature_dim)
        desired_embeddings = self.decoder.forward_train(F, input_states, decoder_mask)

        # prediction of the first token is irrelevant as it's a sentence begin token
        desired_embeddings = desired_embeddings[:,1:,:]

        # shape: (batch_size, seq_len + 1, output_dim)
        probs = F.softmax(self.output_layer(desired_embeddings), axis=-1)
        return probs

    def forward_inference(self, state):
        # shape: (batch_size, seq_len, feature_dim)
        token_embeddings = self.embedding(state.tokens[:,-1]).expand_dims(axis=1)

        if state.t == 0:
            # shape: (batch_size, seq_len + 1, feature_dim)
            input_states = mx.nd.concat(state.initial_state,
                                        token_embeddings, dim=1)
        else:
            input_states = token_embeddings

        desired_embedding = self.decoder.forward_inference(state, input_states)
        probs = mx.nd.softmax(self.output_layer(desired_embedding), axis=-1)
        return probs


class Model(mx.gluon.HybridBlock):

    def __init__(self,
                 config: ModelConfig,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Creating a model with the following configuration:")
        config.output_to_stream(sys.stdout)
        with self.name_scope():
            self.decoder = Decoder(config.decoder_config)
            self.encoder = Encoder(config.encoder_config)

    def hybrid_forward(self, F, tokens, seq_lens, classes):
        # encode the inputs
        means, vars = self.encoder(tokens, seq_lens, classes)

        # sample in the hidden space
        z_sampled = means + mx.nd.random_normal(loc=0, scale=1., shape=means.shape) * vars

        # now decode the sequence
        probs = self.decoder.forward_train(F, tokens, seq_lens, z_sampled, classes)
        return probs, means, vars




