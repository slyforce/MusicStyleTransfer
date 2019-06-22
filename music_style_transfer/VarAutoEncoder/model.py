import mxnet as mx
from mxnet.gluon.rnn import LSTM
from mxnet.gluon.nn import Dense, Embedding

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


class DecoderConfig(Config):
    def __init__(self,
                 lstm_config: LSTMConfig,
                 latent_dim: int,
                 num_classes: int,
                 output_dim: int):
        super().__init__()
        self.lstm_config = lstm_config
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.output_dim = output_dim


class EncoderConfig(Config):
    def __init__(self,
                 lstm_config: LSTMConfig,
                 latent_dim: int,
                 num_classes: int,
                 input_dim: int):
        super().__init__()
        self.lstm_config = lstm_config
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
                                       output_dim=self.config.lstm_config.hidden_dim)

            self.encoder_embedding = Embedding(input_dim=self.config.input_dim,
                                               output_dim=self.config.lstm_config.hidden_dim)

            self.encoder = LSTM(self.config.lstm_config.hidden_dim, # // 2,
                                self.config.lstm_config.n_layers,
                                bidirectional=False,
                                dropout=self.config.lstm_config.dropout,
                                input_size=self.config.lstm_config.hidden_dim,
                                layout='NTC')

            self.latent_proj = Dense(in_units=self.config.lstm_config.hidden_dim,
                                     units=self.config.latent_dim * 2)

    def hybrid_forward(self, F, tokens, seq_length, classes):
        """
        :param tokens: (batch_size, max_seq_len)
        :param seq_length: (batch_size,)
        :param classes: (batch_size,)
        :return:
        """

        # shape: (batch_size, max_seq_len, hidden_dim)
        token_emb = self.encoder_embedding(tokens)

        # shape: (batch_size, hidden_dim)
        class_emb = self.class2hid(classes)

        encoder_input = F.broadcast_add(F.expand_dims(class_emb, axis=1), token_emb)

        # shape: (batch_size, max_seq_len, hidden_dim)
        encoder_output = self.encoder(encoder_input)
        
        # shape: (batch_size, hidden_dim)
        last_states = F.SequenceLast(encoder_output,
                                     axis=1,
                                     sequence_length=seq_length,
                                     use_sequence_length=True)
        
        # shape: (batch_size, 2 * latent_dim)
        latent_state = self.latent_proj(last_states)

        # shape: (batch_size, latent_dim)
        [means, stddevs] = F.split(latent_state, num_outputs=2, axis=1)
        return means, stddevs


class Decoder(mx.gluon.HybridBlock):
    def __init__(self, config: DecoderConfig):
        super().__init__()
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


class TrainingDecoder(Decoder):
    def __init__(self, config: DecoderConfig):
        super().__init__(config)

    def hybrid_forward(self, F, tokens, seq_length, hidden_states, classes):
        init_state = self.get_initial_state(F, classes, hidden_states)

        # shape: (batch_size, seq_len, feature_dim)
        token_embeddings = self.embedding(tokens)

        # shape: (batch_size, seq_len, feature_dim)
        lstm_outputs = self.decoder(token_embeddings, init_state)[0]

        # shape: (batch_size, seq_len, output_dim)
        probs = F.softmax(self.output_layer(lstm_outputs), axis=-1)
        return probs


class InferenceDecoder(Decoder):
    def __init__(self, config: DecoderConfig):
        super().__init__(config)

    def hybrid_forward(self, F, tokens, previous_state, classes, step):
        # performs one step of the recurrency
        raise NotImplemented


class TrainingModel(mx.gluon.HybridBlock):

    def __init__(self,
                 config: ModelConfig):
        super().__init__()
        self.decoder = TrainingDecoder(config.decoder_config)
        self.encoder = Encoder(config.encoder_config)

    def hybrid_forward(self, F, tokens, seq_lens, classes):

        # encode the inputs
        means, vars = self.encoder(tokens, seq_lens, classes)

        # sample in the hidden space
        z_sampled = means + mx.nd.random_normal(loc=0, scale=1., shape=means.shape) * vars

        # now decode the sequence
        probs = self.decoder(tokens, seq_lens, z_sampled, classes)
        return probs, means, vars




