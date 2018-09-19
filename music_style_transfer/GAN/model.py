import mxnet as mx
from mxnet.gluon.rnn import LSTM
from mxnet.gluon.nn import Dense, Embedding


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


class ModelConfig:
    def __init__(self,
                 encoder_config: EncoderConfig,
                 embedding_config: EmbeddingConfig,
                 conditional_class_config: EmbeddingConfig,
                 output_layer_config: OutputLayerConfig,
                 class_output_layer_config: OutputLayerConfig):
        self.encoder_config = encoder_config
        self.embedding_config = embedding_config
        self.conditional_class_config = conditional_class_config
        self.output_layer_config = output_layer_config
        self.class_output_layer_config = class_output_layer_config


class Generator(mx.gluon.HybridBlock):
    def __init__(self,
                 config: ModelConfig,
                 **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.config = config

        with self.name_scope():
            self.encoder = LSTM(config.encoder_config.hidden_dim,
                                config.encoder_config.n_layers,
                                dropout=0.0,
                                input_size=self.config.embedding_config.hidden_dim)

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

            self.class_output_layer = Dense(
                in_units=config.encoder_config.hidden_dim,
                units=self.config.class_output_layer_config.output_dim,
                flatten=False
            )

    def hybrid_forward(self, F, tokens, classes):
        """
        inputs is a list of values
         - source tokens: the tokens being fed into the encoder (batch_size, max_sequence_len)
         - conditional class: start token for the decoder sequence (batch_size)
         - sequence length: length of the input sequence (batch_size)
        :param inputs:
        :param training:
        :param mask:
        :return:
        """

        # get the embeddings for the source tokens and conditional class
        # shape: (batch_size, seq_len, h_dim)
        source_emb = self.embeddings.forward(tokens)
        # shape: (batch_size, h_dim)
        class_emb = self.class_embeddings.forward(classes)

        # shape: (batch_size, 1, h_dim)
        class_emb = F.expand_dims(class_emb, axis=1)

        # place the class embedding in the first position
        encoder_input = F.concat(source_emb, class_emb, dim=1)

        # call encoder on the sequence to generate a sequence of encoded tokens
        # of the same length
        encoder_output = self.encoder.forward(encoder_input)

        # separate class token and embeddings
        class_emb = F.slice_axis(encoder_output, axis=1, begin=0, end=1)
        token_emb = F.slice_axis(encoder_output, axis=1, begin=1, end=None)

        # project the encoder outputs to the respective class vocabulary sizes
        token_output = F.softmax(self.output_layer.forward(token_emb))
        class_output = F.squeeze(F.softmax(self.class_output_layer.forward(class_emb)))

        return token_output, class_output


class Discriminator(mx.gluon.HybridBlock):
    def __init__(self,
                 config: ModelConfig):
        super(Discriminator, self).__init__()
        self.config = config
        with self.name_scope():
            self.encoder = LSTM(config.encoder_config.hidden_dim,
                                config.encoder_config.n_layers,
                                dropout=0.0,
                                input_size=self.config.embedding_config.hidden_dim)

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
                flatten=False
            )


    def hybrid_forward(self, F, tokens, classes, seq_lens):
        """
        inputs is a list of values
         - source tokens: the tokens being fed into the encoder (batch_size, max_sequence_len)
         - conditional class: start token for the decoder sequence (batch_size)
        """

        # get the embeddings for the source tokens and conditional class
        # shape: (batch_size, seq_len, h_dim)
        source_emb = self.embeddings.forward(tokens)
        # shape: (batch_size, 1, h_dim)
        class_emb = F.expand_dims(
            self.class_embeddings.forward(classes), axis=1)

        # place the class embedding in the first position
        encoder_input = F.concat(class_emb, source_emb, dim=1)

        # call encoder on the sequence to generate a sequence of encoded tokens
        # of the same length
        encoder_output = self.encoder.forward(encoder_input)

        # mask the values at this point
        # set to a large negative value
        encoder_output = F.SequenceMask(encoder_output,
                                        axis=1,
                                        sequence_length=seq_lens,
                                        use_sequence_length=True,
                                        value=-10000.)


        # take the maximum values over the time axis
        # shape (batch_size, hidden_dim)
        encoder_output = F.max(encoder_output, axis=1)

        # project to a scalar
        # shape (batch_size, 1)
        encoder_output = F.sigmoid(self.output_layer.forward(encoder_output))

        return encoder_output

    def convert_to_one_hot(self, source_tokens, conditional_class):

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

