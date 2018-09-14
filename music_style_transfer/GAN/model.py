import tensorflow as tf
tf.enable_eager_execution()


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
                 hidden_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim


class OutputLayerConfig:
    def __init__(self,
                 output_dim: int,
                 softmax: bool=True):
        self.output_dim = output_dim
        self.softmax = softmax


from opennmt.encoders.self_attention_encoder import SelfAttentionEncoder


class Encoder:
    def __init__(self,
                 config: TransformerConfig):
        self.config = config

        self.encoder = SelfAttentionEncoder(
            num_layers=config.n_layers,
            num_units=config.model_dim,
            ffn_inner_dim=config.hidden_dim
        )

    def encode(self, input, seq_len, mode='train'):
        """
        :param input: shape (batch_size, seq_len)
        :param seq_len: shape (batch_size,)
        :param mode: 'train' or 'infer'
        :return:
        """
        return self.encoder.encode(input, seq_len, mode)


class Embedding:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.embedding = tf.keras.layers.Embedding(
            input_dim=config.input_dim,
            output_dim=config.hidden_dim,
            mask_zero=True)

    def encode(self, input):
        return self.embedding.call(input)


class BagOfEmbeddings:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.embedding = tf.keras.layers.Dense(units=config.hidden_dim,
                                               use_bias=False)

    def encode(self, input):
        if len(tf.shape(input)) == 2:
            input = tf.one_hot(input,
                               depth=self.config.input_dim,
                               on_value=1.,
                               off_value=0.)

        return self.embedding.call(input)


class OutputLayer:
    def __init__(self, config: OutputLayerConfig):
        self.config = config

        self.layer = tf.keras.layers.Dense(units=self.config.output_dim)

        if not self.config.softmax:
            self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        x = self.layer.call(x)
        if not self.config.softmax:
            x = self.softmax(x)
        return x


class GeneratorConfig:
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
                 config: GeneratorConfig):
        super(Generator, self).__init__()
        self.config = config

        self.encoder = Encoder(self.config.encoder_config)
        self.embeddings = Embedding(self.config.embedding_config)
        self.class_embeddings = Embedding(self.config.conditional_class_config)

        self.output_layer = OutputLayer(self.config.output_layer_config)
        self.class_output_layer = OutputLayer(self.config.output_layer_config)

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
        [source_tokens, sequence_length, conditional_class] = inputs

        # get the embeddings for the source tokens and conditional class
        # shape: (batch_size, seq_len, h_dim)
        source_emb = self.embeddings.encode(source_tokens)
        # shape: (batch_size, h_dim)
        class_emb = self.class_embeddings.encode(conditional_class)

        # TODO: this should be a layer (lambda layer?)
        # shape: (batch_size, 1, h_dim)
        class_emb = tf.expand_dims(class_emb, axis=1)

        # place the class embedding in the first position
        encoder_input = tf.concat([class_emb, source_emb], concat_dim=1)
        extended_sequence_length = sequence_length + \
            tf.ones_like(sequence_length)

        # call encoder on the sequence to generate a sequence of encoded tokens
        # of the same length
        encoder_output = self.encoder.encode(
            encoder_input,
            extended_sequence_length,
            'train' if training else 'infer')

        # separate class token and embeddings
        class_emb = encoder_output[:, 0, :]
        token_emb = encoder_output[:, 1:, :]

        # project the encoder outputs to the respective class vocabulary sizes
        class_output = self.class_output_layer.call(class_emb)
        token_output = self.output_layer.call(token_emb)

        return token_output, sequence_length, class_output


class Discriminator(tf.keras.Model):
    def __init__(self,
                 config: GeneratorConfig):
        super(Discriminator, self).__init__()
        self.config = config

        self.encoder = Encoder(self.config.encoder_config)
        self.embeddings = BagOfEmbeddings(self.config.embedding_config)
        self.class_embeddings = BagOfEmbeddings(
            self.config.conditional_class_config)

        self.output_layer = OutputLayer(self.config.output_layer_config)

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
        [source_tokens, sequence_length, conditional_class] = inputs

        # get the embeddings for the source tokens and conditional class
        # shape: (batch_size, seq_len, h_dim)
        source_emb = self.embeddings.encode(source_tokens)
        # shape: (batch_size, h_dim)
        class_emb = self.class_embeddings.encode(conditional_class)

        # TODO: this should be a layer (lambda layer?)
        # shape: (batch_size, 1, h_dim)
        class_emb = tf.expand_dims(class_emb, axis=1)

        # place the class embedding in the first position
        encoder_input = tf.concat([class_emb, source_emb], concat_dim=1)
        extended_sequence_length = sequence_length + \
            tf.ones_like(sequence_length)

        # call encoder on the sequence to generate a sequence of encoded tokens
        # of the same length
        encoder_output = self.encoder.encode(
            encoder_input,
            extended_sequence_length,
            'train' if training else 'infer')

        # take the maximum values over the time axis
        # shape (batch_size, hidden_dim)
        encoder_output = tf.reduce_max(encoder_output, axis=1)

        # project to a scalar
        # shape (batch_size, 1)
        encoder_output = self.output_layer.call(encoder_output)

        return encoder_output


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
