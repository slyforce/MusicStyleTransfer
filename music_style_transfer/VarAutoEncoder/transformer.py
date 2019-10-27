import mxnet as mx
import numpy as np
from .config import Config
from music_style_transfer.MIDIUtil.defaults import *

from typing import Optional

class TransformerConfig(Config):
    def __init__(self,
                 model_size: int,
                 dropout: float,
                 num_layers: int,
                 num_heads: int,
                 vocab_size: Optional[int] = None
                 ):
        super().__init__()
        self.model_size = model_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size


class DualFeedForward(mx.gluon.HybridBlock):
    def __init__(self,
                 input_dimension: int,
                 dimension: int,
                 dropout: float):
        super().__init__()
        self.input_dimension = input_dimension
        self.dimension = dimension
        self.dropout = dropout

        with self.name_scope():
            self.dropout = mx.gluon.nn.Dropout(rate=self.dropout)
            self.ff1 = mx.gluon.nn.Dense(units=self.dimension,
                                         in_units=self.input_dimension,
                                         activation="relu", flatten=False)
            self.ff2 = mx.gluon.nn.Dense(units=self.input_dimension,
                                         in_units=self.dimension, flatten=False)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.ff1(x)
        x = self.dropout(x)
        x = self.ff2(x)
        return x


class MultiHeadDotAttention(mx.gluon.HybridBlock):
    def __init__(self,
                 attention_dim: int,
                 model_dim: int,
                 num_heads: int,
                 is_self_attention: bool,
                 mask_future_timesteps: bool):
        super().__init__()
        self.attention_dim = attention_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.is_self_attention = is_self_attention
        self.mask_future_timesteps = mask_future_timesteps

        with self.name_scope():
            self.W_k = mx.gluon.nn.Dense(units=self.model_dim, in_units=self.model_dim, flatten=False)
            self.W_q = mx.gluon.nn.Dense(units=self.model_dim, in_units=self.model_dim, flatten=False)
            self.W_v = mx.gluon.nn.Dense(units=self.model_dim, in_units=self.model_dim, flatten=False)
            self.W_proj = mx.gluon.nn.Dense(units=self.model_dim, in_units=self.model_dim, flatten=False)

    def hybrid_forward(self,
                       F,
                       keys_values: mx.nd.ndarray,
                       queries: mx.nd.ndarray,
                       keys_values_mask: mx.nd.ndarray):
        B, T_K, _ = keys_values.shape
        _, T_Q, _ = queries.shape

        K = self.W_k(keys_values).reshape(shape=[B, T_K, self.num_heads, -1]).swapaxes(1, 2)
        Q = self.W_q(queries).reshape(shape=[B, T_Q, self.num_heads, -1]).swapaxes(1, 2)
        V = self.W_v(keys_values).reshape(shape=[B, T_K, self.num_heads, -1]).swapaxes(1, 2)

        att_logits = mx.nd.linalg_gemm2(K, Q, transpose_b=True)
        att_logits = att_logits / mx.nd.sqrt(mx.nd.full(shape=1, val=self.attention_dim))
        att_logits = self._mask_logits(att_logits, keys_values_mask)
        att_probs = mx.nd.softmax(att_logits)

        attended_values = mx.nd.linalg_gemm2(att_probs, V, transpose_a=True)
        attended_values = attended_values.swapaxes(1, 2).reshape([B, T_Q, -1])

        attended_values_proj = self.W_proj(attended_values)
        return attended_values_proj

    def _mask_logits(self, att_logits: mx.nd.ndarray, padding_mask: mx.nd.ndarray):
        # att_logits [batch_size, num_heads , T_K, T_Q]
        # padding mask [batch_size, T_K]
        B, _, T_K, T_Q = att_logits.shape

        mask_value = -1e9
        mask = mx.nd.where(padding_mask > 0.,
                           mx.nd.zeros_like(padding_mask),
                           mx.nd.ones_like(padding_mask) * mask_value)
        mask = mx.nd.repeat(mask.expand_dims(2), axis=2, repeats=T_Q) # [batch_size, T_K, T_Q]
        mask = mask.expand_dims(1) # [batch_size, 1, T_K, T_Q]

        if self.mask_future_timesteps:
            assert self.is_self_attention

            masked_entries = mx.nd.full(shape=(T_Q * (T_Q-1) / 2), val=mask_value)
            masked_matrix = mx.nd.linalg.maketrian(masked_entries, offset=0) # Todo: upper vs lower??
            mask = mx.nd.broadcast_add(mask, masked_matrix)

        output = mx.nd.broadcast_add(att_logits, mask)
        return output



class TransformerEncoderLayer(mx.gluon.HybridBlock):
    def __init__(self,
                 config: TransformerConfig):
        super().__init__()
        self.config = config
        assert self.config.model_size % self.config.num_heads == 0

        with self.name_scope():
            self.self_attention = MultiHeadDotAttention(attention_dim=int(self.config.model_size / self.config.num_heads),
                                                        num_heads=self.config.num_heads,
                                                        model_dim=self.config.model_size,
                                                        is_self_attention=True,
                                                        mask_future_timesteps=False)
            self.ln1 = mx.gluon.nn.LayerNorm(in_channels=self.config.model_size)

            self.ff = DualFeedForward(input_dimension=self.config.model_size,
                                      dimension=self.config.model_size * 4,
                                      dropout=self.config.dropout)
            self.ln2 = mx.gluon.nn.LayerNorm(in_channels=self.config.model_size)

            self.dropout = mx.gluon.nn.Dropout(rate=self.config.dropout)

    def hybrid_forward(self, F, input, input_mask):
        x = input
        x_att = self.self_attention(input, input, input_mask)
        x_att = self.ln1(self.dropout(x_att))
        x = x + x_att

        x_att = self.ff(input)
        x_att = self.ln2(self.dropout(x_att))
        x = x + x_att

        return x


class TransformerEncoder(mx.gluon.HybridBlock):
    def __init__(self,
                 config: TransformerConfig,
                 maximum_sequence_length: Optional[int] = None):
        super().__init__()
        self.config = config
        self.maximum_sequence_length = maximum_sequence_length

        with self.name_scope():
            self.pos_embeddings = self._get_positional_encodings()
            self.layers = [TransformerEncoderLayer(config)
                           for _ in range(self.config.num_layers)]

            # filthy hack to use lists of models
            for i, layer in enumerate(self.layers):
                self.__setattr__("layer{}".format(i), layer)

    def hybrid_forward(self, F, inputs, input_mask):
        seq_len = inputs.shape[1]

        inputs = mx.nd.sqrt(mx.nd.full(shape=(1,), val=self.config.model_size)) * inputs + self.pos_embeddings[:seq_len]
        for layer in self.layers:
            inputs = layer.forward(inputs, input_mask)

        return inputs

    def _get_positional_encodings(self):
        """Init the sinusoid position encoding table """
        if self.maximum_sequence_length is None:
            print("Setting maximum sequence length to 10000 for transformer positional encondings.")
            self.maximum_sequence_length = 10000

        position_enc = np.arange(self.maximum_sequence_length).reshape((-1, 1)) \
                       / (np.power(10000, (2. / self.config.model_size) * np.arange(self.config.model_size).reshape((1, -1))))
        # Apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        return mx.nd.array(position_enc)
