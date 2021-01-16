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
        self.is_incremental = is_self_attention and mask_future_timesteps # basically decoder self attention

        with self.name_scope():
            self.W_k = mx.gluon.nn.Dense(units=self.model_dim, in_units=self.model_dim, flatten=False)
            self.W_q = mx.gluon.nn.Dense(units=self.model_dim, in_units=self.model_dim, flatten=False)
            self.W_v = mx.gluon.nn.Dense(units=self.model_dim, in_units=self.model_dim, flatten=False)
            self.W_proj = mx.gluon.nn.Dense(units=self.model_dim, in_units=self.model_dim, flatten=False)

    def compute_with_cache(self, x, module, key, cache):
        if cache is None:
            return module(x)
        if key in cache and self.is_incremental:
            cache[key] = mx.nd.concat([cache[key], module(x)], axis=1)
        elif key not in cache:
            cache[key] = module(x)
        return cache[key]

    def hybrid_forward(self,
                       F,
                       keys_values: mx.nd.ndarray,
                       queries: mx.nd.ndarray,
                       keys_values_mask: mx.nd.ndarray,
                       cache = None):
        B, T_K, _ = keys_values.shape
        _, T_Q, _ = queries.shape

        K = self.compute_with_cache(keys_values, self.W_k, "keys", cache)
        V = self.compute_with_cache(keys_values, self.W_v, "values", cache)

        K = K.reshape(shape=[B, T_K, self.num_heads, -1]).swapaxes(1, 2)
        V = V.reshape(shape=[B, T_K, self.num_heads, -1]).swapaxes(1, 2)
        Q = self.W_q(queries).reshape(shape=[B, T_Q, self.num_heads, -1]).swapaxes(1, 2)

        print(K.shape, V.shape, Q.shape)
        att_logits = mx.nd.linalg_gemm2(K, Q, transpose_b=True)
        print(att_logits.shape)
        att_logits = att_logits / mx.nd.sqrt(mx.nd.full(shape=1, val=self.attention_dim))
        att_logits = self._mask_logits(att_logits, keys_values_mask)
        att_probs = mx.nd.softmax(att_logits)

        attended_values = mx.nd.linalg_gemm2(att_probs, V, transpose_a=True)
        attended_values = attended_values.swapaxes(1, 2).reshape([B, T_Q, -1])
        return self.W_proj(attended_values)

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

    def hybrid_forward(self, F, x, input_mask):

        # self-attention with no caching
        x_att = self.self_attention(x, x, input_mask, None)
        x = self.ln1(x + self.dropout(x_att))

        x_att = self.ff(x)
        x = self.ln2(x + self.dropout(x_att))
        return x


class TransformerDecoderLayer(mx.gluon.HybridBlock):
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
            self.ln3 = mx.gluon.nn.LayerNorm(in_channels=self.config.model_size)

            self.dropout = mx.gluon.nn.Dropout(rate=self.config.dropout)

    def get_cache(self, cache, key):
        if cache is None:
            return None

        if key not in cache:
            cache[key] = {}
        return cache[key]

    def hybrid_forward(self, F,
                       decoder_in,  decoder_mask,
                       caches,
                       ):
        x_att = self.self_attention(decoder_in, decoder_in, decoder_mask, self.get_cache(caches, "self-attention"))
        x = self.ln1(decoder_in + self.dropout(x_att))

        x_att = self.ff(x)
        x = self.ln3(x_att + self.dropout(x_att))
        return x


def positional_encodings(model_size, maximum_sequence_length):
    position_enc = np.arange(maximum_sequence_length).reshape((-1, 1)) \
                   / (np.power(10000,
                               (2. / model_size) * np.arange(model_size).reshape((1, -1))))
    # Apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    return mx.nd.array(position_enc)


class TransformerDecoder(mx.gluon.HybridBlock):
    def __init__(self,
                 config: TransformerConfig,
                 maximum_sequence_length: Optional[int] = 10000):
        super().__init__()
        self.config = config
        self.maximum_sequence_length = maximum_sequence_length

        with self.name_scope():
            self.pos_embeddings = positional_encodings(self.config.model_size, self.maximum_sequence_length)
            self.layers = [TransformerDecoderLayer(config)
                           for _ in range(self.config.num_layers)]

            # filthy hack to use lists of models
            for i, layer in enumerate(self.layers):
                self.__setattr__("layer{}".format(i), layer)

    def hybrid_forward(self, F, inputs, input_mask):
        return self.forward_train(F, inputs, input_mask)

    def forward_train(self, F,
                      decoder_inputs, decoder_mask):
        seq_len = decoder_inputs.shape[1]
        inputs = mx.nd.sqrt(mx.nd.full(shape=(1,), val=self.config.model_size)) * decoder_inputs + self.pos_embeddings[:seq_len]
        for layer in self.layers:
            inputs = layer.forward(inputs, decoder_mask, None)
        return inputs

    def forward_inference(self, decoder_state, inputs):
        mask = mx.nd.ones_like(inputs) # during inference, we do not need to pad anything

        t = decoder_state.t
        inputs = mx.nd.sqrt(mx.nd.full(shape=(1,), val=self.config.model_size)) * inputs + self.pos_embeddings[t:t+1]
        for layer_idx, layer in enumerate(self.layers):
            inputs = layer.forward(inputs, mask, decoder_state.caches[layer_idx])
        return inputs


class TransformerEncoder(mx.gluon.HybridBlock):
    def __init__(self,
                 config: TransformerConfig,
                 maximum_sequence_length: Optional[int] = 10000):
        super().__init__()
        self.config = config
        self.maximum_sequence_length = maximum_sequence_length
        with self.name_scope():
            self.pos_embeddings = positional_encodings(self.config.model_size, self.maximum_sequence_length)
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

