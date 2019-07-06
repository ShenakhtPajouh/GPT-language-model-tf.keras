import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from utils import shape_list, gelu, swish, argmax, top_k_sampling, nucleus_sampling, sampling

act_fns = {
    'relu': tf.nn.relu,
    'swish': swish,
    'gelu': gelu
}

class Norm(Model):
    """
    n_state = shape_list(x)[-1]
    """

    def __init__(self, name, n_state, **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_state = n_state

    def build(self, input_shape):
        self.g = self.add_weight(name='g', shape=[self.n_state], dtype=tf.float32,
                                 initializer=tf.keras.initializers.constant(1))
        self.b = self.add_weight(name="b", shape=[self.n_state], dtype=tf.float32,
                                 initializer=tf.keras.initializers.constant(0))
        super(Norm, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self._norm(inputs, self.g, self.b, axis=[-1])

    def _norm(self, x, g=None, b=None, e=1e-5, axis=[1]):
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + e)
        if g is not None and b is not None:
            x = x * g + b
        return x

    def compute_output_shape(self, input_shape):
        return super(Norm, self).compute_output_shape(input_shape)


class Conv1D(Model):

    def __init__(self, name, nx, nf, rf, **kwargs):
        super().__init__(name, **kwargs)
        self.nx = nx
        self.nf = nf
        self.rf = rf

    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=[self.rf, self.nx, self.nf], dtype=tf.float32,
                                 initializer=tf.keras.initializers.random_normal(stddev=0.02))
        self.b = self.add_weight(name="b", shape=[self.nf], dtype=tf.float32,
                                 initializer=tf.keras.initializers.constant(0))
        super(Conv1D, self).build(input_shape=input_shape)

    def call(self, inputs, **kwargs):
        if self.rf == 1:
            c = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, self.nx]), tf.reshape(self.w, [-1, self.nf])) + self.b,
                           shape_list(inputs)[:-1] + [self.nf])
        else:
            c = tf.nn.conv1d(value=inputs, filters=self.w, stride=1, padding='VALID') + self.b
        return c

    def compute_output_shape(self, input_shape):
        return super(Conv1D, self).compute_output_shape(input_shape)


class Attention(Model):
    """
    nx = shape_list(x)[-1]
    where x in inputs args of call
    """

    def __init__(self, name, nx, n_state, n_head, scale=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.nx = nx
        self.n_state = n_state
        self.n_head = n_head
        self.scale = scale
        self.conv1d_c = Conv1D(name='c_attn', nx=self.nx, nf=self.n_state * 3, rf=1)
        self.conv1d_a = Conv1D(name='c_proj', nx=self.nx, nf=self.n_state, rf=1)

    def assign(self, a, b, l, k):
        ret = []
        if k:
            ret.append(a[:, :, :, :l])
            ret.append(b)
            ret.append(a[:, :, :, l+1:])
            return tf.concat(ret, axis = 3)

        else:
            ret.append(a[:, :, :l, :])
            ret.append(b)
            ret.append(a[:, :, l+1:, :])
            return tf.concat(ret, axis = 2)

    def call(self, inputs, mem_k, mem_v, length):
        c = self.conv1d_c(inputs)
        q, k, v = tf.split(c, 3, 2)
        q = self.split_heads(q, self.n_head)
        k = self.split_heads(k, self.n_head, k=True)
        v = self.split_heads(v, self.n_head)
        k = self.assign(mem_k, k, length, k=True)
        v = self.assign(mem_v, v, length, k=False)
        a = self._attn(q, k, v, length)
        a = self.merge_heads(a)
        a = self.conv1d_a(a)
        return a, k, v

    def split_states(self, x, n):
        x_shape = shape_list(x)
        m = x_shape[-1]
        new_x_shape = x_shape[:-1] + [n, m // n]
        return tf.reshape(x, new_x_shape)

    def merge_states(self, x):
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
        return tf.reshape(x, new_x_shape)

    def split_heads(self, x, n, k=False):
        if k:
            return tf.transpose(self.split_states(x, n), [0, 2, 3, 1])
        else:
            return tf.transpose(self.split_states(x, n), [0, 2, 1, 3])

    def merge_heads(self, x):
        return self.merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(self, w, l):
        n = shape_list(w)[-1]
        b = tf.concat([tf.ones([l]), tf.zeros([n - l])], axis = 0)
        b = tf.reshape(b, [1, 1, 1, n])
        w = w * b + -1e9 * (1 - b)
        return w

    def _attn(self, q, k, v, l):
        w = tf.matmul(q, k)
        if self.scale:
            n_state = shape_list(v)[-1]
            w = w * tf.rsqrt(tf.cast(n_state, tf.float32))
        w = self.mask_attn_weights(w, l)
        w = tf.nn.softmax(w)
        a = tf.matmul(w, v)
        return a

    def compute_output_shape(self, input_shape):
        return super(Attention, self).compute_output_shape(input_shape)


class MLP(Model):
    def __init__(self, name, n_embd, n_state, afn):
        """
        The multilayer perceptron is a class of feedforward.
        This module can be used as a one-dimensional convolutional neural network
        or as a fully-connected neural network.
        """
        super().__init__(name=name)
        self.n_embd = n_embd
        self.n_state = n_state
        self.act = act_fns[afn]
        self.conv_fc = Conv1D("c_fc", self.n_embd, self.n_state, 1)
        self.conv_proj = Conv1D("c_proj", self.n_state, self.n_embd, 1)

    def call(self, inputs):
        hidden1 = self.act(self.conv_fc(inputs))
        hidden2 = self.conv_proj(hidden1)
        return hidden2


class Block(Model):
    def __init__(self, name, n_vocab, n_embd, n_head, afn, scale):
        """
          The Transformer block is the core of the model.
          It contains attention layer, layer normalization and multilayer perceptron (i.e. feedforward)
        """
        super().__init__(name=name)
        self.n_vocab = n_vocab
        self.n_embd = n_embd
        self.n_head = n_head
        self.afn = afn
        self.scale = scale
        self.attn = Attention("/attn", self.n_embd, self.n_embd, self.n_head, self.scale)
        self.norm1 = Norm("/ln_1", self.n_embd)
        self.mlp = MLP("/mlp", self.n_embd, 4 * self.n_embd, self.afn)
        self.norm2 = Norm("/ln_2", self.n_embd)

    def call(self, inputs, mem_k, mem_v, length):
        a, mem_k, mem_v = self.attn(inputs, mem_k, mem_v, length)
        n = self.norm1(inputs + a)
        m = self.mlp(n)
        h = self.norm2(n + m)
        return h, mem_k, mem_v


class EmbeddingLayer(keras.layers.Layer):
    def __init__(self, name, n_vocab, n_ctx=512, n_embd=768, stddev=0.02):
        super().__init__(name=name)
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.stddev = stddev

    def build(self, input_shape):
        self.we = self.add_weight(name="we", shape=(self.n_ctx + self.n_vocab, self.n_embd), dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(stddev=self.stddev))
        super().build(input_shape=input_shape)

    def call(self, inputs):
        return tf.reduce_sum(tf.gather(self.we, inputs), 2)


class Transformer(Model):
    def __init__(self, name, n_vocab, n_ctx=512, n_embd=768, n_layer=12, n_head=12, afn="gelu", scale=False):
        """
          This is the transformer model in
          'https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf'
          fine-tuned for language-model.

          Args:
            name: The name of the model
            n_vocab: Size of the vocabulary
            n_ctx: Size of the context
            n_embd: Embeddings dimension
            n_layer: Number of the transformer blocks
            n_head: Number of attention heads
            afn: The non-linear activation function in MLP
            scale: It is a boolean which is true when attention weights are scaled
        """
        super().__init__(name=name)
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.afn = afn
        self.scale = scale
        self.embed = EmbeddingLayer("embedding", n_vocab, n_ctx, n_embd)

        self.transformer_stack = []
        for layer in range(n_layer):
            self.transformer_stack.append(Block("h", n_vocab, n_embd, n_head, afn, scale))

    def call(self, inputs, decoder_type = 'argmax', k = 25, p = 0.9, temperature = 0.8):
        """
        Args:
            inputs: it is a list of the previous token, memorized keys and values, and size of the previous tokens.
                    token shape = (1, 1, 2 (ID and position))
                    mem_k shape = (number of layers, 1, number of heads, attn hidden size, context size)
                    mem_v shape = (number of layers, 1, number of heads, context size, attn hidden size)
                    length = an integer

        Returns:
            next_token: shape = (1, 1, 2 (ID and position))
            mem_k: shape = (number of layers, 1, number of heads, attn hidden size, context size)
            mem_v: shape = (number of layers, 1, number of heads, context size, attn hidden size)
        """

        token = inputs[0]
        mem_k = inputs[1]
        mem_v = inputs[2]
        length = inputs[3]

        new_mem_k = []
        new_mem_v = []

        hidden = self.embed(token)

        for i, block in enumerate(self.transformer_stack):
            hidden, k, v = block(hidden, mem_k[i], mem_v[i], length)
            new_mem_k.append(k)
            new_mem_v.append(v)

        mem_k = tf.stack(new_mem_k)
        mem_v = tf.stack(new_mem_v)

        logit = tf.reshape(tf.matmul(hidden[0, :, :], self.embed.we[:self.n_vocab, :], transpose_b=True),
                          [self.n_vocab])

        if decoder_type == 'argmax':
            next_token = argmax(logit)

        elif decoder_type == 'top-k':
            next_token = top_k_sampling(logit, k, temperature)

        elif decoder_type == 'nucleus':
            next_token = nucleus_sampling(logit, p)

        else:
            next_token = sampling(logit, temperature)

        return next_token, mem_k, mem_v
