import jax
import jax.numpy as jnp
import flax.linen as nn


act_fn = jax.nn.silu


def norm_layer(x):
    return nn.LayerNorm(use_bias=False, use_scale=True)(x)


def small_init():
    def init(key, shape, dtype=jnp.float_):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        std = jnp.sqrt(2 / (5 * shape[-2]))
        return jax.random.normal(key, shape, dtype) * std

    return init


def wang_init(n_layers):
    def init(key, shape, dtype=jnp.float_):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        std = 2 / n_layers / jnp.sqrt(shape[-2])
        return jax.random.normal(key, shape, dtype) * std

    return init


class MLP(nn.Module):
    hidden_dim: int
    n_layers: int

    @nn.compact
    def __call__(self, x):
        C = x.shape[-1]

        gate_kernel = self.param("gate_kernel", small_init(), (C, self.hidden_dim))
        up_kernel = self.param("up_kernel", small_init(), (C, self.hidden_dim))
        down_kernel = self.param(
            "down_kernel", wang_init(self.n_layers), (self.hidden_dim, C)
        )

        gate = jnp.dot(x, gate_kernel)
        gate = act_fn(gate)
        up = jnp.dot(x, up_kernel)
        x = gate * up
        x = norm_layer(x)  # normformer
        down = jnp.dot(x, down_kernel)

        return down


class BlockDense(nn.Module):
    @nn.compact
    def __call__(self, x):
        assert x.shape[-1] % 4 == 0, "x.shape[-1] must be divisible by 4 for BlockDense"
        head_dim = x.shape[-1] // 4
        layer = self.param(f"block_kernel", small_init(), (4, head_dim, head_dim))
        out = jnp.reshape(x, (x.shape[0], 4, head_dim))
        out = jnp.einsum("...hi,hij->...hj", out, layer)
        return jnp.reshape(out, x.shape)
    

def init_hidden_state(batch_size, hidden_dim, dtype=jnp.float_):
    c_t = jnp.zeros((batch_size, hidden_dim), dtype=dtype)
    n_t = jnp.ones((batch_size, hidden_dim), dtype=dtype)
    h_t = jnp.zeros((batch_size, hidden_dim), dtype=dtype)
    m_t = jnp.zeros((batch_size, hidden_dim), dtype=dtype)
    return c_t, n_t, h_t, m_t


class sLSTM(nn.Module):
    """sLSTM cell.

    In this context we use slstm one step at a time (seq len = 0) so inputs 
    are shape (batch_size, hidden_dim). This module can be scanned over a seq
    or applied one step at a time.
    """

    ff_dim: int
    n_layers: int  # only used for init

    @nn.compact
    def __call__(self, seq, hid):
        B, C = seq.shape  # single step in sequence
        N = 4
        assert C % N == 0
        c_tm1, n_tm1, h_tm1, m_tm1 = hid

        # weights
        with_bias = True
        W_z = nn.Dense(features=C, use_bias=with_bias, kernel_init=small_init())
        W_i = nn.Dense(features=C, use_bias=with_bias, kernel_init=small_init())
        W_f = nn.Dense(features=C, use_bias=with_bias, kernel_init=small_init())
        W_o = nn.Dense(features=C, use_bias=with_bias, kernel_init=small_init())

        # recurrent weights
        R_z = BlockDense()
        R_i = BlockDense()
        R_f = BlockDense()
        R_o = BlockDense()

        # input norm
        x_t = norm_layer(seq)

        # slstm
        z_t = W_z(x_t) + R_z(h_tm1)
        i_t = W_i(x_t) + R_i(h_tm1)
        f_t = W_f(x_t) + R_f(h_tm1)
        o_t = W_o(x_t) + R_o(h_tm1)

        z_t = jnp.tanh(z_t)
        m_t = jnp.maximum(f_t + m_tm1, i_t)
        i_t = jnp.exp(i_t - m_t)
        f_t = jnp.exp(f_t - m_t + m_tm1)
        o_t = jax.nn.sigmoid(o_t)

        c_t = f_t * c_tm1 + i_t * z_t
        n_t = f_t * n_tm1 + i_t
        h_t = o_t * (c_t / n_t)

        # output norm
        out = nn.GroupNorm(
            num_groups=N,
            epsilon=1e-6,
            use_bias=False,
            use_scale=True,
            reduction_axes=[-1],
        )(h_t)
        out += seq

        out = MLP(self.ff_dim, self.n_layers)(norm_layer(out))
        out += seq

        return out, (c_t, n_t, h_t, m_t)