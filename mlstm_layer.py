import jax.numpy as jnp
import flax.linen as nn

from mlstm_cell import mLSTMCell


class mLSTMLayer(nn.Module):
    """mLSTM layer.

    Attributes:
        embedding_dim: number of output features
        hidden_dim: number of hidden features, usually ~1.3 * embedding_dim
        num_heads: number of attention heads
        context_length: sequence length of the input
        conv1d_kernel_size: kernel size of the causal 1D convolution
        qkv_proj_blocksize: blocksize of the linear projections
        bias: whether to use bias in the linear layers
        num_blocks: number of blocks in the model, used for weight initialization
    """

    embedding_dim: int
    hidden_dim: int
    num_heads: int
    context_length: int
    conv1d_kernel_size: int = 4
    qkv_proj_blocksize: int = 4
    bias: bool = False
    num_blocks: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_inner = nn.Dense(
            features=2 * self.hidden_dim,
            use_bias=self.bias,
            kernel_init=small_init(x.shape[-1]),
        )(x)

        x_mlstm, z = jnp.split(x_inner, 2, axis=-1)

        x_mlstm_conv = CausalConv1d(
            feature_dim=self.hidden_dim,
            kernel_size=self.conv1d_kernel_size,
            causal_conv_bias=True,
            channel_mixing=False,
        )(x_mlstm)
        x_mlstm_conv_act = nn.silu(x_mlstm_conv)

        num_proj_heads = round(self.hidden_dim // self.qkv_proj_blocksize)
        q = LinearHeadwiseExpand(
            in_features=self.hidden_dim,
            num_heads=num_proj_heads,
            expand_factor_up=1.0,
            bias=self.bias,
        )(x_mlstm_conv_act)

        k = LinearHeadwiseExpand(
            in_features=self.hidden_dim,
            num_heads=num_proj_heads,
            expand_factor_up=1.0,
            bias=self.bias,
        )(x_mlstm_conv_act)

        v = LinearHeadwiseExpand(
            in_features=self.hidden_dim,
            num_heads=num_proj_heads,
            expand_factor_up=1.0,
            bias=self.bias,
        )(x_mlstm)

        h_tilde_state = mLSTMCell(
            embedding_dim=self.hidden_dim,
            num_heads=self.num_heads,
            context_length=self.context_length,
        )(q, k, v)

        learnable_skip = self.param(
            "learnable_skip", nn.initializers.ones_init(), (self.hidden_dim,)
        )
        h_tilde_state_skip = h_tilde_state + (learnable_skip * x_mlstm_conv_act)

        h_state = h_tilde_state_skip * nn.silu(z)

        y = nn.Dense(
            features=self.embedding_dim,
            use_bias=self.bias,
            kernel_init=wang_init(h_state.shape[-1], self.num_blocks),
        )(h_state)

        return y


small_init = lambda dim: nn.initializers.normal(jnp.sqrt(2 / (5 * dim)))
wang_init = lambda dim, num_blocks: nn.initializers.normal(2 / num_blocks / jnp.sqrt(dim))


class LinearHeadwiseExpand(nn.Module):
    in_features: int
    num_heads: int
    expand_factor_up: float = 1.0
    _out_features: int = -1
    bias: bool = True
    trainable_weight: bool = True
    trainable_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out_features = (
            self._out_features
            if self._out_features > 0
            else round(self.expand_factor_up * self.in_features)
        )
        out_features_per_head = out_features // self.num_heads

        weight = (
            self.param(
                "weight",
                small_init(self.in_features // self.num_heads),
                (
                    self.num_heads,
                    self.in_features // self.num_heads,
                    out_features_per_head,
                ),
            )
            if self.trainable_weight
            else jnp.zeros(...)
        )

        if self.bias and self.trainable_bias:
            bias = self.param("bias", nn.initializers.zeros_init(), (out_features,))

        shape = x.shape
        x = jnp.reshape(x, (*shape[:-1], self.num_heads, -1))
        x = jnp.einsum("...hd,hdo->...ho", x, weight)
        x = jnp.reshape(x, (*shape[:-1], -1))

        if self.bias and self.trainable_bias:
            x = x + bias

        return x


class CausalConv1d(nn.Module):
    feature_dim: int
    kernel_size: int = 4
    causal_conv_bias: bool = True
    channel_mixing: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.kernel_size == 0:
            return x

        groups = self.feature_dim if not self.channel_mixing else 1
        y = nn.Conv(
            features=self.feature_dim,
            kernel_size=(self.kernel_size,),
            padding="CAUSAL",
            feature_group_count=groups,
            use_bias=self.causal_conv_bias,
            kernel_init=nn.initializers.he_normal(),
        )(x)

        return y