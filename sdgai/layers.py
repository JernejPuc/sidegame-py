"""Basic building blocks of PCNet components."""

from typing import Hashable, Iterable, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sdgai.utils import StateStore, init_std, silog


class GDWPool1d(nn.Module):
    """Pooling (1D) with global depthwise kernels."""

    def __init__(self, in_channels: int, out_channels: int, width: int):
        super().__init__()

        self.silu = nn.SiLU()
        self.expansion = nn.Conv1d(in_channels, out_channels, 1)
        self.depthwise = nn.Conv1d(out_channels, out_channels, width, groups=out_channels)

        nn.init.normal_(self.expansion.weight, mean=0., std=init_std(self.expansion))
        nn.init.normal_(self.depthwise.weight, mean=0., std=init_std(self.depthwise))
        nn.init.constant_(self.expansion.bias, 0.)
        nn.init.constant_(self.depthwise.bias, 0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.silu(self.expansion(x))
        x = self.silu(self.depthwise(x))

        return x


class FMBConv(nn.Module):
    """
    Fused MBConv block (without squeeze and excitation), modified for
    pre-activation and to use Fixup-like initialisation and parameters.

    Fixup multipliers and biases are added to main branches in residual
    blocks to replace batch normalisation layers. In a convolutional network,
    the main branch of each residual block contains one multiplier,
    while biases are added before each convolution or non-linear operation.
    Branches without batch normalisation are not modified with Fixup.

    NOTE: To use Fixup initialisation, the number of sequential blocks in the
    convolutional network should be provided with `n_blocks`, as the variance
    of randomly initialised weights depends on it.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 1,
        padding_mode: str = 'zeros',
        downscale: bool = False,
        expansion_ratio: Union[int, float] = 2,
        n_blocks: int = None,
        dim: int = 2
    ):
        super().__init__()

        assert out_channels >= in_channels, \
            f'Cannot preserve identity with given channel mismatch: {in_channels} and {out_channels}.'

        assert dim in (1, 2), f'`dim` must be 1 or 2, got {dim}.'

        conv = nn.Conv2d if dim == 2 else nn.Conv1d
        pool = nn.AvgPool2d if dim == 2 else nn.AvgPool1d
        exp_channels = int(in_channels * expansion_ratio)

        # Activation
        self.silu = nn.SiLU()

        # Main components
        if downscale:
            self.expansion = conv(
                in_channels, exp_channels, kernel_size, stride=2, padding=padding, padding_mode=padding_mode)

            self.id_avg = pool(2, stride=2)

        else:
            self.expansion = conv(in_channels, exp_channels, kernel_size, padding=padding, padding_mode=padding_mode)
            self.id_avg = None

        self.projection = conv(exp_channels, out_channels, 1, bias=False)
        self.id_ext = (0, 0)*dim + (0, out_channels - in_channels) if out_channels != in_channels else None

        # Fixup components and init
        self.pre_act_bias = nn.Parameter(torch.zeros((1, in_channels) + (1,)*dim))
        self.pre_exp_bias = nn.Parameter(torch.zeros((1, in_channels) + (1,)*dim))
        self.pre_proj_bias = nn.Parameter(torch.zeros((1, exp_channels) + (1,)*dim))
        self.id_scale = nn.Parameter(torch.ones((1, in_channels) + (1,)*dim))

        nn.init.normal_(self.expansion.weight, mean=0., std=init_std(self.expansion, n_blocks, n_layers=2))
        nn.init.constant_(self.expansion.bias, 0.)
        nn.init.constant_(self.projection.weight, 0.)

    def forward(self, x):
        x_res = self.silu(x + self.pre_act_bias)
        x_res = self.expansion(x_res + self.pre_exp_bias)
        x_res = self.silu(x_res)  # Bias in expansion
        x_res = self.projection(x_res + self.pre_proj_bias)

        # Modify identity to allow addition
        if self.id_avg is not None:
            x = self.id_avg(x)

        x = x * self.id_scale

        if self.id_ext is not None:
            x = F.pad(x, self.id_ext)

        return x + x_res


class BResBlock(nn.Module):
    """
    Bottleneck residual block, modified for pre-activation and to use
    Fixup-like initialisation and parameters.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        padding_mode: str = 'zeros',
        downscale: bool = False,
        n_blocks: int = None
    ):
        super().__init__()

        # Activation
        self.silu = nn.SiLU()

        # Main components
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.conv3 = nn.Conv2d(mid_channels, in_channels, 1, bias=False)

        if downscale:
            self.conv2 = nn.Conv2d(
                mid_channels, mid_channels, 4, stride=2, padding=1, padding_mode=padding_mode)

            self.id_avg = nn.AvgPool2d(2, stride=2)

        else:
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, padding_mode=padding_mode)
            self.id_avg = None

        # Fixup components and init
        self.pre_act_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.pre_conv1_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.pre_conv2_bias = nn.Parameter(torch.zeros(1, mid_channels, 1, 1))
        self.pre_conv3_bias = nn.Parameter(torch.zeros(1, mid_channels, 1, 1))
        self.id_scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        nn.init.normal_(self.conv1.weight, mean=0., std=init_std(self.conv1, n_blocks=n_blocks, n_layers=3))
        nn.init.normal_(self.conv2.weight, mean=0., std=init_std(self.conv2, n_blocks=n_blocks, n_layers=3))
        nn.init.constant_(self.conv3.weight, 0.)
        nn.init.constant_(self.conv1.bias, 0.)
        nn.init.constant_(self.conv2.bias, 0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = self.silu(x + self.pre_act_bias)
        x_res = self.conv1(x_res + self.pre_conv1_bias)
        x_res = self.silu(x_res)  # Bias in conv1
        x_res = self.conv2(x_res + self.pre_conv2_bias)
        x_res = self.silu(x_res)  # Bias in conv2
        x_res = self.conv3(x_res + self.pre_conv3_bias)

        # Modify identity to allow addition
        if self.id_avg is not None:
            x = self.id_avg(x)

        return x * self.id_scale + x_res


class PosConv(nn.Module):
    """1x1 convolution with positional embeddings."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        preactivate: bool = True
    ):
        super().__init__()

        self.pos_embeddings_h = nn.Parameter(torch.zeros(1, in_channels, height, 1))
        self.pos_embeddings_w = nn.Parameter(torch.zeros(1, in_channels, 1, width))

        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.normal_(self.conv.weight, mean=0., std=init_std(self.conv))
        nn.init.constant_(self.conv.bias, 0.)

        self.silu = nn.SiLU()
        self.preactivate = preactivate
        self.pre_act_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1)) if preactivate else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.preactivate:
            x = self.silu(x + self.pre_act_bias)

        x = x + self.pos_embeddings_h + self.pos_embeddings_w
        x = self.conv(x)
        x = self.silu(x)

        return x


class SASAtten(nn.Module):
    """
    Residual (inv.) bottleneck block with stand-alone self-attention,
    pre-activation, and Fixup-like initialisation.
    """

    def __init__(
        self,
        in_channels: int,
        exp_channels: int,
        height: int,
        width: int,
        n_heads: int = 1,
        n_blocks: int = None
    ):
        super().__init__()

        self.height = height
        self.width = width
        self.feat_size = exp_channels
        exp_channels = self.feat_size * 3
        self.head_size = self.feat_size // n_heads
        self.n_heads = n_heads

        assert not (self.feat_size % n_heads), 'Expanded feature size must be divisible by `n_heads`.'

        # Activation
        self.silu = nn.SiLU()

        # Main components
        self.qkv_extraction = nn.Conv2d(in_channels, exp_channels, 1, bias=True)
        self.projection = nn.Conv2d(self.feat_size, in_channels, 1, bias=False)

        self.pos_embeddings_q_h = nn.Parameter(torch.zeros(1, self.feat_size, self.height, 1))
        self.pos_embeddings_q_w = nn.Parameter(torch.zeros(1, self.feat_size, 1, self.width))
        self.pos_embeddings_k_h = nn.Parameter(torch.zeros(1, self.feat_size, self.height, 1))
        self.pos_embeddings_k_w = nn.Parameter(torch.zeros(1, self.feat_size, 1, self.width))
        self.dot_scale = self.head_size**0.5

        # Fixup components
        self.pre_act_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.pre_qkv_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.pre_proj_bias = nn.Parameter(torch.zeros(1, self.feat_size, 1, 1))
        self.id_scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        # Fixup init
        nn.init.normal_(self.qkv_extraction.weight, mean=0., std=init_std(self.qkv_extraction, n_blocks, n_layers=3))
        nn.init.constant_(self.qkv_extraction.bias, 0.)
        nn.init.constant_(self.projection.weight, 0.)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-headed scaled dot product global self-attention."""

        # Extract queries, keys, and values
        q, k, v = torch.split(x, self.feat_size, dim=1)

        # Add absolute positional embeddings
        q = q + self.pos_embeddings_q_h + self.pos_embeddings_q_w
        k = k + self.pos_embeddings_k_h + self.pos_embeddings_k_w

        # Reshape for matrix operations
        b = x.shape[0]

        q = q.view(b, self.n_heads, self.head_size, self.height*self.width)
        k = k.view(b, self.n_heads, self.head_size, self.height*self.width)
        v = v.view(b, self.n_heads, self.head_size, self.height*self.width)

        # b, n, c//n, hw, hw
        logits = torch.matmul(q, torch.transpose(k, 2, 3)) / self.dot_scale
        weights = F.softmax(logits, dim=-1)

        # b, n, c//n, hw
        v = torch.matmul(weights, v)
        v = v.view(b, self.feat_size, self.height, self.width)

        return v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = self.silu(x + self.pre_act_bias)
        x_res = self.qkv_extraction(x_res + self.pre_qkv_bias)

        x_res = self.attention(x_res)  # Bias in qkv
        # Role of activation satisfied by mult. with softmax in film_attention

        x_res = self.projection(x_res + self.pre_proj_bias)

        return x * self.id_scale + x_res


class FiLMAtten(nn.Module):
    """
    Dense block (uses concatenation instead of residual addition) with
    pre-activation and externally modulated attention.

    Feature-wise linear modulation (FiLM) is used to add an external, global
    objective to the query by highlighting its pattern, e.g. emphasising
    positional tokens.
    """

    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        n_heads: int = 1
    ):
        super().__init__()

        self.height = height
        self.width = width
        self.feat_size = in_channels
        exp_channels = self.feat_size * 3
        self.head_size = self.feat_size // n_heads
        self.n_heads = n_heads

        assert not (self.feat_size % n_heads), 'Expanded feature size must be divisible by `n_heads`.'

        # Activation
        self.silu = nn.SiLU()

        self.qkv_extraction = nn.Conv2d(in_channels, exp_channels, 1, bias=True)

        self.pos_embeddings_q_h = nn.Parameter(torch.zeros(1, self.feat_size, self.height, 1))
        self.pos_embeddings_q_w = nn.Parameter(torch.zeros(1, self.feat_size, 1, self.width))
        self.pos_embeddings_k_h = nn.Parameter(torch.zeros(1, self.feat_size, self.height, 1))
        self.pos_embeddings_k_w = nn.Parameter(torch.zeros(1, self.feat_size, 1, self.width))
        self.dot_scale = self.head_size**0.5

        # Fixup components
        self.pre_act_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.pre_qkv_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

        # Fixup init
        nn.init.normal_(self.qkv_extraction.weight, mean=0., std=init_std(self.qkv_extraction))
        nn.init.constant_(self.qkv_extraction.bias, 0.)

    def film_attention(self, x: torch.Tensor, mod_multiplier: torch.Tensor, mod_bias: torch.Tensor) -> torch.Tensor:
        """Multi-headed scaled dot product global self-attention with FiLM."""

        # Extract queries, keys, and values
        q, k, v = torch.split(x, self.feat_size, dim=1)

        # Add absolute positional embeddings
        q = q + self.pos_embeddings_q_h + self.pos_embeddings_q_w
        k = k + self.pos_embeddings_k_h + self.pos_embeddings_k_w

        # Feature-wise linear modulation
        q = q * mod_multiplier[..., None, None] + mod_bias[..., None, None]

        # Reshape for matrix operations
        b = x.shape[0]

        q = q.view(b, self.n_heads, self.head_size, self.height*self.width)
        k = k.view(b, self.n_heads, self.head_size, self.height*self.width)
        v = v.view(b, self.n_heads, self.head_size, self.height*self.width)

        # b, n, c//n, hw, hw
        logits = torch.matmul(q, torch.transpose(k, 2, 3)) / self.dot_scale
        weights = F.softmax(logits, dim=-1)

        # b, n, c//n, hw
        v = torch.matmul(weights, v)
        v = v.view(b, self.feat_size, self.height, self.width)

        return v

    def forward(self, x: torch.Tensor, mod_multiplier: torch.Tensor, mod_bias: torch.Tensor) -> torch.Tensor:
        x_preact = self.silu(x + self.pre_act_bias)
        x_res = self.qkv_extraction(x_preact + self.pre_qkv_bias)

        x_res = self.film_attention(x_res, mod_multiplier, mod_bias)  # Bias in qkv
        # Role of activation satisfied by mult. with softmax in film_attention

        return torch.cat((x_preact, x_res), dim=1)


class IRCell(nn.Module):
    """
    Multi-actor implementation of an independently recurrent neural cell (2 layers)
    with custom (log-like) activation.

    By specifying keys corresponding to different actor instances, concurrent or
    otherwise, associated hidden states can be kept, retrieved, and updated
    in batches.

    NOTE: Truncated backpropagation through time (TBPTT) can be used if
    hidden states are regularly detached from the computational graph
    and if batched sequences/trajectories are not sampled from the same actor
    instance (as their keys are not unique and their order is ambiguous).

    References:
    - https://arxiv.org/abs/1803.04831
    - https://arxiv.org/abs/1910.06251
    """

    def __init__(
        self,
        op1: Union[nn.Linear, nn.Conv1d, nn.Conv2d],
        op2: Union[nn.Linear, nn.Conv1d, nn.Conv2d],
        u1: nn.Parameter,
        u2: nn.Parameter,
        hidden_dim: Union[int, Tuple[int]]
    ):
        super().__init__()

        self.activation = silog
        self.op1 = op1
        self.op2 = op2
        self.u1 = u1
        self.u2 = u2

        self.store_1 = StateStore(hidden_dim)
        self.store_2 = StateStore(hidden_dim)

    def clear(self, keys: Iterable[Hashable] = None):
        """Clear final (first) hidden states of both recurrent layers."""

        self.store_1.clear(keys=keys)
        self.store_2.clear(keys=keys)

    def detach(self, keys: Iterable[Hashable] = None):
        """Detach final (first) hidden states of both recurrent layers."""

        self.store_1.detach(keys=keys)
        self.store_2.detach(keys=keys)

    def move(self, device: Union[str, torch.device]):
        """Move initial/default states to new device."""

        self.store_1.move(device)
        self.store_2.move(device)

    def forward(self, x: torch.Tensor, keys: Iterable[Hashable]) -> torch.Tensor:
        # Load recurrent states
        hidden_states_1 = self.store_1.get(keys=keys)
        hidden_states_2 = self.store_2.get(keys=keys)

        # Infer new recurrent states
        hidden_states_1 = self.activation(self.op1(x) + self.u1 * hidden_states_1)
        hidden_states_2 = self.activation(self.op2(hidden_states_1) + self.u2 * hidden_states_2)

        # Save new recurrent states
        self.store_1.append(hidden_states_1)
        self.store_2.append(hidden_states_2)

        return hidden_states_2


class IRLinearCell(IRCell):
    """
    Multi-actor implementation of an independently recurrent neural cell (2 layers)
    with linear processing of inputs and custom (log-like) activation.
    """

    def __init__(self, input_size: int, hidden_size: int):
        fc1 = nn.Linear(input_size, hidden_size, bias=True)
        fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        u1 = nn.Parameter(torch.empty(1, hidden_size))
        u2 = nn.Parameter(torch.empty(1, hidden_size))

        nn.init.normal_(fc1.weight, mean=0., std=init_std(fc1))
        nn.init.normal_(fc2.weight, mean=0., std=init_std(fc2))
        nn.init.constant_(fc1.bias, 0.)
        nn.init.constant_(fc2.bias, 0.)
        nn.init.uniform_(u1, a=0., b=1.)
        nn.init.uniform_(u2, a=0., b=1.)

        super().__init__(fc1, fc2, u1, u2, hidden_size)


class IRConv2dCell(IRCell):
    """
    Multi-actor implementation of an independently recurrent neural cell (2 layers)
    with (2D) convolution of inputs and custom (log-like) activation.

    NOTE: Unlike its linear counterpart, conv cells are intended for encoding,
    where shorter time frames are preferred and spatial distinctions should not
    be randomly variant at the start. Thus, temporal retention is initialised to
    a smaller constant.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        height: int,
        width: int,
        padding_mode: str = 'zeros'
    ):
        conv_1 = nn.Conv2d(
            input_size,
            hidden_size,
            kernel_size,
            padding=kernel_size//2,
            bias=True,
            padding_mode=padding_mode)

        conv_2 = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size,
            padding=kernel_size//2,
            bias=True,
            padding_mode=padding_mode)

        u1 = nn.Parameter(torch.empty(1, hidden_size, height, width))
        u2 = nn.Parameter(torch.empty(1, hidden_size, height, width))

        nn.init.normal_(conv_1.weight, mean=0., std=init_std(conv_1))
        nn.init.normal_(conv_2.weight, mean=0., std=init_std(conv_2))
        nn.init.constant_(conv_1.bias, 0.)
        nn.init.constant_(conv_2.bias, 0.)
        nn.init.constant_(u1, 0.5)
        nn.init.constant_(u2, 0.5)

        super().__init__(conv_1, conv_2, u1, u2, (hidden_size, height, width))


class IRConv1dCell(IRCell):
    """
    Multi-actor implementation of an independently recurrent neural cell (2 layers)
    with (1D) convolution of inputs and custom (log-like) activation.

    NOTE: Unlike its linear counterpart, conv cells are intended for encoding,
    where shorter time frames are preferred and spatial distinctions should not
    be randomly variant at the start. Thus, temporal retention is initialised to
    a smaller constant.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        width: int,
        padding_mode: str = 'zeros'
    ):
        conv_1 = nn.Conv1d(
            input_size,
            hidden_size,
            kernel_size,
            padding=kernel_size//2,
            bias=True,
            padding_mode=padding_mode)

        conv_2 = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size,
            padding=kernel_size//2,
            bias=True,
            padding_mode=padding_mode)

        u1 = nn.Parameter(torch.empty(1, hidden_size, width))
        u2 = nn.Parameter(torch.empty(1, hidden_size, width))

        nn.init.normal_(conv_1.weight, mean=0., std=init_std(conv_1))
        nn.init.normal_(conv_2.weight, mean=0., std=init_std(conv_2))
        nn.init.constant_(conv_1.bias, 0.)
        nn.init.constant_(conv_2.bias, 0.)
        nn.init.constant_(u1, 0.5)
        nn.init.constant_(u2, 0.5)

        super().__init__(conv_1, conv_2, u1, u2, (hidden_size, width))
