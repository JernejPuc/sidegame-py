"""Basic building blocks of PCNet components."""

from typing import Hashable, Iterable, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sdgai.utils import StateStore, init_std, silog


class XTraPool(nn.Module):
    """
    Expansive trainable pooling with pre-activation and Fixup-like initialisation.

    It applies multiple strided depthwise convolutions per channel, followed by
    projection, to pool with consideration for several spatial configurations.

    Beyond reducing spatial dimensions, it can be used to address the final
    stage of convolutional networks, by using an all-encompassing kernel and
    pooling globally. Compared to the sequence of 1x1 conv. -> avg. pooling -> FC,
    spatial relations are not completely disregarded, and compared to flattening
    the final tensor and passing it through a linear layer, parameter counts can
    be significantly lower, as well.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        padding_mode: str = 'zeros',
        expansion_ratio: int = 5,
        dim: int = 2,
        preactivate: bool = True
    ):
        super().__init__()

        assert dim in (1, 2), f'`dim` must be 1 or 2, got {dim}.'

        conv = nn.Conv2d if dim == 2 else nn.Conv1d
        expanded_channels = int(in_channels * expansion_ratio)
        self.preactivate = preactivate

        # Activation
        self.silu = nn.SiLU()

        # Residual components
        self.multi_depthwise = conv(
            in_channels,
            expanded_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=True,
            padding_mode=padding_mode)

        self.projection = conv(expanded_channels, out_channels, 1, bias=True)

        # Fixup components
        if preactivate:
            self.pre_act_bias = nn.Parameter(torch.zeros((1, in_channels) + (1,)*dim))

        else:
            self.pre_act_bias = None

        self.pre_mdw_bias = nn.Parameter(torch.zeros((1, in_channels) + (1,)*dim))
        self.pre_proj_bias = nn.Parameter(torch.zeros((1, expanded_channels) + (1,)*dim))

        nn.init.constant_(self.multi_depthwise.weight, 1./np.prod(self.multi_depthwise.kernel_size))
        nn.init.normal_(self.projection.weight, mean=0., std=0.01)

        nn.init.constant_(self.multi_depthwise.bias, 0.)
        nn.init.constant_(self.projection.bias, 0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.preactivate:
            x = self.silu(x + self.pre_act_bias)

        x = self.multi_depthwise(x + self.pre_mdw_bias)
        x = self.silu(x)  # Bias included in mdw
        x = self.projection(x + self.pre_proj_bias)

        return x


class FMBConv2d(nn.Module):
    """
    Fused MBConv2d block with optional squeeze and excitation, modified for
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
        padding_mode: str = 'zeros',
        downscale: bool = False,
        expansion_ratio: int = 4,
        squeeze_ratio: int = 0,
        n_blocks: int = None
    ):
        super().__init__()

        assert out_channels >= in_channels, \
            f'Cannot preserve identity with given channel mismatch: {in_channels} and {out_channels}.'

        exp_channels = int(in_channels * expansion_ratio)

        # Activation
        self.silu = nn.SiLU()

        # Main components
        if downscale:
            self.expansion = nn.Conv2d(
                in_channels, exp_channels, 4, stride=2, padding=1, padding_mode=padding_mode, bias=True)

            self.id_avg = nn.AvgPool2d(2, stride=2)

        else:
            self.expansion = nn.Conv2d(in_channels, exp_channels, 3, padding=1, padding_mode=padding_mode, bias=True)
            self.id_avg = None

        self.projection = nn.Conv2d(exp_channels, out_channels, 1, bias=False)

        if out_channels != in_channels:
            self.id_ext = nn.Conv2d(in_channels, out_channels-in_channels, 1, bias=False)

        else:
            self.id_ext = None

        # Fixup components and init
        self.pre_act_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.pre_exp_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.pre_proj_bias = nn.Parameter(torch.zeros(1, exp_channels, 1, 1))
        self.id_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))

        nn.init.normal_(self.expansion.weight, mean=0., std=init_std(self.expansion, n_blocks, n_layers=2))
        nn.init.constant_(self.expansion.bias, 0.)
        nn.init.constant_(self.projection.weight, 0.)

        if self.id_ext is not None:
            nn.init.normal_(self.id_ext.weight, mean=0., std=0.05)

        # SE components
        self.use_se = bool(squeeze_ratio)

        if self.use_se:
            squeezed_channels = int(exp_channels // squeeze_ratio)

            self.sigmoid = nn.Sigmoid()
            self.squeeze = nn.Conv2d(exp_channels, squeezed_channels, 1, bias=True)
            self.excitation = nn.Conv2d(squeezed_channels, exp_channels, 1, bias=True)

            nn.init.normal_(self.squeeze.weight, mean=0., std=0.03)
            nn.init.normal_(self.excitation.weight, mean=0., std=0.03)

        else:
            self.sigmoid = None
            self.squeeze = None
            self.excite = None

    def forward(self, x):
        x_preact = self.silu(x + self.pre_act_bias) + self.pre_exp_bias
        x_res = self.expansion(x_preact)  # Bias in x_preact
        x_res = self.silu(x_res)  # Bias in expansion

        if self.use_se:
            # Depthwise global average pooling
            x_se = torch.mean(x_res, dim=(2, 3), keepdim=True)

            # Pointwise squeeze and excitation
            x_se = self.squeeze(x_se)
            x_se = self.silu(x_se)
            x_se = self.excitation(x_se)

            # Pointwise gating
            x_res = x_res * self.sigmoid(x_se)

        x_res = self.projection(x_res + self.pre_proj_bias)

        # Modify identity to allow addition
        if self.id_avg is not None:
            x = self.id_avg(x)

        if self.id_ext is not None:
            x_preact = x_preact if self.id_avg is None else self.id_avg(x_preact)
            x_ext = self.id_ext(x_preact)  # Bias in x_preact
            x = torch.cat((x, x_ext), dim=1)

        return x * self.id_scale + x_res


class SEResBlock2d(nn.Module):
    """
    Basic residual block with optional squeeze and excitation, modified for
    pre-activation and to use Fixup-like initialisation and parameters.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding_mode: str = 'zeros',
        downscale: bool = False,
        squeeze_ratio: bool = 0,
        n_blocks: int = None
    ):
        super().__init__()

        assert out_channels >= in_channels, \
            f'Cannot preserve identity with given channel mismatch: {in_channels} and {out_channels}.'

        # Activation
        self.silu = nn.SiLU()

        # Main components
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, padding_mode=padding_mode, bias=True)

        if downscale:
            self.conv2 = nn.Conv2d(
                in_channels, out_channels, 4, stride=2, padding=1, padding_mode=padding_mode, bias=False)

            self.id_avg = nn.AvgPool2d(2, stride=2)

        else:
            self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode=padding_mode, bias=False)
            self.id_avg = None

        if out_channels != in_channels:
            self.id_ext = nn.Conv2d(in_channels, out_channels-in_channels, 1, bias=False)

        else:
            self.id_ext = None

        # Fixup components and init
        self.pre_act_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.pre_conv1_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.pre_conv2_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.id_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))

        nn.init.normal_(self.conv1.weight, mean=0., std=init_std(self.conv1, n_blocks=n_blocks))
        nn.init.constant_(self.conv1.bias, 0.)
        nn.init.constant_(self.conv2.weight, 0.)

        if self.id_ext is not None:
            nn.init.normal_(self.id_ext.weight, mean=0., std=0.05)

        # SE components
        self.use_se = bool(squeeze_ratio)

        if self.use_se:
            self.sigmoid = nn.Sigmoid()
            self.squeeze = nn.Conv2d(out_channels, out_channels // squeeze_ratio, 1, bias=True)
            self.excite = nn.Conv2d(out_channels // squeeze_ratio, out_channels, 1, bias=True)

            nn.init.normal_(self.squeeze.weight, mean=0., std=0.03)
            nn.init.normal_(self.excite.weight, mean=0., std=0.03)

        else:
            self.sigmoid = None
            self.squeeze = None
            self.excite = None

    def forward(self, x):
        x_preact = self.silu(x + self.pre_act_bias) + self.pre_conv1_bias
        x_res = self.conv1(x_preact)  # Bias in x_preact
        x_res = self.silu(x_res)  # Bias in conv1
        x_res = self.conv2(x_res + self.pre_conv2_bias)

        if self.use_se:
            # Depthwise global average pooling
            x_se = torch.mean(x_res, dim=(2, 3), keepdim=True)

            # Pointwise squeeze and excitation
            x_se = self.squeeze(x_se)
            x_se = self.silu(x_se)
            x_se = self.excite(x_se)

            # Pointwise gating
            x_res = x_res * self.sigmoid(x_se)

        # Modify identity to allow addition
        if self.id_avg is not None:
            x = self.id_avg(x)

        if self.id_ext is not None:
            x_preact = x_preact if self.id_avg is None else self.id_avg(x_preact)
            x_ext = self.id_ext(x_preact)  # Bias in x_preact
            x = torch.cat((x, x_ext), dim=1)

        return x * self.id_scale + x_res


class SEResBlock1d(nn.Module):
    """
    Basic residual block with optional squeeze and excitation, modified for
    pre-activation and to use Fixup-like initialisation and parameters.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding_mode: str = 'replicate',
        downscale: bool = False,
        squeeze_ratio: bool = 0,
        n_blocks: int = None
    ):
        super().__init__()

        assert out_channels >= in_channels, \
            f'Cannot preserve identity with given channel mismatch: {in_channels} and {out_channels}.'

        # Activation
        self.silu = nn.SiLU()

        # Main components
        self.conv1 = nn.Conv1d(in_channels, in_channels, 3, padding=1, padding_mode=padding_mode, bias=True)

        if downscale:
            self.conv2 = nn.Conv1d(
                in_channels, out_channels, 4, stride=2, padding=1, padding_mode=padding_mode, bias=False)

            self.id_avg = nn.AvgPool1d(2, stride=2)

        else:
            self.conv2 = nn.Conv1d(in_channels, out_channels, 3, padding=1, padding_mode=padding_mode, bias=False)
            self.id_avg = None

        if out_channels != in_channels:
            self.id_ext = nn.Conv1d(in_channels, out_channels-in_channels, 1, bias=False)

        else:
            self.id_ext = None

        # Fixup components and init
        self.pre_act_bias = nn.Parameter(torch.zeros(1, in_channels, 1))
        self.pre_conv1_bias = nn.Parameter(torch.zeros(1, in_channels, 1))
        self.pre_conv2_bias = nn.Parameter(torch.zeros(1, in_channels, 1))
        self.id_scale = nn.Parameter(torch.ones(1, out_channels, 1))

        nn.init.normal_(self.conv1.weight, mean=0., std=init_std(self.conv1, n_blocks=n_blocks))
        nn.init.constant_(self.conv1.bias, 0.)
        nn.init.constant_(self.conv2.weight, 0.)

        if self.id_ext is not None:
            nn.init.normal_(self.id_ext.weight, mean=0., std=0.05)

        # SE components
        self.use_se = bool(squeeze_ratio)

        if self.use_se:
            self.sigmoid = nn.Sigmoid()
            self.squeeze = nn.Conv1d(out_channels, out_channels // squeeze_ratio, 1, bias=True)
            self.excite = nn.Conv1d(out_channels // squeeze_ratio, out_channels, 1, bias=True)

            nn.init.normal_(self.squeeze.weight, mean=0., std=0.03)
            nn.init.normal_(self.excite.weight, mean=0., std=0.03)

        else:
            self.sigmoid = None
            self.squeeze = None
            self.excite = None

    def forward(self, x):
        x_preact = self.silu(x + self.pre_act_bias) + self.pre_conv1_bias
        x_res = self.conv1(x_preact)  # Bias in x_preact
        x_res = self.silu(x_res)  # Bias in conv1
        x_res = self.conv2(x_res + self.pre_conv2_bias)

        if self.use_se:
            # Depthwise global average pooling
            x_se = torch.mean(x_res, dim=2, keepdim=True)

            # Pointwise squeeze and excitation
            x_se = self.squeeze(x_se)
            x_se = self.silu(x_se)
            x_se = self.excite(x_se)

            # Pointwise gating
            x_res = x_res * self.sigmoid(x_se)

        # Modify identity to allow addition
        if self.id_avg is not None:
            x = self.id_avg(x)

        if self.id_ext is not None:
            x_preact = x_preact if self.id_avg is None else self.id_avg(x_preact)
            x_ext = self.id_ext(x_preact)  # Bias in x_preact
            x = torch.cat((x, x_ext), dim=1)

        return x * self.id_scale + x_res


class FiLMAtten2d(nn.Module):
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
        nn.init.normal_(self.qkv_extraction.weight, mean=0., std=0.05)
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
        x_preact = self.silu(x + self.pre_act_bias) + self.pre_qkv_bias
        x_res = self.qkv_extraction(x_preact)  # Bias in x_preact

        x_res = self.film_attention(x_res, mod_multiplier, mod_bias)  # Bias in qkv
        # Role of activation satisfied by mult. with softmax in film_attention

        return torch.cat((x_preact, x_res), dim=1)


class IRLinearCell(nn.Module):
    """
    Multi-actor implementation of an independently recurrent neural cell (2 layers)
    with linear processing of inputs and custom (log-like) activation.

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

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.activation = silog

        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.u1 = nn.Parameter(torch.empty(1, hidden_size))

        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.u2 = nn.Parameter(torch.empty(1, hidden_size))

        nn.init.normal_(self.fc1.weight, mean=0., std=0.01)
        nn.init.normal_(self.fc2.weight, mean=0., std=0.01)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)
        nn.init.uniform_(self.u1, a=0., b=1.)
        nn.init.uniform_(self.u2, a=0., b=1.)

        self.store_1 = StateStore(hidden_size)
        self.store_2 = StateStore(hidden_size)

    def forward(self, x: torch.Tensor, keys: Iterable[Hashable]) -> torch.Tensor:
        # Load recurrent states
        hidden_states_1 = self.store_1.get(keys=keys)
        hidden_states_2 = self.store_2.get(keys=keys)

        # Infer new recurrent states
        hidden_states_1 = self.activation(self.fc1(x) + self.u1 * hidden_states_1)
        hidden_states_2 = self.activation(self.fc2(hidden_states_1) + self.u2 * hidden_states_2)

        # Save new recurrent states
        self.store_1.append(hidden_states_1)
        self.store_2.append(hidden_states_2)

        return hidden_states_2


class IRConv2dCell(nn.Module):
    """
    Multi-actor implementation of an independently recurrent neural cell (2 layers)
    with (2D) convolution of inputs and custom (log-like) activation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        height: int,
        width: int,
        padding_mode: str = 'zeros',
    ):
        super().__init__()

        self.activation = silog

        self.conv_1 = nn.Conv2d(
            input_size,
            hidden_size,
            kernel_size,
            padding=kernel_size//2,
            bias=True,
            padding_mode=padding_mode)

        self.conv_2 = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size,
            padding=kernel_size//2,
            bias=True,
            padding_mode=padding_mode)

        self.u1 = nn.Parameter(torch.empty(1, hidden_size, height, width))
        self.u2 = nn.Parameter(torch.empty(1, hidden_size, height, width))

        nn.init.normal_(self.conv_1.weight, mean=0., std=0.01)
        nn.init.normal_(self.conv_2.weight, mean=0., std=0.01)
        nn.init.constant_(self.conv_1.bias, 0.)
        nn.init.constant_(self.conv_2.bias, 0.)
        nn.init.constant_(self.u1, 0.5)
        nn.init.constant_(self.u2, 0.5)

        self.store_1 = StateStore((hidden_size, height, width))
        self.store_2 = StateStore((hidden_size, height, width))

    def forward(self, x: torch.Tensor, keys: Iterable[Hashable]) -> torch.Tensor:
        # Load recurrent states
        hidden_states_1 = self.store_1.get(keys=keys)
        hidden_states_2 = self.store_2.get(keys=keys)

        # Infer new recurrent states
        hidden_states_1 = self.activation(self.conv_1(x) + self.u1 * hidden_states_1)
        hidden_states_2 = self.activation(self.conv_2(hidden_states_1) + self.u2 * hidden_states_2)

        # Save new recurrent states
        self.store_1.append(hidden_states_1)
        self.store_2.append(hidden_states_2)

        return hidden_states_2
