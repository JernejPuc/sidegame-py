"""Basic building blocks of PCNet components."""

from typing import Dict, Hashable, Iterable, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from sdgai.utils import init_std


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

        nn.init.normal_(self.multi_depthwise.weight, mean=0., std=init_std(self.multi_depthwise))
        nn.init.normal_(self.projection.weight, mean=0., std=init_std(self.projection))

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
            nn.init.normal_(self.id_ext.weight, mean=0., std=init_std(self.id_ext))

        # SE components
        self.use_se = bool(squeeze_ratio)

        if self.use_se:
            squeezed_channels = int(exp_channels // squeeze_ratio)

            self.sigmoid = nn.Sigmoid()
            self.squeeze = nn.Conv2d(exp_channels, squeezed_channels, 1, bias=True)
            self.excitation = nn.Conv2d(squeezed_channels, exp_channels, 1, bias=True)

            nn.init.normal_(self.squeeze.weight, mean=0., std=init_std(self.squeeze))
            nn.init.normal_(self.excitation.weight, mean=0., std=init_std(self.excitation))

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
            nn.init.normal_(self.id_ext.weight, mean=0., std=init_std(self.id_ext))

        # SE components
        self.use_se = bool(squeeze_ratio)

        if self.use_se:
            self.sigmoid = nn.Sigmoid()
            self.squeeze = nn.Conv2d(out_channels, out_channels // squeeze_ratio, 1, bias=True)
            self.excite = nn.Conv2d(out_channels // squeeze_ratio, out_channels, 1, bias=True)

            nn.init.normal_(self.squeeze.weight, mean=0., std=init_std(self.squeeze))
            nn.init.normal_(self.excite.weight, mean=0., std=init_std(self.excite))

        else:
            self.sigmoid = None
            self.squeeze = None
            self.excite = None

    def forward(self, x):
        x_preact = self.silu(x + self.pre_act_bias) + self.pre_conv1_bias
        x_res = self.conv1(x_preact)  # Bias in x_preact
        x_res = self.silu(x_res)  # Bias included in conv1
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


class SASAtten1d(nn.Module):
    """
    Residual (inv.) bottleneck block with attention, pre-activation, and Fixup-like
    initialisation. Based on stand-alone and transformer self-attention models.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int,
        downscale: bool = False,
        expansion_ratio: int = 1,
        n_heads: int = 1,
        n_blocks: int = None
    ):
        super().__init__()

        assert out_channels >= in_channels, \
            f'Cannot preserve identity with given channel mismatch: {in_channels} and {out_channels}.'

        self.width = (width//2) if downscale else width
        self.feat_size = int(in_channels * expansion_ratio)
        exp_channels = self.feat_size * 3
        self.head_size = self.feat_size // n_heads
        self.n_heads = n_heads

        assert not (self.feat_size % n_heads), 'Expanded feature size must be divisible by `n_heads`.'

        # Activation
        self.silu = nn.SiLU()

        # Main components
        if downscale:
            self.id_avg = nn.AvgPool1d(2, stride=2)

        else:
            self.id_avg = None

        if out_channels != in_channels:
            self.id_ext = nn.Conv1d(in_channels, out_channels-in_channels, 1, bias=False)

        else:
            self.id_ext = None

        self.qkv_extraction = nn.Conv1d(in_channels, exp_channels, 1, bias=True)
        self.projection = nn.Conv1d(self.feat_size, out_channels, 1, bias=False)

        self.pos_embeddings_q = nn.Parameter(torch.zeros(1, self.feat_size, self.width))
        self.pos_embeddings_k = nn.Parameter(torch.zeros(1, self.feat_size, self.width))
        self.dot_scale = self.head_size**0.5

        # Fixup components and init
        self.pre_act_bias = nn.Parameter(torch.zeros(1, in_channels, 1))
        self.pre_qkv_bias = nn.Parameter(torch.zeros(1, in_channels, 1))
        self.pre_proj_bias = nn.Parameter(torch.zeros(1, self.feat_size, 1))
        self.id_scale = nn.Parameter(torch.ones(1, out_channels, 1))

        nn.init.normal_(self.qkv_extraction.weight, mean=0., std=init_std(self.qkv_extraction, n_blocks, 2))
        nn.init.constant_(self.qkv_extraction.bias, 0.)
        nn.init.constant_(self.projection.weight, 0.)

        if self.id_ext is not None:
            nn.init.normal_(self.id_ext.weight, mean=0., std=init_std(self.id_ext))

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-headed scaled dot product global self-attention."""

        # Extract queries, keys, and values
        q, k, v = torch.split(x, self.feat_size, dim=1)

        # Add absolute positional embeddings
        q = q + self.pos_embeddings_q
        k = k + self.pos_embeddings_k

        # Reshape for matrix ops.
        b = x.shape[0]

        q = q.view(b, self.n_heads, self.head_size, self.width)
        k = k.view(b, self.n_heads, self.head_size, self.width)
        v = v.view(b, self.n_heads, self.head_size, self.width)

        # b, n, c//n, w, w
        logits = torch.matmul(q, torch.transpose(k, 2, 3)) / self.dot_scale
        weights = F.softmax(logits, dim=-1)

        # b, n, c//n, w
        v = torch.matmul(weights, v)
        v = v.view(b, self.feat_size, self.width)

        return v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.id_avg is not None:
            x = self.id_avg(x)

        x_preact = self.silu(x + self.pre_act_bias) + self.pre_qkv_bias
        x_res = self.qkv_extraction(x_preact)  # Bias in x_preact

        x_res = self.attention(x_res)  # Bias in qkv
        # Role of activation satisfied by mult. with softmax in attention

        x_res = self.projection(x_res + self.pre_proj_bias)

        # Modify identity to allow addition
        if self.id_ext is not None:
            x_ext = self.id_ext(x_preact)  # Bias in x_preact
            x = torch.cat((x, x_ext), dim=1)

        return x * self.id_scale + x_res


class FiLMSAtten2d(nn.Module):
    """
    Residual (inv.) bottleneck block with modulated attention, pre-activation,
    and Fixup-like initialisation.

    Feature-wise linear modulation (FiLM) is used to add an external, global
    objective to the query by highlighting its pattern, e.g. emphasising
    positional tokens.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        downscale: bool = False,
        expansion_ratio: int = 1,
        n_heads: int = 1,
        n_blocks: int = None
    ):
        super().__init__()

        assert out_channels >= in_channels, \
            f'Cannot preserve identity with given channel mismatch: {in_channels} and {out_channels}.'

        self.height = (height//2) if downscale else height
        self.width = (width//2) if downscale else width
        self.feat_size = in_channels * expansion_ratio
        exp_channels = self.feat_size * 3
        self.head_size = self.feat_size // n_heads
        self.n_heads = n_heads

        assert not (self.feat_size % n_heads), 'Expanded feature size must be divisible by `n_heads`.'

        # Activation
        self.silu = nn.SiLU()

        # Main components
        if downscale:
            self.id_avg = nn.AvgPool2d(2, stride=2)

        else:
            self.id_avg = None

        if out_channels != in_channels:
            self.id_ext = nn.Conv2d(in_channels, out_channels-in_channels, 1, bias=False)

        else:
            self.id_ext = None

        self.qkv_extraction = nn.Conv2d(in_channels, exp_channels, 1, bias=True)
        self.projection = nn.Conv2d(self.feat_size, out_channels, 1, bias=False)

        self.pos_embeddings_q_h = nn.Parameter(torch.zeros(1, self.feat_size, self.height, 1))
        self.pos_embeddings_q_w = nn.Parameter(torch.zeros(1, self.feat_size, 1, self.width))
        self.pos_embeddings_k_h = nn.Parameter(torch.zeros(1, self.feat_size, self.height, 1))
        self.pos_embeddings_k_w = nn.Parameter(torch.zeros(1, self.feat_size, 1, self.width))
        self.dot_scale = self.head_size**0.5

        # Fixup components
        self.pre_act_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.pre_qkv_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.pre_proj_bias = nn.Parameter(torch.zeros(1, self.feat_size, 1, 1))
        self.id_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))

        # Fixup init
        nn.init.normal_(self.qkv_extraction.weight, mean=0., std=init_std(self.qkv_extraction, n_blocks, n_layers=2))
        nn.init.constant_(self.qkv_extraction.bias, 0.)
        nn.init.constant_(self.projection.weight, 0.)

        if self.id_ext is not None:
            nn.init.normal_(self.id_ext.weight, std=init_std(self.id_ext))

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
        if self.id_avg is not None:
            x = self.id_avg(x)

        x_preact = self.silu(x + self.pre_act_bias) + self.pre_qkv_bias
        x_res = self.qkv_extraction(x_preact)  # Bias in x_preact

        x_res = self.film_attention(x_res, mod_multiplier, mod_bias)  # Bias in qkv
        # Role of activation satisfied by mult. with softmax in film_attention

        # Pointwise projection
        x_res = self.projection(x_res + self.pre_proj_bias)

        # Modify identity to allow addition
        if self.id_ext is not None:
            x_ext = self.id_ext(x_preact)  # Bias in x_preact
            x = torch.cat((x, x_ext), dim=1)

        return x * self.id_scale + x_res


class MALSTMCell(nn.Module):
    """
    Multi-actor LSTM cell, allowing multiple input sequences to be processed
    semi-simultaneously.

    By specifying keys corresponding to different actor instances, concurrent or
    otherwise, associated hidden/cell states can be kept, retrieved, and updated
    in batches.

    NOTE: Truncated backpropagation through time (TBPTT) can be used if
    hidden/cell states are regularly detached from the computational graph
    and if batched sequences/trajectories are not sampled from the same actor
    instance (as their keys are not unique and their order is ambiguous).

    References for initialisation:
    - https://proceedings.mlr.press/v37/jozefowicz15.pdf
    - https://arxiv.org/abs/1804.11188
    - https://github.com/pytorch/pytorch/issues/750
    - https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        init_horizon: Union[int, float] = 6.,
        device: Union[str, torch.device] = None
    ):
        super().__init__()

        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.hidden_states: Dict[Hashable, torch.Tensor] = {}
        self.cell_states: Dict[Hashable, torch.Tensor] = {}
        self.default_state = torch.zeros(hidden_size, device=device)
        self.device = device

        with torch.no_grad():
            self.lstm_cell.bias_ih[hidden_size:2*hidden_size].uniform_(1., init_horizon-1.).log_().div_(2.)
            self.lstm_cell.bias_ih[:hidden_size] = -self.lstm_cell.bias_ih[hidden_size:2*hidden_size]
            self.lstm_cell.bias_ih[2*hidden_size:].fill_(0.)

            # NOTE: Redundant biases
            self.lstm_cell.bias_hh[:] = self.lstm_cell.bias_ih

    def move(self, device: Union[str, torch.device]):
        """Set new device and move initial/default state to it."""

        self.default_state = self.default_state.to(device=device)

    def clear(self, keys: Iterable[Hashable] = None):
        """Clear hidden/cell states."""

        if keys is None:
            keys = tuple(self.hidden_states.keys())

        for key in keys:
            del self.hidden_states[key]
            del self.cell_states[key]

    def detach(self, keys: Iterable[Hashable] = None):
        """Detach hidden/cell states."""

        if keys is None:
            keys = self.hidden_states.keys()

        for key in keys:
            self.hidden_states[key] = self.hidden_states[key].detach()
            self.cell_states[key] = self.cell_states[key].detach()

    def forward(self, x: torch.Tensor, keys: Iterable[Hashable]) -> torch.Tensor:
        # Load recurrent states
        hidden_states = torch.stack([self.hidden_states.get(key, self.default_state) for key in keys])
        cell_states = torch.stack([self.cell_states.get(key, self.default_state) for key in keys])

        # Infer new recurrent states
        hidden_states, cell_states = self.lstm_cell(x, (hidden_states, cell_states))

        # Save new recurrent states
        for key, hidden_state, cell_state in zip(keys, hidden_states, cell_states):
            self.hidden_states[key] = hidden_state
            self.cell_states[key] = cell_state

        return hidden_states


class MAConvLSTMCell(nn.Module):
    """
    Multi-actor ConvLSTM cell, allowing multiple input sequences to be processed
    semi-simultaneously.

    Reference:
    - https://arxiv.org/abs/1506.04214
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        height: int,
        width: int,
        padding_mode: str = 'zeros',
        init_horizon: Union[int, float] = 6.
    ):
        super().__init__()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.conv_input = nn.Conv2d(
            input_size,
            hidden_size*4,
            kernel_size,
            padding=kernel_size//2,
            bias=True,
            padding_mode=padding_mode)

        self.conv_hidden = nn.Conv2d(
            hidden_size,
            hidden_size*4,
            kernel_size,
            padding=kernel_size//2,
            bias=False,
            padding_mode=padding_mode)

        self.weight_ci = nn.Parameter(torch.zeros(1, hidden_size, height, width))
        self.weight_cf = nn.Parameter(torch.zeros(1, hidden_size, height, width))
        self.weight_co = nn.Parameter(torch.zeros(1, hidden_size, height, width))

        self.hidden_size = hidden_size
        self.hidden_states: Dict[Hashable, torch.Tensor] = {}
        self.cell_states: Dict[Hashable, torch.Tensor] = {}
        self.default_state = torch.zeros((hidden_size, height, width))

        with torch.no_grad():
            self.conv_input.bias[hidden_size:2*hidden_size].uniform_(1., init_horizon-1.).log_()
            self.conv_input.bias[:hidden_size] = -self.conv_input.bias[hidden_size:2*hidden_size]
            self.conv_input.bias[2*hidden_size:].fill_(0.)

    def move(self, device: Union[str, torch.device]):
        """Set new device and move initial/default state to it."""

        self.default_state = self.default_state.to(device=device)

    def clear(self, keys: Iterable[Hashable] = None):
        """Clear hidden/cell states."""

        if keys is None:
            keys = tuple(self.hidden_states.keys())

        for key in keys:
            del self.hidden_states[key]
            del self.cell_states[key]

    def detach(self, keys: Iterable[Hashable] = None):
        """Detach hidden/cell states."""

        if keys is None:
            keys = self.hidden_states.keys()

        for key in keys:
            self.hidden_states[key] = self.hidden_states[key].detach()
            self.cell_states[key] = self.cell_states[key].detach()

    def forward(self, x: torch.Tensor, keys: Iterable[Hashable]) -> torch.Tensor:
        # Load recurrent states
        hidden_states = torch.stack([self.hidden_states.get(key, self.default_state) for key in keys])
        cell_states = torch.stack([self.cell_states.get(key, self.default_state) for key in keys])

        # Infer new recurrent states
        conv_xi, conv_xf, conv_xc, conv_xo = torch.split(self.conv_input(x), self.hidden_size, dim=1)
        conv_hi, conv_hf, conv_hc, conv_ho = torch.split(self.conv_hidden(hidden_states), self.hidden_size, dim=1)
        had_ci = self.weight_ci * cell_states
        had_cf = self.weight_cf * cell_states

        i = self.sigmoid(conv_xi + conv_hi + had_ci)  # Bias in conv_xi
        f = self.sigmoid(conv_xf + conv_hf + had_cf)  # Bias in conv_xf
        cell_states = f * cell_states + i * self.tanh(conv_xc + conv_hc)  # Bias in conv_xc
        o = self.sigmoid(conv_xo + conv_ho + self.weight_co * cell_states)  # Bias in conv_xo
        hidden_states = o * self.tanh(cell_states)

        # Save new recurrent states
        for key, hidden_state, cell_state in zip(keys, hidden_states, cell_states):
            self.hidden_states[key] = hidden_state
            self.cell_states[key] = cell_state

        return hidden_states
