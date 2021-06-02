"""Components and assembly of PCNet"""

from typing import Hashable, Iterable, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from sdgai.utils import ChainLock, get_n_params, batch_crop
from sdgai.layers import XTraPool, FMBConv2d, SEResBlock2d, SASAtten1d, FiLMSAtten2d, MALSTMCell, MAConvLSTMCell


class PrimaryVisualEncoding(nn.Module):
    """
    Performs the first stages of visual processing.

    As it fully processes images that correspond to the entire screen,
    details in its output will be inevitably lost or only roughly described,
    but the overall 'picture' of the scene and its main landmarks should be
    sufficiently accounted for.

    In attempt to emphasise absolute positions on the screen, the final (global)
    pooling layer can be trained to consider spatial relations to an extent.
    """

    HEIGHT = 144
    WIDTH = 256
    FINAL_HEIGHT = HEIGHT // 2**4
    FINAL_WIDTH = WIDTH // 2**4
    FINAL_DEPTH = 96
    ENC_SIZE = 192
    N_BLOCKS = 5

    def __init__(self):
        super().__init__()

        # 256x144x3 -> 128x72x32
        self.stem = nn.Conv2d(3, 32, 6, stride=2, padding=2, bias=False)

        self.core = nn.Sequential(
            # 128x72x32 -> 64x36x48
            FMBConv2d(32, 48, downscale=True, n_blocks=self.N_BLOCKS),
            # 64x36x48 -> 32x18x64
            FMBConv2d(48, 64, downscale=True, n_blocks=self.N_BLOCKS),
            # 32x18x64 -> 16x9x96
            FMBConv2d(64, 96, downscale=True, n_blocks=self.N_BLOCKS),
            SEResBlock2d(96, 96, squeeze_ratio=4, n_blocks=self.N_BLOCKS),
            SEResBlock2d(96, 96, squeeze_ratio=4, n_blocks=self.N_BLOCKS))

        # 16x9x96 -> 1x1xO
        self.pool = XTraPool(self.FINAL_DEPTH, self.ENC_SIZE, (self.FINAL_HEIGHT, self.FINAL_WIDTH))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x_stem = self.stem(x)
        x_visual = self.core(x_stem)

        x_visual_enc = self.pool(x_visual)
        x_visual_enc = torch.flatten(x_visual_enc, start_dim=1)

        return x_stem, x_visual, x_visual_enc


class FocusedVisualEncoding(nn.Module):
    """
    Performs visual processing on an image that has already gone through a
    convolutional stem, corresponding to a cutout around a point of focus.

    Cutout size is determined by the size of icons (15x15) at post-stem conv.
    resolution (approx. 7x7) with buffer given at the borders to compensate for
    kernel size 3 (1 on each side).
    """

    CROP_LENGTH = 9
    FINAL_DEPTH = 96
    ENC_SIZE = FINAL_DEPTH + PrimaryVisualEncoding.FINAL_DEPTH
    N_BLOCKS = 5

    def __init__(self):
        super().__init__()

        # 9x9x32 -> 9x9x64 -> 9x9x128
        self.core = nn.Sequential(
            FMBConv2d(32, 64, n_blocks=self.N_BLOCKS),
            SEResBlock2d(64, 64, squeeze_ratio=4, n_blocks=self.N_BLOCKS),
            SEResBlock2d(64, 64, squeeze_ratio=4, n_blocks=self.N_BLOCKS),
            SEResBlock2d(64, 64, squeeze_ratio=4, n_blocks=self.N_BLOCKS),
            FMBConv2d(64, 96, n_blocks=self.N_BLOCKS))

        self.silu = nn.SiLU()
        self.pre_act_bias = nn.Parameter(torch.zeros(1, self.FINAL_DEPTH + PrimaryVisualEncoding.FINAL_DEPTH, 1, 1))

        self.batch_indices = torch.arange(1, dtype=torch.long)[:, None, None]
        self.length_indices = torch.arange(self.CROP_LENGTH, dtype=torch.long)
        self.single_index = torch.arange(1, dtype=torch.long)

    def move(self, device: Union[str, torch.device]):
        """Move cached indices to new device."""

        self.batch_indices = self.batch_indices.to(device=device)
        self.length_indices = self.length_indices.to(device=device)
        self.single_index = self.single_index.to(device=device)

    def forward(self, x_stem: torch.Tensor, x_visual: torch.Tensor, focal_points: torch.Tensor) -> torch.Tensor:
        # Update cached indices
        if self.batch_indices.shape[0] != focal_points.shape[0]:
            self.batch_indices = torch.arange(
                focal_points.shape[0], dtype=torch.long, device=self.batch_indices.device)[:, None, None]

        # Get 9x9 cutout
        x_stem = batch_crop(
            x_stem,
            focal_points,
            length=self.CROP_LENGTH,
            batch_indices=self.batch_indices,
            length_indices=self.length_indices)

        # Get 1x1 cutout
        x_visual = batch_crop(
            x_visual,
            (focal_points / 8).to(torch.long),
            length=1,
            batch_indices=self.batch_indices,
            length_indices=self.single_index)

        # Process, pool, and add features from primary branch
        x_focused = self.core(x_stem)
        x_focused = torch.mean(x_focused, dim=(2, 3), keepdim=True)
        x_focused = torch.cat((x_focused, x_visual), dim=1)

        x_focused = self.silu(x_focused + self.pre_act_bias)
        x_focused = torch.flatten(x_focused, start_dim=1)

        return x_focused


class SpatioTemporalCentre(nn.Module):
    """
    Performs spatio-temporal contextualisation. Intended to allow the network
    to compensate the delay between observations and reactions by predicting
    the effective state, in which the action will be implemented.
    """

    HEIGHT = PrimaryVisualEncoding.FINAL_HEIGHT
    WIDTH = PrimaryVisualEncoding.FINAL_WIDTH
    INPUT_SIZE = PrimaryVisualEncoding.FINAL_DEPTH
    HIDDEN_SIZE = 96
    ENC_SIZE = 192
    KERNEL_SIZE = 3

    def __init__(self):
        super().__init__()

        self.silu = nn.SiLU()
        self.pre_act_bias = nn.Parameter(torch.zeros(1, self.INPUT_SIZE, 1, 1))

        self.conv_lstm_cell = MAConvLSTMCell(
            self.INPUT_SIZE, self.HIDDEN_SIZE, self.KERNEL_SIZE, self.HEIGHT, self.WIDTH)

        # 16x9x96 -> 1x1xO
        self.pool = XTraPool(self.HIDDEN_SIZE, self.ENC_SIZE, (self.HEIGHT, self.WIDTH), preactivate=False)

    def forward(self, x_visual: torch.Tensor, actor_keys: Iterable[Hashable]) -> Tuple[torch.Tensor]:
        x_visual = self.silu(x_visual + self.pre_act_bias)
        x_visual = self.conv_lstm_cell(x_visual, actor_keys)

        x_visual_enc = self.pool(x_visual)
        x_visual_enc = torch.flatten(x_visual_enc, start_dim=1)

        return x_visual, x_visual_enc


class SpectralAudioEncoding(nn.Module):
    """
    Processes pairs of spectral vectors (slices of a 'running' spectrogram).
    Specifically, it focuses on auditory entities with a local convolution and
    global inter-frequency attention.

    In a later component, the extracted feature vector is put through an LSTM
    for spectro-temporal analysis and contextualisation.
    """

    WIDTH = 64
    ENC_SIZE = 96
    N_BLOCKS = 5

    def __init__(self):
        super().__init__()

        self.blocks = nn.Sequential(
            # 64x2 -> 32x32
            nn.Conv1d(2, 32, 6, stride=2, padding=2, bias=False, padding_mode='replicate'),
            # 32x32 -> 16x32
            nn.MaxPool1d(2, stride=2),
            # 16x32 -> 16x64
            SASAtten1d(32, 32, self.WIDTH//4, expansion_ratio=4, n_heads=8, n_blocks=self.N_BLOCKS),
            SASAtten1d(32, 64, self.WIDTH//4, expansion_ratio=4, n_heads=8, n_blocks=self.N_BLOCKS),
            # 16x64 -> 16x96
            SASAtten1d(64, 64, self.WIDTH//4, expansion_ratio=4, n_heads=8, n_blocks=self.N_BLOCKS),
            SASAtten1d(64, 64, self.WIDTH//4, expansion_ratio=4, n_heads=8, n_blocks=self.N_BLOCKS),
            SASAtten1d(64, 96, self.WIDTH//4, expansion_ratio=4, n_heads=8, n_blocks=self.N_BLOCKS),
            # 16x96 -> 1xO
            XTraPool(96, self.ENC_SIZE, self.WIDTH//4, dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = torch.flatten(x, start_dim=1)
        return x


class AttentionCentre(nn.Module):
    """
    Applies global attention to a visual encoding.

    Attention is modulated by the perceived and partially processed observed
    state (based on inputs besides the visual), which should allow the network
    to (passively or actively) focus on different visual entities, depending on
    the context, and to be capable of reflexes.
    """

    HEIGHT = SpatioTemporalCentre.HEIGHT
    WIDTH = SpatioTemporalCentre.WIDTH
    ENC_SIZE = 192
    FEAT_SIZE = SpatioTemporalCentre.HIDDEN_SIZE
    EXP_RATIO = 1
    N_HEADS = 6
    N_BLOCKS = 2

    def __init__(self, input_state_size: int):
        super().__init__()

        self.modulation_size = self.FEAT_SIZE * self.EXP_RATIO

        self.tanh = nn.Tanh()
        self.modulation = nn.Linear(input_state_size, self.modulation_size*2)
        nn.init.constant_(self.modulation.bias, 0.)

        self.film_attention = FiLMSAtten2d(
            self.FEAT_SIZE, self.FEAT_SIZE, self.HEIGHT, self.WIDTH,
            expansion_ratio=self.EXP_RATIO, n_heads=self.N_HEADS, n_blocks=self.N_BLOCKS)

        # 16x9x96 -> 1x1xO
        self.pool = XTraPool(self.FEAT_SIZE, self.ENC_SIZE, (self.HEIGHT, self.WIDTH))

    def forward(self, x_visual: torch.Tensor, x_state: torch.Tensor) -> Tuple[torch.Tensor]:

        mod = self.modulation(x_state)
        mod = self.tanh(mod)
        mod_multiplier, mod_bias = torch.split(mod, self.modulation_size, dim=1)

        x_visual = self.film_attention(x_visual, mod_multiplier, mod_bias)

        x_visual_enc = self.pool(x_visual)
        x_visual_enc = torch.flatten(x_visual_enc, start_dim=1)

        x_state = torch.cat((x_visual_enc, x_state), dim=1)

        return x_visual, x_state


class MotorCentre(nn.Module):
    """
    Interpretes/decodes the action state to produce an action vector and
    a focus heatmap, i.e. probabilities of key presses or of the cursor being
    moved for a specific amount and a spatial distribution of probabilities
    that the focus should be at some point.
    """

    HEIGHT = AttentionCentre.HEIGHT
    WIDTH = AttentionCentre.WIDTH
    FEAT_SIZE = AttentionCentre.FEAT_SIZE
    ACTION_BASE_SIZE = 96
    ACTION_SIZE = 72

    KERNEL_SIZE_1 = 3
    KERNEL_SIZE_2 = 5
    KERNEL_SIZE_3 = 7
    KERNEL_UNROLL_1 = KERNEL_SIZE_1**2
    KERNEL_UNROLL_2 = KERNEL_SIZE_2**2
    KERNEL_UNROLL_3 = KERNEL_SIZE_3**2

    RES_SPLIT = (FEAT_SIZE, HEIGHT*WIDTH, ACTION_BASE_SIZE)
    KERNEL_SPLIT = (KERNEL_UNROLL_1, KERNEL_UNROLL_2)

    def __init__(self, input_state_size: int):
        super().__init__()

        self.silu = nn.SiLU()
        self.tanh = nn.Tanh()

        # 384 -> 336 (96 + 16x9 + 96)
        self.base_fc = nn.Linear(input_state_size, self.FEAT_SIZE + self.HEIGHT*self.WIDTH + self.ACTION_BASE_SIZE)

        # 240 (16x9 + 96) -> 34 (3x3 + 5x5)
        self.kernel_fc_1 = nn.Linear(
            self.HEIGHT*self.WIDTH + self.ACTION_BASE_SIZE,
            self.KERNEL_UNROLL_1 + self.KERNEL_UNROLL_2)

        # 130 (34 + 96) -> 49 (7x7)
        self.kernel_fc_2 = nn.Linear(
            self.KERNEL_UNROLL_1 + self.KERNEL_UNROLL_2 + self.ACTION_BASE_SIZE,
            self.KERNEL_UNROLL_3)

        # 480 (384 + 96) -> 72
        self.action_fc = nn.Linear(input_state_size + self.ACTION_BASE_SIZE, self.ACTION_SIZE)

        nn.init.constant_(self.base_fc.bias, 0.)
        nn.init.constant_(self.kernel_fc_1.bias, 0.)
        nn.init.constant_(self.kernel_fc_2.bias, 0.)
        nn.init.constant_(self.action_fc.bias, 0.)

        self.interp_size_1 = (self.HEIGHT*2, self.WIDTH*2)  # 32x18
        self.interp_size_2 = (self.HEIGHT*4, self.WIDTH*4)  # 64x36
        self.interp_size_3 = (self.HEIGHT*8, self.WIDTH*8)  # 128x72

    def refine(self, x: torch.Tensor, kernels: torch.Tensor, kernel_size: int, interp_size: Tuple[int]) -> torch.Tensor:
        """Shape an interpolated base according to a given kernel."""

        x = F.interpolate(x, interp_size, mode='nearest')

        b, _, h, w = x.shape

        # Out channels, in channels, dimensions
        kernels = kernels.view(b, 1, kernel_size, kernel_size)

        x_res = self.silu(x)
        x_res = x_res.view(1, b, h, w)

        # Need to use `F.pad` for modes besides `zeros`
        x_res = F.pad(x_res, (kernel_size // 2,) * 4, mode='replicate')

        x_res = F.conv2d(x_res, kernels, groups=b)
        x_res = x_res.reshape(b, 1, h, w)

        return x + x_res

    def forward(self, x_visual: torch.Tensor, x_state: torch.Tensor) -> Tuple[torch.Tensor]:
        focus_res = self.base_fc(x_state)
        base_query, base_focus, fine_action = torch.split(focus_res, self.RES_SPLIT, dim=1)

        base_query = self.tanh(base_query)
        base_focus = self.tanh(base_focus)
        fine_action = self.silu(fine_action)

        focus_res = self.kernel_fc_1(torch.cat((base_focus, fine_action), dim=1))
        focus_res = self.tanh(focus_res)
        kernel_weights_1, kernel_weights_2 = torch.split(focus_res, self.KERNEL_SPLIT, dim=1)

        kernel_weights_3 = self.kernel_fc_2(torch.cat((focus_res, fine_action), dim=1))
        kernel_weights_3 = self.tanh(kernel_weights_3)

        # Mouse/keyboard actions
        x_action = self.action_fc(torch.cat((x_state, fine_action), dim=1))

        # Get base focus
        b, c, h, w = x_visual.shape

        base_query = base_query.reshape(b, self.FEAT_SIZE, 1, 1)
        base_focus = base_focus.reshape(1, b, self.HEIGHT, self.WIDTH)

        x_focus = x_visual.view(1, b*c, h, w)
        x_focus = F.conv2d(x_focus, base_query, groups=b) + base_focus
        x_focus = x_focus.reshape(b, 1, h, w)

        # Refine focus in a series of upscalings and adaptive convolutions
        x_focus = self.refine(x_focus, kernel_weights_1, self.KERNEL_SIZE_1, self.interp_size_1)
        x_focus = self.refine(x_focus, kernel_weights_2, self.KERNEL_SIZE_2, self.interp_size_2)
        x_focus = self.refine(x_focus, kernel_weights_3, self.KERNEL_SIZE_3, self.interp_size_3)

        return x_focus, x_action


class PCNet(nn.Module):
    """
    Peripheral-compatible recurrent neural network.

    The name is intended to be evocative of personal computer or player
    character, but otherwise refers to its interface with audio-visual inputs
    and mouse/keyboard outputs.

    Layers of the network take different amounts of time to process, and,
    along with external factors, there is a possibility of racing conditions
    in multi-threaded inference: since processing for a call of some actor
    directly depends on the result of its immediate predecessor, it must not be
    allowed to overtake it.

    The prevent this, the forward pass is divided into 6 parts, corresponding to
    the intended number of inference worker threads, where access to each part
    is controlled through the use of chain locks to maintain execution in FIFO
    order. Similar reasoning can be applied to refreshing model parameters on
    the fly.
    """

    N_FPS = 30
    N_DELAY = 6
    N_WORKERS = N_DELAY
    STATE_LSTM_SIZE = 384
    INTENT_LSTM_SIZE = 384
    MAX_FOCAL_OFFSET = float(PrimaryVisualEncoding.WIDTH)
    MKBD_ENC_SIZE = 22

    def __init__(self):
        super().__init__()

        input_state_size = PrimaryVisualEncoding.ENC_SIZE + SpectralAudioEncoding.ENC_SIZE + self.MKBD_ENC_SIZE
        ctx_state_size = SpatioTemporalCentre.ENC_SIZE + FocusedVisualEncoding.ENC_SIZE + self.STATE_LSTM_SIZE
        attended_state_size = ctx_state_size + AttentionCentre.ENC_SIZE

        # Cognitive centres
        self.primary_visual_encoding = PrimaryVisualEncoding()
        self.focused_visual_encoding = FocusedVisualEncoding()
        self.spectral_audio_encoding = SpectralAudioEncoding()

        self.temporal_centre = MALSTMCell(input_state_size, self.STATE_LSTM_SIZE, init_horizon=self.N_DELAY)
        self.spatio_temporal_centre = SpatioTemporalCentre()
        self.attention_centre = AttentionCentre(ctx_state_size)
        self.intent_inference = MALSTMCell(attended_state_size, self.INTENT_LSTM_SIZE, init_horizon=(self.N_FPS//2))
        self.motor_decoding = MotorCentre(self.INTENT_LSTM_SIZE)

        # Mouse/key input adjustments
        self.mkbd_scale = nn.Parameter(torch.ones(1, self.MKBD_ENC_SIZE))
        self.mkbd_bias = nn.Parameter(torch.zeros(1, self.MKBD_ENC_SIZE))

        # Locks for multi-threaded inference
        self.lock_1 = ChainLock(self.N_WORKERS)
        self.lock_2 = ChainLock(self.N_WORKERS)
        self.lock_3 = ChainLock(self.N_WORKERS)
        self.lock_4 = ChainLock(self.N_WORKERS)
        self.lock_5 = ChainLock(self.N_WORKERS)
        self.lock_6 = ChainLock(self.N_WORKERS)

    def get_n_params(self, trainable: bool = True) -> int:
        """Get the number of (trainable) parameters in the instantiated model."""

        return get_n_params(self, trainable=trainable)

    def clear(self, keys: Iterable[Hashable] = None):
        """Clear final (first) hidden/cell states of LSTM cells for TBPTT."""

        self.temporal_centre.clear(keys=keys)
        self.spatio_temporal_centre.conv_lstm_cell.clear(keys=keys)
        self.intent_inference.clear(keys=keys)

    def detach(self, keys: Iterable[Hashable] = None):
        """Detach final (first) hidden/cell states of LSTM cells for TBPTT."""

        self.temporal_centre.detach(keys=keys)
        self.spatio_temporal_centre.conv_lstm_cell.detach(keys=keys)
        self.intent_inference.detach(keys=keys)

    def move(self, device: Union[str, torch.device]) -> 'PCNet':
        """Move model parameters and initial/default states to new device."""

        self.focused_visual_encoding.move(device)
        self.temporal_centre.move(device)
        self.spatio_temporal_centre.conv_lstm_cell.move(device)
        self.intent_inference.move(device)

        return self.to(device=device)

    def load(self, path: str, device: str = None) -> 'PCNet':
        """Load model parameters from a state dict at the specified path."""

        if device is not None:
            device = torch.device(device)

        self.load_state_dict(torch.load(path, map_location=device))

        if device is not None:
            return self.to(device=device)

        return self

    def save(self, path: str):
        """Save model parameters to a state dict at the specified path."""

        torch.save(self.state_dict(), path)

    def forward(
        self,
        x_visual: torch.Tensor,
        x_audio: torch.Tensor,
        x_mkbd: torch.Tensor,
        focal_points: torch.Tensor,
        actor_keys: Iterable[Hashable],
        return_state: bool = False
    ) -> Tuple[torch.Tensor]:

        # Primary visual encoding
        with self.lock_1:
            x_visual_stem, x_visual, x_visual_enc = self.primary_visual_encoding(x_visual)

        # Fine visual encoding
        with self.lock_2:
            x_focused_enc = self.focused_visual_encoding(x_visual_stem, x_visual, focal_points)

        # Audio encoding and temporal contextualisation
        with self.lock_3:
            x_audio = self.spectral_audio_encoding(x_audio)

            x_mkbd = torch.hstack((x_mkbd, focal_points / self.MAX_FOCAL_OFFSET))
            x_mkbd = x_mkbd * self.mkbd_scale + self.mkbd_bias

            x_state = torch.hstack((x_visual_enc, x_audio, x_mkbd))
            x_state = self.temporal_centre(x_state, actor_keys)

        # Spatio-temporal contextualisation
        with self.lock_4:
            x_visual, x_visual_enc = self.spatio_temporal_centre(x_visual, actor_keys)

        # Attention
        with self.lock_5:
            x_state = torch.hstack((x_visual_enc, x_focused_enc, x_state))
            x_visual, x_state = self.attention_centre(x_visual, x_state)

        # Action inference and motor decoding
        with self.lock_6:
            x_state = self.intent_inference(x_state, actor_keys)
            x_focus, x_action = self.motor_decoding(x_visual, x_state)

        # State can be used for a critic/advantage estimator head in RL, but not needed for inference
        if return_state:
            return x_focus, x_action, x_state

        return x_focus, x_action
