"""Components and assembly of PCNet"""

from typing import Hashable, Iterable, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Manager
from sdgai.utils import ChainLock, get_n_params, batch_crop, init_std, symlog
from sdgai.layers import GDWPool1d, FMBConv, BResBlock, PosConv, SASAtten, FiLMAtten, \
    IRLinearCell, IRConv2dCell, IRConv1dCell


class PrimaryVisualEncoding(nn.Module):
    """
    Performs the first stages of visual processing.

    As it processes images that correspond to the entire (in-game) screen,
    details in its output will be inevitably lost or only roughly described,
    but the overall 'picture' of the scene and its main landmarks should be
    able to be accounted for.
    """

    HEIGHT = 144
    WIDTH = 256
    DIV_FACTOR = 2**3
    FINAL_HEIGHT = HEIGHT // DIV_FACTOR
    FINAL_WIDTH = WIDTH // DIV_FACTOR
    FINAL_DEPTH = 96
    ENC_SIZE = 320
    N_BLOCKS = 9

    def __init__(self):
        super().__init__()

        # 256x144x3 -> 128x72x32
        self.stem = nn.Conv2d(3, 32, 6, stride=2, padding=2, bias=False)

        with torch.no_grad():
            rgb_mean = self.stem.weight.mean(dim=1, keepdim=False)
            self.stem.weight[:, 0] = rgb_mean
            self.stem.weight[:, 1] = rgb_mean
            self.stem.weight[:, 2] = rgb_mean

        self.core = nn.Sequential(
            # 128x72x32 -> 64x36x48
            FMBConv(32, 48, 4, downscale=True, n_blocks=self.N_BLOCKS),
            # 64x36x48 -> 32x18x64
            FMBConv(48, 64, 4, downscale=True, n_blocks=self.N_BLOCKS),
            FMBConv(64, 64, 3, n_blocks=self.N_BLOCKS),
            # 32x18x64 -> 32x18x96
            FMBConv(64, 96, 3, n_blocks=self.N_BLOCKS),
            FMBConv(96, 96, 3, expansion_ratio=1.334, n_blocks=self.N_BLOCKS),
            FMBConv(96, 96, 3, expansion_ratio=1.334, n_blocks=self.N_BLOCKS))

        # 32x18x96 -> 1x1xO
        self.pool = nn.Sequential(
            PosConv(96, self.ENC_SIZE, self.FINAL_HEIGHT, self.FINAL_WIDTH),
            nn.MaxPool2d((self.FINAL_HEIGHT, self.FINAL_WIDTH)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.stem(x)
        x_visual = self.core(x)

        x_visual_enc = self.pool(x_visual)
        x_visual_enc = torch.flatten(x_visual_enc, start_dim=1)

        return x_visual, x_visual_enc


class FocusedVisualEncoding(nn.Module):
    """
    Performs visual processing on a cutout of an image around a point of focus.
    To alleviate redundancy, some features are inherited from primary encoding.
    """

    MAIN_CROP_SIZE = 24
    DEEP_CROP_SIZE = 3
    FINAL_CROP_SIZE = MAIN_CROP_SIZE // 2**2
    BASE_DEPTH = 32
    FINAL_DEPTH = BASE_DEPTH + PrimaryVisualEncoding.FINAL_DEPTH
    ENC_SIZE = 256
    DIV_FACTOR = PrimaryVisualEncoding.DIV_FACTOR
    N_BLOCKS = 6
    DEPADDING = 2

    def __init__(self):
        super().__init__()

        # 24x24x3 -> 24x24x32
        self.stem = nn.Conv2d(3, 32, 5, padding=2, bias=False)

        with torch.no_grad():
            rgb_mean = self.stem.weight.mean(dim=1, keepdim=False)
            self.stem.weight[:, 0] = rgb_mean
            self.stem.weight[:, 1] = rgb_mean
            self.stem.weight[:, 2] = rgb_mean

        # 24x24x32 -> 24x24x32
        self.preproc = nn.Sequential(
            FMBConv(self.BASE_DEPTH, self.BASE_DEPTH, 3, n_blocks=self.N_BLOCKS),
            FMBConv(self.BASE_DEPTH, self.BASE_DEPTH, 3, n_blocks=self.N_BLOCKS),
            FMBConv(self.BASE_DEPTH, self.BASE_DEPTH, 3, n_blocks=self.N_BLOCKS))

        self.fuse = nn.Sequential(
            # 24x24x128 -> 12x12x128
            BResBlock(self.FINAL_DEPTH, self.FINAL_DEPTH, downscale=True, n_blocks=self.N_BLOCKS),
            # 12x12x128 -> 6x6x128
            BResBlock(self.FINAL_DEPTH, self.FINAL_DEPTH, downscale=True, n_blocks=self.N_BLOCKS),
            # Incorporate surroundings before central crop
            SASAtten(self.FINAL_DEPTH, self.FINAL_DEPTH, self.FINAL_CROP_SIZE, self.FINAL_CROP_SIZE, 4, self.N_BLOCKS))

        self.pre_act_bias = nn.Parameter(torch.zeros(1, self.FINAL_DEPTH, 1, 1))
        self.silu = nn.SiLU()
        self.expand = nn.Conv2d(self.FINAL_DEPTH, self.ENC_SIZE, 1, bias=True)
        self.pool = nn.AvgPool2d(2)

        nn.init.normal_(self.expand.weight, mean=0., std=init_std(self.expand))
        self.interp_size = (self.MAIN_CROP_SIZE,)*2

    def forward(self, x: torch.Tensor, x_deep: torch.Tensor, focal_points: torch.Tensor) -> torch.Tensor:
        # Get 24x24 cutout from base image
        x = batch_crop(x, focal_points, length=self.MAIN_CROP_SIZE, naive=True)

        # Get 3x3 cutout from deep features
        focal_points = (focal_points // self.DIV_FACTOR).to(torch.long)
        x_deep = batch_crop(x_deep, focal_points, length=self.DEEP_CROP_SIZE, naive=True)

        # Extract high-res features
        x = self.stem(x)
        x = self.preproc(x)

        # Upscale deep feature cutout
        x_deep = F.interpolate(x_deep, self.interp_size, mode='nearest')

        # Fuse high-res and deep features
        x = torch.cat((x, x_deep), dim=1)
        x = self.fuse(x)

        # Extract 2x2 central features (surroundings incorporated with self-attention in fuse)
        x = x[..., self.DEPADDING:-self.DEPADDING, self.DEPADDING:-self.DEPADDING]

        x = self.silu(x + self.pre_act_bias)
        x = self.expand(x)
        x = self.silu(x)
        x = torch.mean(x, dim=(2, 3), keepdim=False)

        return x


class AudioEncoding(nn.Module):
    """
    Processes pairs of spectral vectors (slices of a 'running' spectrogram).
    Specifically, it focuses on auditory entities with a series of convolutions,
    where some are retaining memory for spectro-temporal analysis.
    """

    WIDTH = 64
    ENC_SIZE = 128
    N_BLOCKS = 3

    def __init__(self, manager: Manager = None):
        super().__init__()

        self.blocks = nn.Sequential(
            # 64x2 -> 32x16
            nn.Conv1d(2, 16, 6, stride=2, padding=2, bias=False, padding_mode='replicate'),
            # 32x16 -> 16x24
            FMBConv(16, 24, 6, padding=2, downscale=True, n_blocks=self.N_BLOCKS, dim=1),
            # 16x24 -> 16x32
            FMBConv(24, 32, 5, padding=2, n_blocks=self.N_BLOCKS, dim=1),
            # 16x32 -> 16x48
            FMBConv(32, 48, 5, padding=2, n_blocks=self.N_BLOCKS, dim=1))

        self.conv_mem_cell = IRConv1dCell(48, 48, 5, self.WIDTH//4, manager=manager)

        # 16x48 -> 1xO
        self.pool = GDWPool1d(48, self.ENC_SIZE, self.WIDTH//4)

        nn.init.uniform_(self.blocks[0].weight, a=(-1. / (6*16)**0.5), b=(1. / (6*16)**0.5))

        with torch.no_grad():
            lr_mean = self.blocks[0].weight.mean(dim=1, keepdim=False)
            self.blocks[0].weight[:, 0] = lr_mean
            self.blocks[0].weight[:, 1] = lr_mean

    def forward(self, x: torch.Tensor, actor_keys: Iterable[Hashable]) -> torch.Tensor:
        x = self.blocks(x)
        x = self.conv_mem_cell(x, actor_keys)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return x


class SpatioTemporalCentre(nn.Module):
    """
    Performs spatio-temporal contextualisation. Intended to allow the network
    to compensate the delay between observations and reactions by predicting
    the effective state, in which the action will be implemented.

    Features are modulated by the perceived and partially processed observed
    state (based on inputs besides the visual), which should allow the network
    to (passively or actively) focus on different visual entities, depending on
    the context, and to be capable of reflexes.
    """

    HEIGHT = PrimaryVisualEncoding.FINAL_HEIGHT
    WIDTH = PrimaryVisualEncoding.FINAL_WIDTH
    INPUT_SIZE = PrimaryVisualEncoding.FINAL_DEPTH
    HIDDEN_SIZE = 96
    ENC_SIZE = 320
    KERNEL_SIZE = 3

    def __init__(self, input_state_size: int, manager: Manager = None):
        super().__init__()

        self.silu = nn.SiLU()

        self.modulation = nn.Linear(input_state_size, self.INPUT_SIZE*2)
        nn.init.constant_(self.modulation.bias, 0.)

        self.attention = FiLMAtten(self.INPUT_SIZE, self.HEIGHT, self.WIDTH, n_heads=4)

        self.conv_mem_cell = IRConv2dCell(
            self.INPUT_SIZE*2, self.HIDDEN_SIZE, self.KERNEL_SIZE, self.HEIGHT, self.WIDTH, manager=manager)

        # 32x18x96 -> 1x1x320
        self.pool_enc = nn.Sequential(
            PosConv(self.HIDDEN_SIZE, self.ENC_SIZE, self.HEIGHT, self.WIDTH, preactivate=False),
            nn.MaxPool2d((self.HEIGHT, self.WIDTH)))

        # 32x18x96 -> 16x9x96
        self.pool_red = nn.AvgPool2d(2)

    def forward(
        self,
        x: torch.Tensor,
        x_state: torch.Tensor,
        actor_keys: Iterable[Hashable]
    ) -> Tuple[torch.Tensor]:

        mod = self.modulation(x_state)
        mod_multiplier, mod_bias = torch.split(mod, self.INPUT_SIZE, dim=1)

        x = self.attention(x, mod_multiplier, mod_bias)
        x = self.conv_mem_cell(x, actor_keys)

        x_visual = self.pool_red(x)
        x_visual_enc = self.pool_enc(x)
        x_visual_enc = torch.flatten(x_visual_enc, start_dim=1)

        return x_visual, x_visual_enc


class MotorCentre(nn.Module):
    """
    Interpretes/decodes the state of intent to produce an action vector and
    a focus heatmap, i.e. probabilities of key presses or of the cursor being
    moved for a specific amount and a spatial distribution of probabilities
    that the focus should be at some point.
    """

    HEIGHT = SpatioTemporalCentre.HEIGHT // 2
    WIDTH = SpatioTemporalCentre.WIDTH // 2
    FEAT_SIZE = SpatioTemporalCentre.HIDDEN_SIZE
    ACTION_BASE_SIZE = 128
    ACTION_SIZE = 72

    KERNEL_SIZE_1 = 3
    KERNEL_SIZE_2 = 5
    KERNEL_SIZE_3 = 7
    KERNEL_UNROLL_1 = KERNEL_SIZE_1**2
    KERNEL_UNROLL_2 = KERNEL_SIZE_2**2
    KERNEL_UNROLL_3 = KERNEL_SIZE_3**2

    RES_SPLIT = (FEAT_SIZE, HEIGHT*WIDTH, ACTION_BASE_SIZE)
    KERNEL_SPLIT = (KERNEL_UNROLL_1, KERNEL_UNROLL_2, KERNEL_UNROLL_3)

    def __init__(self, input_state_size: int):
        super().__init__()

        self.symlog = symlog
        self.silu = nn.SiLU()
        self.softmax = nn.Softmax(dim=1)

        # 384 -> 368 (96 + 16x9 + 128)
        self.base_fc = nn.Linear(input_state_size, self.FEAT_SIZE + self.HEIGHT*self.WIDTH + self.ACTION_BASE_SIZE)

        # 240 (16x9 + 96) -> 83 (3x3 + 5x5 + 7x7)
        self.kernel_fc = nn.Linear(
            self.HEIGHT*self.WIDTH + self.ACTION_BASE_SIZE,
            self.KERNEL_UNROLL_1 + self.KERNEL_UNROLL_2 + self.KERNEL_UNROLL_3)

        # 512 (384 + 128) -> 72
        self.action_fc = nn.Linear(input_state_size + self.ACTION_BASE_SIZE, self.ACTION_SIZE)

        nn.init.constant_(self.base_fc.bias, 0.)
        nn.init.constant_(self.kernel_fc.bias, 0.)
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

        x = x.view(1, b, h, w)

        # Need to use `F.pad` for modes besides `zeros`
        x = F.pad(x, (kernel_size // 2,) * 4, mode='replicate')

        x = F.conv2d(x, kernels, groups=b)
        x = x.reshape(b, 1, h, w)

        return x

    def forward(self, x_visual: torch.Tensor, x_state: torch.Tensor) -> Tuple[torch.Tensor]:
        focus_res = self.base_fc(x_state)
        base_query, base_focus, fine_action = torch.split(focus_res, self.RES_SPLIT, dim=1)

        base_focus = self.symlog(base_focus)
        fine_action = self.silu(fine_action)

        # Mouse/keyboard actions
        x_action = self.action_fc(torch.cat((x_state, fine_action), dim=1))

        # Get base focus
        b, c, h, w = x_visual.shape

        base_query = base_query.reshape(b, self.FEAT_SIZE, 1, 1)
        base_focus = base_focus.reshape(1, b, self.HEIGHT, self.WIDTH)

        x_focus = x_visual.view(1, b*c, h, w)
        x_focus = F.conv2d(x_focus, base_query, groups=b)
        x_focus = self.symlog(x_focus)
        x_focus = x_focus + base_focus

        # Get refinement kernels
        x_base = x_focus[0].reshape(b, h*w)
        x_base = self.softmax(x_base)

        focus_res = self.kernel_fc(torch.cat((x_base, fine_action), dim=1))
        kernel_weights_1, kernel_weights_2, kernel_weights_3 = torch.split(focus_res, self.KERNEL_SPLIT, dim=1)

        kernel_weights_1 = self.softmax(kernel_weights_1)
        kernel_weights_2 = self.softmax(kernel_weights_2)
        kernel_weights_3 = self.softmax(kernel_weights_3)

        # Refine focus in a series of upscalings and adaptive convolutions
        x_focus = x_focus.reshape(b, 1, h, w)

        x_focus = self.refine(x_focus, kernel_weights_1, self.KERNEL_SIZE_1, self.interp_size_1)
        x_focus = self.refine(x_focus, kernel_weights_2, self.KERNEL_SIZE_2, self.interp_size_2)
        x_focus = self.refine(x_focus, kernel_weights_3, self.KERNEL_SIZE_3, self.interp_size_3)

        return x_focus, x_action


class PCNet(nn.Module):
    """
    Peripheral-compatible recurrent neural network.

    The name is intended to allude to the personal computer, but otherwise
    refers to its interface with audio-visual inputs and mouse/keyboard outputs.

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

    N_WORKERS = N_DELAY = 6
    INPUT_MEM_SIZE = 384
    INTENT_MEM_SIZE = 384
    MAX_FOCAL_OFFSET = float(max(PrimaryVisualEncoding.HEIGHT, PrimaryVisualEncoding.WIDTH))
    MKBD_ENC_SIZE = 22

    def __init__(self, manager: Manager = None, critic: bool = False):
        super().__init__()

        encoded_state_size = sum((
            PrimaryVisualEncoding.ENC_SIZE, FocusedVisualEncoding.ENC_SIZE, AudioEncoding.ENC_SIZE, self.MKBD_ENC_SIZE))
        attended_state_size = SpatioTemporalCentre.ENC_SIZE + self.INPUT_MEM_SIZE

        # Cognitive centres
        self.primary_visual_centre = PrimaryVisualEncoding()
        self.focused_visual_centre = FocusedVisualEncoding()
        self.audio_centre = AudioEncoding(manager=manager)

        self.temporal_centre = IRLinearCell(encoded_state_size, self.INPUT_MEM_SIZE, manager=manager)
        self.spatio_temporal_centre = SpatioTemporalCentre(self.INPUT_MEM_SIZE, manager=manager)
        self.intent_centre = IRLinearCell(attended_state_size, self.INTENT_MEM_SIZE, manager=manager)
        self.motor_decoding = MotorCentre(self.INTENT_MEM_SIZE)

        # Mouse/key input adjustments
        self.mkbd_scale = nn.Parameter(torch.ones(1, self.MKBD_ENC_SIZE))
        self.mkbd_bias = nn.Parameter(torch.zeros(1, self.MKBD_ENC_SIZE))

        # Advantage estimator
        if critic:
            self.critic = nn.Linear(self.INTENT_MEM_SIZE, 1)

        else:
            self.critic = None

        # Locks for multi-threaded inference
        self.lock_1 = ChainLock(self.N_WORKERS, manager=manager)
        self.lock_2 = ChainLock(self.N_WORKERS, manager=manager)
        self.lock_3 = ChainLock(self.N_WORKERS, manager=manager)
        self.lock_4 = ChainLock(self.N_WORKERS, manager=manager)
        self.lock_5 = ChainLock(self.N_WORKERS, manager=manager)
        self.lock_6 = ChainLock(self.N_WORKERS, manager=manager)

    def get_n_params(self, trainable: bool = True) -> int:
        """Get the number of (trainable) parameters in the instantiated model."""

        return get_n_params(self, trainable=trainable)

    def clear(self, keys: Iterable[Hashable] = None):
        """Clear final (first) hidden/cell states of LSTM cells for TBPTT."""

        self.audio_centre.conv_mem_cell.clear(keys=keys)
        self.temporal_centre.clear(keys=keys)
        self.spatio_temporal_centre.conv_mem_cell.clear(keys=keys)
        self.intent_centre.clear(keys=keys)

    def detach(self, keys: Iterable[Hashable] = None):
        """Detach final (first) hidden/cell states of LSTM cells for TBPTT."""

        self.audio_centre.conv_mem_cell.detach(keys=keys)
        self.temporal_centre.detach(keys=keys)
        self.spatio_temporal_centre.conv_mem_cell.detach(keys)
        self.intent_centre.detach(keys=keys)

    def move(self, device: Union[str, torch.device]) -> 'PCNet':
        """Move model parameters and initial/default states to new device."""

        self.audio_centre.conv_mem_cell.move(device)
        self.temporal_centre.move(device)
        self.spatio_temporal_centre.conv_mem_cell.move(device)
        self.intent_centre.move(device)

        return self.to(device=device)

    def load(self, path: str, device: Union[str, torch.device] = None, strict: bool = True) -> 'PCNet':
        """Load model parameters from a state dict at the specified path."""

        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=strict)

        if device is not None:
            return self.move(device)

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
        detach: bool = False
    ) -> Tuple[torch.Tensor]:

        # Primary visual encoding
        with self.lock_1:
            x_visual_deep, x_visual_enc = self.primary_visual_centre(x_visual)

        # Focused visual encoding
        with self.lock_2:
            x_focused_enc = self.focused_visual_centre(x_visual, x_visual_deep, focal_points)

        # Audio encoding and mkbd normalisation
        with self.lock_3:
            x_audio = self.audio_centre(x_audio, actor_keys)

            if detach:
                self.audio_centre.conv_mem_cell.detach(keys=actor_keys)

            x_mkbd = torch.hstack((x_mkbd, focal_points / self.MAX_FOCAL_OFFSET))
            x_mkbd = x_mkbd * self.mkbd_scale + self.mkbd_bias

        # Temporal contextualisation
        with self.lock_4:
            x_state = torch.hstack((x_visual_enc, x_focused_enc, x_audio, x_mkbd))
            x_state = self.temporal_centre(x_state, actor_keys)

            if detach:
                self.temporal_centre.detach(keys=actor_keys)

        # Spatio-temporal contextualisation
        with self.lock_5:
            x_visual, x_visual_enc = self.spatio_temporal_centre(x_visual_deep, x_state, actor_keys)

            if detach:
                self.spatio_temporal_centre.conv_mem_cell.detach(keys=actor_keys)

        # Action inference and motor decoding
        with self.lock_6:
            x_state = torch.hstack((x_visual_enc, x_state))
            x_state = self.intent_centre(x_state, actor_keys)

            if detach:
                self.intent_centre.detach(keys=actor_keys)

            x_focus, x_action = self.motor_decoding(x_visual, x_state)

        # States can be used for a critic/advantage estimator head in RL, but not needed for inference
        if self.critic is not None:
            state_values = self.critic(x_state)
            return x_focus, x_action, state_values

        return x_focus, x_action
