"""Utilities for PCNet, SDG AI actors, and training processes"""

from collections import deque
from typing import Deque, Dict, Hashable, Iterable, List, Tuple, Union

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F


DIVLOGPI = 1./np.log(np.pi)


class StateStore:
    """
    Store for sequential states of dynamic combinations of multiple actors
    (intended for distributed inference and EBPTT).
    """

    def __init__(self, dim: Union[int, Tuple[int]]):
        self.batches: List[torch.Tensor] = []
        self.states: Dict[Hashable, torch.Tensor] = {}
        self.default = torch.zeros(dim)

    def move(self, device: Union[str, torch.device]):
        """
        Move states to new device. This will start a new computational graph,
        so current batches are cleared as well.
        """

        self.default = self.default.to(device=device)

        for key in self.states.keys():
            self.states[key] = self.states[key].to(device=device)

        self.batches.clear()

    def clear(self, keys: Iterable[Hashable] = None):
        """
        Clear states for a set of actors, in effect reverting them to the
        initial/default state on subsequent gathering.
        """

        for key in tuple(self.states.keys()) if keys is None else keys:
            del self.states[key]

    def get(self, keys: Iterable[Hashable] = None) -> torch.Tensor:
        """
        Get the last batch of states for a set of actors or gather them into
        an initial batch that is added to current buffer.
        """

        if not self.batches:
            self.batches.append(
                torch.stack(
                    [self.states.get(key, self.default) for key in (self.states.keys() if keys is None else keys)]))

        return self.batches[-1]

    def append(self, states: torch.Tensor):
        """Add new batch of states to current buffer."""

        self.batches.append(states)

    def detach(self, keys: Iterable[Hashable]):
        """
        Collectively detach the states from the computational graph at the final
        batch and clear the explicit buffer.

        States to be preserved for the next iteration are extracted as views
        of the final batch. Dividing batches into views and then stacking them
        according to given keys allows them to be updated separately for
        multiple actors. Once the updates have replaced all of the views of
        a specific batch, its resources should be released, as it is no longer
        referenced.

        NOTE: This was initially formulated as an attempt to break the
        computational graph at an arbitrary point, but certain errors
        stemming from the rigidity of the graph could not be resolved,
        so it only worked for the final batch.

        See:
        - https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500
        - https://discuss.pytorch.org/t/quick-detach-question/1090
        - https://discuss.pytorch.org/t/stop-backward-at-some-intermediate-tensor/74948
        - https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
        """

        if not self.batches:
            return

        # Break computational graph at final batch
        final_batch = self.batches[-1].detach()

        self.batches.clear()

        # Update states with views of the final batch
        for key, state in zip(keys, final_batch):
            self.states[key] = state


def get_n_params(model: nn.Module, trainable: bool = True) -> int:
    """Get the number of (trainable) parameters in a given model."""

    return sum(param.numel() for param in model.parameters() if (not trainable or param.requires_grad))


def init_std(layer: Union[nn.Linear, nn.Conv2d, nn.Conv1d], n_blocks: int = None, n_layers: int = 2) -> float:
    """Get the standard deviation corresponding to He or Fixup initialisation."""

    # He init
    if isinstance(layer, nn.Linear):
        std = np.sqrt(2. / layer.out_features)

    else:
        n_params = np.prod(layer.kernel_size) * layer.out_channels
        std = np.sqrt(2. / n_params)

    # Fixup init
    if n_blocks is not None:
        std *= n_blocks ** (-1 / (2 * n_layers - 2))

    return std


def spatial_softmax(x: torch.Tensor) -> torch.Tensor:
    """Compute the softmax across spatial dimensions."""

    b, c, h, w = x.shape

    x = x.view(b, c, h*w)
    x = F.softmax(x, dim=-1)
    x = x.view(b, c, h, w)

    return x


def spatial_log_softmax(x: torch.Tensor) -> torch.Tensor:
    """Compute the log softmax across spatial dimensions."""

    b, c, h, w = x.shape

    x = x.view(b, c, h*w)
    x = F.log_softmax(x, dim=-1)
    x = x.view(b, c, h, w)

    return x


def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Symmetric logarithmic unit.
    An attempt of saturated yet unbounded activation.
    Divided by `log(pi)` to preserve derivative of `1` at `x == 0`.
    """

    return torch.log(torch.pow(x, 2) + np.pi) * torch.tanh(x) * DIVLOGPI


def silog(x: torch.Tensor) -> torch.Tensor:
    """
    Sigmoid weighted logarithmic unit.
    An attempt of saturated yet (positively) unbounded activation.
    """

    return symlog(x) * torch.sigmoid(x)


def batch_crop(
    x: torch.Tensor,
    centre_indices: torch.Tensor,
    length: int = 9,
    padding: int = 0,
    naive: bool = False,
    batch_indices: torch.Tensor = None,
    length_indices: torch.Tensor = None
) -> torch.Tensor:
    """
    Crop images in a batch, each at its own crop indices, by array indexing
    (should scale better than naive loop + concatenate implementation).

    Expects odd crop lengths and images in BCHW format.

    NOTE: Padding makes things considerably slower.
    NOTE: Cropping by indexing may cause unexpected form in GPU memory.
    """

    if padding:
        x = F.pad(x, (padding,)*4)

    b, _, h, w = x.shape
    half_length = length//2

    h_indices = torch.clip(centre_indices[:, 0] + (padding-half_length), 0, h-length)
    w_indices = torch.clip(centre_indices[:, 1] + (padding-half_length), 0, w-length)

    if naive:
        return torch.cat(
            [
                x[b_idx:b_idx+1, :, h_idx:h_idx+length, w_idx:w_idx+length]
                for b_idx, (h_idx, w_idx) in enumerate(zip(h_indices, w_indices))])

    # These can be cached externally
    if batch_indices is None:
        batch_indices = torch.arange(b, dtype=torch.long, device=centre_indices.device)[:, None, None]

    if length_indices is None:
        length_indices = torch.arange(length, dtype=torch.long, device=centre_indices.device)

    h_indices = (length_indices + h_indices[:, None])[..., None]
    w_indices = (length_indices + w_indices[:, None])[:, None]

    return x[batch_indices, :, h_indices, w_indices].permute(0, 3, 1, 2)


def prepare_inputs(
    frame: np.ndarray,
    spectral_vectors: np.ndarray,
    mkbd_state: List[Union[int, float]],
    focal_point: Tuple[int, int],
    eps: float = 1e-12,
    device: torch.device = None
) -> Tuple[torch.Tensor]:
    """Adjust the range of inputs and convert them to tensors for model inference."""

    # BGR -> RGB and to [0., 1.] range
    frame = frame[..., ::-1] / 255.
    spectral_vectors = spectral_vectors / (-10.*np.log10(eps)) + 1.

    # Move channels to first axis, add batch axis, and convert to tensors on target device
    x_visual = torch.as_tensor(np.moveaxis(frame, 2, 0)[None], dtype=torch.float, device=device)
    x_audio = torch.as_tensor(spectral_vectors[None], dtype=torch.float, device=device)
    x_mkbd = torch.as_tensor(mkbd_state, dtype=torch.float, device=device)[None]
    x_focus = torch.as_tensor(focal_point, dtype=torch.long, device=device)[None]

    return x_visual, x_audio, x_mkbd, x_focus


def supervised_loss(
    x_focus: torch.Tensor,
    x_action: torch.Tensor,
    demo_focus: torch.Tensor,
    demo_action: torch.Tensor,
    fl_gamma: float = 2,
    fl_alpha: float = 0.95,
    keys: List[str] = None
) -> Tuple[torch.Tensor, Dict[str, Dict[str, float]]]:
    """
    Supervised loss function based on cross entropy (related to MLE and KL div)
    with focal (loss) adjustment (not to be confused with the focal term)
    and some 'class' imbalance compensation.

    Different terms are summed together and then averaged across the batch.

    NOTE: Due to unreliable pseudo-labels for focal coordinates, the focal term
    is treated as an abundant class to be assigned a smaller weight. Similarly,
    mouse movements include a lot of noise (meaningless or inconsistent moves)
    and have adjusted weight.

    NOTE: Regularisation is omitted (handled by weight decay in the optimiser).
    """

    kbd_term = F.binary_cross_entropy_with_logits(x_action[:, :19], demo_action[:, :19], reduction='none')

    mvmt_y_term = torch.sum(-F.log_softmax(x_action[:, 19:44], dim=1) * demo_action[:, 19:44], dim=1, keepdim=True)
    mvmt_x_term = torch.sum(-F.log_softmax(x_action[:, 44:69], dim=1) * demo_action[:, 44:69], dim=1, keepdim=True)
    mwhl_y_term = torch.sum(-F.log_softmax(x_action[:, 69:72], dim=1) * demo_action[:, 69:72], dim=1, keepdim=True)

    # To log of spatial distribution
    x_focus = -spatial_log_softmax(x_focus)

    # Probabilities at demo focal indices are implicitly 1. and otherwise 0.
    focal_term = torch.cat([x_focus[i:i+1, 0, y, x] for i, (y, x) in enumerate(demo_focus)])
    focal_term = focal_term[:, None]

    terms_per_sample = torch.cat((kbd_term, mvmt_y_term, mvmt_x_term, mwhl_y_term, focal_term), dim=1)

    # Weights to prevent dominance of many null/still examples
    focal_loss_weights = torch.pow(1. - torch.exp(-terms_per_sample), fl_gamma)

    if fl_alpha == 0.5:
        loss_per_sample = torch.sum(fl_alpha * focal_loss_weights * terms_per_sample, dim=1)

    else:
        alpha_flags = torch.cat(
            (
                demo_action[:, :19],
                (1. - demo_action[:, 31:32]) / 3.,
                (1. - demo_action[:, 56:57]) / 3.,
                1. - demo_action[:, 70:71],
                torch.zeros_like(focal_term)), dim=1)

        focal_loss_alphas = alpha_flags * fl_alpha + (1. - alpha_flags) * (1. - fl_alpha)

        loss_per_sample = torch.sum(focal_loss_alphas * focal_loss_weights * terms_per_sample, dim=1)

    if keys is None:
        return torch.mean(loss_per_sample), {}

    # Loss terms per key
    kbd_scalars = {}
    mvmt_y_scalars = {}
    mvmt_x_scalars = {}
    mwhl_y_scalars = {}
    focal_scalars = {}
    alpha_scalars = {}
    loss_scalars = {}

    with torch.no_grad():
        for idx, key in enumerate(keys):
            kbd_scalars[key] = kbd_term[idx].sum().item()
            mvmt_y_scalars[key] = mvmt_y_term[idx].item()
            mvmt_x_scalars[key] = mvmt_x_term[idx].item()
            mwhl_y_scalars[key] = mwhl_y_term[idx].item()
            focal_scalars[key] = focal_term[idx].item()
            alpha_scalars[key] = alpha_flags[idx].sum().item()
            loss_scalars[key] = loss_per_sample[idx].item()

    scalars = {
        'kbd': kbd_scalars,
        'mmot_y': mvmt_y_scalars,
        'mmot_x': mvmt_x_scalars,
        'mwhl_y': mwhl_y_scalars,
        'focus': focal_scalars,
        'num_alpha': alpha_scalars,
        'loss_per_key': loss_scalars}

    return torch.mean(loss_per_sample), scalars


class SequenceIterator:
    """
    Abstracts iteration over a collection of sequence arrays.

    Iteration step is strict and determined by `slice_length`. Iteration ends
    when reaching the final slice of a set of sequences, or, if length is
    exponentiated on reset, the final slice of a set of current sub-sequences.

    If `rng` is given, iterations will start from a small random offset
    (up to `slice_length` or at sub-sequence starting index). This way,
    if the loss, gradients, and/or updates are computed for samples
    at a fixed interval, all samples can eventually be in that place.
    """

    def __init__(
        self,
        sequences: Tuple[Union[np.ndarray, h5py.Dataset]],
        slice_length: int = 1,
        exp_length_on_reset: float = None,
        rng: np.random.Generator = None,
        key: Hashable = None
    ):
        self.sequences = sequences
        self.slice_length = slice_length
        self.rng: Union[np.random.Generator, None] = rng
        self.key = key

        self.slice_idx = 0
        self.slice_offset = 0
        self.total_length = min(len(seq) for seq in sequences)
        self.max_slices = self.total_length // slice_length

        assert self.max_slices > 0, 'Not enough samples to construct a single slice.'

        self.exp_length_ctr = 0
        self.exp_length_on_reset = exp_length_on_reset

    def reset(self):
        """Set new random offset and restart the slice counter."""

        max_length = self.total_length - self.slice_length

        if self.exp_length_on_reset is not None and self.exp_length_on_reset:
            exp_length = int(self.slice_length * self.exp_length_on_reset ** self.exp_length_ctr)

            if exp_length < max_length:
                max_length = exp_length
                self.exp_length_ctr += 1

            else:
                self.exp_length_on_reset = None

        self.slice_idx = 0
        self.slice_offset: int = 0 if self.rng is None else self.rng.integers(0, self.total_length - max_length)
        virtual_length = min(max_length, min(len(seq) - self.slice_offset for seq in self.sequences))
        self.max_slices = virtual_length // self.slice_length

        assert self.max_slices > 0, 'Not enough samples to construct a single slice.'

    def __len__(self) -> int:
        return self.max_slices - self.slice_idx

    def __iter__(self) -> 'SequenceIterator':
        self.reset()
        return self

    def __next__(self) -> Iterable[np.ndarray]:
        if not self.__len__():
            raise StopIteration

        idx = self.slice_idx * self.slice_length + self.slice_offset
        self.slice_idx += 1

        return (seq[idx:idx+self.slice_length] for seq in self.sequences)


class Dataset:
    """
    A custom data store and iterator to keep iteration over batches strictly
    ordered, i.e. single threaded.

    To keep the number of sequences in batches consistent, the sequences can be
    set to repeat: when a sequence runs out, it is restarted anew (and can then
    overlap with other sequences in a different way). Otherwise, iteration ends
    with the shortest sequence (lest batches change size).
    """

    def __init__(
        self,
        files: Iterable[h5py.File],
        slice_length: int = 30,
        max_steps_with_repeat: int = 0,
        max_batch_size: int = None,
        resume_step: int = 0,
        exp_length_on_reset: float = None,
        seed: int = None,
        device: torch.device = None
    ):
        self.slice_length = slice_length
        self.max_steps_with_repeat = max_steps_with_repeat
        self.rng: np.random.Generator = np.random.default_rng(seed=seed)
        self.device = device
        self.resume_step = resume_step

        self.steps: int = None
        self.reset_keys: Deque[Hashable] = deque()
        self.sequences: List[SequenceIterator] = []

        for file in files:
            self.sequences.append(
                SequenceIterator(
                    (
                        file['image'],
                        file['spectrum'],
                        file['mkbd'],
                        file['focus'],
                        file['action'],
                        file.attrs['key'].repeat(file.attrs['len'])[:, None]),
                    slice_length=self.slice_length,
                    exp_length_on_reset=exp_length_on_reset,
                    rng=self.rng,
                    key=file.attrs['key']))

        self.batch_size = len(self.sequences) if max_batch_size is None else min(max_batch_size, len(self.sequences))
        self.seq_indices = np.arange(len(self.sequences))

    def reset(self):
        """Set new random offset and restart the slice counter for each sequence."""

        for seq in self.sequences:
            seq.reset()

        self.steps = 0
        self.resume_next()

    def __len__(self) -> int:
        return min(len(seq) for seq in self.sequences)

    def __iter__(self) -> 'Dataset':
        self.reset()
        return self

    def get_batch(
        self,
        concurrent_slices: Iterable[Tuple[np.ndarray]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor, np.ndarray], torch.Tensor]:
        """
        Batch sequence slices by data field, forming a collection of inputs
        and an output.
        """

        images, spectra, mkbds, foci, actions, keys = zip(*concurrent_slices)

        return (
            (
                torch.as_tensor(np.concatenate(images, axis=1), device=self.device),
                torch.as_tensor(np.concatenate(spectra, axis=1), device=self.device),
                torch.as_tensor(np.concatenate(mkbds, axis=1), device=self.device),
                torch.as_tensor(np.concatenate(foci, axis=1), dtype=torch.long, device=self.device),
                np.concatenate(keys, axis=1)),
            torch.as_tensor(np.concatenate(actions, axis=1), device=self.device))

    def __next__(self):
        if self.max_steps_with_repeat and self.steps >= self.max_steps_with_repeat:
            raise StopIteration

        elif all(self.sequences):
            self.steps += 1

            return self.get_batch(
                next(self.sequences[seq_idx]) for seq_idx in self.rng.choice(
                    self.seq_indices, size=self.batch_size, replace=False, shuffle=False))

        elif self.max_steps_with_repeat:
            for seq in self.sequences:
                if not seq:
                    seq.reset()

                    # Allow states of reset sequences to be cleared externally
                    self.reset_keys.append(seq.key)

            self.steps += 1

            return self.get_batch(
                next(self.sequences[seq_idx]) for seq_idx in self.rng.choice(
                    self.seq_indices, size=self.batch_size, replace=False, shuffle=False))

        else:
            raise StopIteration

    def resume_next(self):
        """
        Advance iteration until reaching the step to resume.
        Calls to the random generator are replayed as well.
        """

        while self.steps < self.resume_step:
            if self.max_steps_with_repeat and self.steps >= self.max_steps_with_repeat:
                break

            elif all(self.sequences):
                self.steps += 1

                _ = [
                    next(self.sequences[seq_idx]) for seq_idx in self.rng.choice(
                        self.seq_indices, size=self.batch_size, replace=False, shuffle=False)]

            elif self.max_steps_with_repeat:
                for seq in self.sequences:
                    if not seq:
                        seq.reset()

                self.steps += 1

                _ = [
                    next(self.sequences[seq_idx]) for seq_idx in self.rng.choice(
                        self.seq_indices, size=self.batch_size, replace=False, shuffle=False)]

            else:
                break
