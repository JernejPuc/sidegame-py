"""Utilities for PCNet, SDG AI actors, and training processes"""

from collections import deque
from typing import Deque, Hashable, Iterable, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Lock


class ChainLock:
    """
    A lock with a buffer to ensure execution of waiting threads in FIFO order.
    This is done by making the entrant thread focus only on the lock of its
    predecessor, instead of a lock that is universally shared.

    NOTE: By specifying `n_workers` and drawing from a pre-initialised pool of
    locks, the chain can be viewed as 'finite', consisting of the same links
    (i.e. locks) that repeat periodically. If they were initialised on the fly,
    each link would be new and unique, and the chain 'infinite'.
    """

    def __init__(self, n_workers: int):
        self._free_locks = deque(Lock() for _ in range(n_workers))
        self._active_locks = deque()

    def __enter__(self):
        own_lock = self._free_locks.popleft()
        own_lock.acquire()

        self._active_locks.append(own_lock)

        # Wait for predecessor to finish
        if len(self._active_locks) > 1:
            with self._active_locks[-2]:
                pass

    def __exit__(self, *_exc_args):
        own_lock = self._active_locks.popleft()
        own_lock.release()

        self._free_locks.append(own_lock)


def init_std(layer: Union[nn.Conv2d, nn.Conv1d], n_blocks: int = None, n_layers: int = 2) -> float:
    """Get the standard deviation corresponding to He or Fixup initialisation."""

    # He init
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
    device: Union[torch.device, str] = None
) -> Tuple[torch.Tensor]:
    """Adjust the range of inputs and convert them to tensors for model inference."""

    # BGR -> RGB and to [0., 1.] range
    frame = frame[..., ::-1] / 255.
    spectral_vectors = spectral_vectors / (-10.*np.log10(eps)) + 1.

    # Move channels to first axis, add batch axis, and convert to tensors on target device
    x_visual = torch.Tensor(np.moveaxis(frame, 2, 0)[None], device=device)
    x_audio = torch.Tensor(spectral_vectors[None], device=device)
    x_mkbd = torch.Tensor(mkbd_state, device=device)[None]
    x_focus = torch.LongTensor(focal_point, device=device)[None]

    return x_visual, x_audio, x_mkbd, x_focus


def get_n_params(model: nn.Module, trainable: bool = True) -> int:
    """Get the number of (trainable) parameters in a given model."""

    return sum(param.numel() for param in model.parameters() if (not trainable or param.requires_grad))


def supervised_loss(
    x_focus: torch.Tensor,
    x_action: torch.Tensor,
    demo_focus: torch.Tensor,
    demo_action: torch.Tensor,
    focal_weight: float = 0.05
) -> torch.Tensor:
    """
    Supervised loss function based on cross entropy (related to MLE and KL div),
    averaged over the batch and constituent terms.

    Regularisation is omitted (handled by weight decay in the optimiser).
    """

    kbd_term = F.binary_cross_entropy_with_logits(x_action[:, :19], demo_action[:, :19], reduction='none')

    mvmt_y_term = torch.sum(-F.log_softmax(x_action[:, 19:44], dim=1) * demo_action[:, 19:44], dim=1, keepdim=True)
    mvmt_x_term = torch.sum(-F.log_softmax(x_action[:, 44:69], dim=1) * demo_action[:, 44:69], dim=1, keepdim=True)
    mwhl_y_term = torch.sum(-F.log_softmax(x_action[:, 69:72], dim=1) * demo_action[:, 69:72], dim=1, keepdim=True)

    action_term = torch.mean(torch.cat((kbd_term, mvmt_y_term, mvmt_x_term, mwhl_y_term), dim=1))

    # To log of spatial distribution
    x_focus = -spatial_log_softmax(x_focus)

    # Probabilities at demo focal indices are implicitly 1. and otherwise 0.
    focal_term = torch.mean(torch.cat([x_focus[i:i+1, 0, y, x] for i, (y, x) in enumerate(demo_focus)]))

    return action_term + focal_weight * focal_term


class SequenceIterator:
    """
    Abstracts iteration over a collection of sequence arrays.

    Iteration step is strict and determined by `slice_length`.

    If `rng` is given, any iteration will start from a small random offset
    (up to `slice_length`) so that, if backprop is only performed on final
    samples in a temporal slice, all samples can eventually be in that place.
    """

    def __init__(self, sequences: Tuple[np.ndarray], slice_length: int = 1, rng: np.random.Generator = None):
        self.sequences = sequences
        self.slice_length = slice_length
        self.rng: Union[np.random.Generator, None] = rng

        self.slice_idx = 0
        self.slice_offset = 0
        self.min_samples = min(len(seq) for seq in sequences)
        self.max_slices = self.min_samples // slice_length

    def reset(self):
        """Set new random offset and restart the slice counter."""

        self.slice_idx = 0
        self.slice_offset: int = 0 if self.rng is None else self.rng.integers(0, self.slice_length)
        self.min_samples = min(len(seq) - self.slice_offset for seq in self.sequences)
        self.max_slices = self.min_samples // self.slice_length

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

    IDX_KEY = -1

    def __init__(
        self,
        npzs: Iterable[np.lib.npyio.NpzFile],
        truncated_length: int = 16,
        max_steps_with_repeat: int = 0,
        max_focal_offset: float = 10.,
        max_batch_size: int = None,
        seed: int = None,
        device: str = None
    ):
        self.truncated_length = truncated_length
        self.max_steps_with_repeat = max_steps_with_repeat
        self.max_focal_offset = max_focal_offset
        self.rng: np.random.Generator = np.random.default_rng(seed=seed)
        self.device = device

        self.steps: int = None
        self.reset_keys: Deque[Hashable] = deque()
        self.reset_indices: Deque[int] = deque()

        self.sequences: List[SequenceIterator] = []

        for npz in npzs:
            self.sequences.append(
                SequenceIterator(
                    (
                        np.moveaxis(npz['image'], 3, 1)[:, None],
                        npz['spectrum'][:, None],
                        npz['mkbd'][:, None],
                        npz['cursor'][:, None],
                        npz['action'][:, None],
                        npz['meta'][:, None].repeat(len(npz['cursor']), axis=0)),
                    slice_length=self.truncated_length,
                    rng=self.rng))

            del npz

        self.batch_size = len(self.sequences) if max_batch_size is None else min(max_batch_size, len(self.sequences))
        self.seq_indices = np.arange(len(self.sequences))

    def reset(self):
        """Set new random offset and restart the slice counter for each sequence."""

        for seq in self.sequences:
            seq.reset()

        self.steps = 0

    def __len__(self) -> int:
        return min(len(seq) for seq in self.sequences)

    def __iter__(self) -> 'Dataset':
        self.reset()
        return self

    def get_focus_from_cursor(self, cursor: np.ndarray) -> np.ndarray:
        """
        Get indices of focal points by adding small random offsets to
        cursor coordinates.
        """

        focus = (cursor + self.rng.uniform(-self.max_focal_offset, self.max_focal_offset, cursor.shape)) / 2.
        focus[..., 0] = np.clip(focus[..., 0], 0., 71.)
        focus[..., 1] = np.clip(focus[..., 1], 0., 127.)

        return np.around(focus)

    def get_batch(
        self,
        concurrent_slices: Iterable[Tuple[np.ndarray]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor, np.ndarray], torch.Tensor]:
        """
        Batch sequence slices by data field, forming a collection of inputs
        and an output.
        """

        images, spectra, mkbds, cursors, actions, keys = zip(*concurrent_slices)

        return (
            (
                torch.Tensor(np.concatenate(images, axis=1), device=self.device),
                torch.Tensor(np.concatenate(spectra, axis=1), device=self.device),
                torch.Tensor(np.concatenate(mkbds, axis=1), device=self.device),
                torch.LongTensor(self.get_focus_from_cursor(np.concatenate(cursors, axis=1)), device=self.device),
                np.concatenate(keys, axis=1)),
            torch.Tensor(np.concatenate(actions, axis=1), device=self.device))

    def __next__(self):
        if self.max_steps_with_repeat and self.steps >= self.max_steps_with_repeat:
            raise StopIteration

        elif all(self.sequences):
            self.steps += 1

            return self.get_batch(
                next(self.sequences[seq_idx]) for seq_idx in self.rng.choice(
                    self.seq_indices, size=self.batch_size, replace=False, shuffle=False))

        elif self.max_steps_with_repeat:
            for idx, seq in enumerate(self.sequences):
                if not seq:
                    seq.reset()

                    # Allow states of reset sequences to be cleared externally
                    key_array = seq.sequences[self.IDX_KEY]
                    self.reset_keys.append(key_array[(0,)*len(key_array.shape)])
                    self.reset_indices.append(idx)

            self.steps += 1

            return self.get_batch(
                next(self.sequences[seq_idx]) for seq_idx in self.rng.choice(
                    self.seq_indices, size=self.batch_size, replace=False, shuffle=False))

        else:
            raise StopIteration
