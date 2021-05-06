"""Planar audio system and supporting objects."""

import os
import wave
import threading
from typing import Deque, Dict, List, Optional, Tuple, Union
from collections import deque
from time import sleep

import pyaudio
import numpy as np
from scipy.signal import lfilter, butter

from sidegame.physics import OrientedEntity


class OrientedSound:
    """
    A wrapper around a sound array that allows it to be changed wrt.
    relative distance and angle between source and listener entities.
    """

    def __init__(self, sound: np.ndarray, listener: OrientedEntity, source: OrientedEntity):
        self.sound = sound
        self.listener = listener
        self.source = source

    def get_distance(self) -> float:
        """Get the relative distance between the listener and the source."""

        return np.linalg.norm(self.listener.pos - self.source.pos)

    def get_angle(self) -> Union[float, None]:
        """
        Get the angle of the source relative to the listener.
        If the listener is itself the source, return `None` instead.
        """

        if self.source is self.listener:
            return None

        relative_x, relative_y = self.source.pos - self.listener.pos
        source_angle = np.arctan2(relative_y, relative_x)

        # Negated, because HRIR angle keys are not adapted for y-axis inversion
        # (see the docstring for `OrientedEntity`)
        relative_angle = -(source_angle - self.listener.angle)

        return relative_angle


def deinterleave(sound: np.ndarray, mono: bool = False) -> np.ndarray:
    """
    Reshape a stereo sound array from a singular vector into two
    explicit channels. Mono sounds are handled by copying.
    """

    return np.vstack((sound, sound)) if mono else sound.reshape(2, -1, order='F')


def interleave(sound: np.ndarray) -> np.ndarray:
    """
    Reshape a stereo sound array from two explicit channels into a singular
    vector.
    """

    return sound.ravel(order='F')


def load_sound(filename: str) -> np.ndarray:
    """Read a sound file and load it as an uninterleaved stereo array."""

    with wave.open(filename, 'rb') as sound_file:
        mono = sound_file.getnchannels() == 1
        sound = np.frombuffer(sound_file.readframes(sound_file.getnframes()), dtype=np.int16).astype(np.float32)

    return deinterleave(sound, mono=mono)


def get_distance_filters(
    max_distance: float = 100.,
    distance_scaling: float = 1.,
    sampling_rate: Union[int, float] = 44100.
) -> Dict[int, Tuple[np.ndarray]]:
    """
    Prepare (low-pass and scaling) filter components (numerator and denominator)
    for distance attenuation at discrete (integer-resolution) intervals.

    The filters are built around 1-metre units, but these can be converted
    through `distance_scaling`.
    """

    distance_filters = {0: (np.array(1., ndmin=1), np.array(1., ndmin=1))}

    for distance in range(1, int(distance_scaling*max_distance)+1):
        num, den = butter(
            3,
            sampling_rate/2. * np.exp(-0.006875*(distance/distance_scaling)**1.375),
            fs=sampling_rate)

        num *= np.exp(-0.1375*(distance/distance_scaling)**0.75)

        distance_filters[distance] = num, den

    return distance_filters


def get_hrir_filters() -> Dict[int, np.ndarray]:
    """
    Load HRIR (head-related impulse response) data (arrays to be convolved)
    for positional audio filtering.

    The filters cover a full circle of relative angles (in 2-degree steps)
    on a single plane, i.e. around a single axis.
    """

    data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'hrir2d.npy'))

    hrir_filters = {deg: datum for deg, datum in zip(range(0, 360, 360 // len(data)), data)}

    return hrir_filters


class PlanarAudioSystem:
    """
    An audio system using HRIR, low-pass, and scaling filters to simulate
    dynamic positioning of sound sources on a 2D plane.

    It can be used either by starting a streaming thread in the background
    or stepping it manually, extracting processed sound from an externally
    accessible buffer.

    Some parameters, e.g. sampling rate of 44.1kHz and format of (s)int16,
    are hard-coded to conform to the HRIR data and game sounds.

    For similar reasons, padding width (and other constants related to
    HRIR array length of 256) that are required for per-chunk convolution,
    have not been made general either.

    NOTE: Should be played over headphones.
    """

    SAMPLING_RATE = 44100
    _N_CHANNELS = 2
    _SAMPLE_WIDTH = 2
    _MIN_SOUND_VAL = -2.**(_SAMPLE_WIDTH*8-1)
    _MAX_SOUND_VAL = 2.**(_SAMPLE_WIDTH*8-1) - 1.

    _DEG_TO_RAD_FACTOR = 180./np.pi

    def __init__(
        self,
        step_freq: Union[int, float] = 30,
        max_distance: float = 100.,
        distance_scaling: float = 1.,
        load_attenuation: float = 0.25,
        volume: float = 1.,
        max_n_sounds: int = 25
    ):
        self._audio: pyaudio.PyAudio = None
        self._stream: pyaudio.Stream = None
        self._player: threading.Thread = None
        self.streaming = False

        self._audio_channels: List[Deque[OrientedSound]] = [deque() for _ in range(max_n_sounds)]
        self._audio_channels_io_lock = threading.Lock()

        self._chunk_size = int(self.SAMPLING_RATE // step_freq)
        self._max_distance = int(distance_scaling*max_distance)

        self._distance_filters = get_distance_filters(max_distance, distance_scaling, self.SAMPLING_RATE)
        self._hrir_filters = get_hrir_filters()

        self._load_attenuation = load_attenuation
        self.volume = volume

        self.external_buffer: Deque[np.ndarray] = deque()
        self.external_buffer_io_lock = threading.Lock()
        self._null_chunk: np.ndarray = np.zeros((self._N_CHANNELS, self._chunk_size + 510))

    def start(self):
        """Start a streaming thread in the background."""

        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(
            rate=self.SAMPLING_RATE,
            channels=self._N_CHANNELS,
            format=self._audio.get_format_from_width(self._SAMPLE_WIDTH),
            output=True)

        self._player = threading.Thread(target=self._play, daemon=True)
        self._player.start()

    def _play(self):
        """Automatically step the audio stream."""

        self.streaming = True

        while self.streaming:
            self.step()

    def stop(self):
        """Stop the streaming thread and clean up after it."""

        if self._player is None:
            return

        self.streaming = False

        self._player.join()
        self._player = None

        self._stream.stop_stream()
        self._stream.close()
        self._stream = None

        self._audio.terminate()
        self._audio = None

    def step(self):
        """
        Process an iteration (first available chunks) of queued sounds.

        The resulting sound is added to the audio stream, if available.
        Otherwise, it is added to an externally accessible buffer.
        """

        with self._audio_channels_io_lock:
            concurrent_sounds = [queued_sound.popleft() for queued_sound in self._audio_channels if queued_sound]

        if concurrent_sounds and self.volume != 0.:
            processed_sounds = self._process_sounds(concurrent_sounds)
            summed_sound = self._sum_sounds(processed_sounds) * self.volume

            if self.streaming:
                valid_sound = summed_sound[..., 255:-255]
                self._stream.write(interleave(valid_sound).astype(np.int16).tobytes())

            else:
                with self.external_buffer_io_lock:
                    self.external_buffer.append(summed_sound)

        elif self.streaming:
            sleep(1e-3)

        else:
            with self.external_buffer_io_lock:
                self.external_buffer.append(self._null_chunk)

    def load_sound(self, filename: str) -> List[np.ndarray]:
        """
        Read sound file and load the sound as a sequence of padded (overlapping)
        sound chunks.

        Padding of sound chunks serves two purposes: it allows per-chunk
        convolutions and is useful for windowing in spectral analysis.

        The sound is also attenuated to leave room for sound summation.
        """

        sound = load_sound(filename) * self._load_attenuation

        back_padding = (self._chunk_size - sound.shape[1] % self._chunk_size) % self._chunk_size + 127

        padded_sound = np.pad(sound, ((0, 0), (128, back_padding)))
        chunk_offset = 128
        chunks = deque()

        while (chunk_offset + self._chunk_size + 127) <= padded_sound.shape[1]:
            chunk = padded_sound[:, (chunk_offset - 128):(chunk_offset + self._chunk_size + 127)]
            chunks.append(chunk)

            chunk_offset += self._chunk_size

        return list(chunks)

    def queue_sound(self, sound: List[np.ndarray], listener: OrientedEntity, source: OrientedEntity):
        """Append a copy of the sound sequence to the first free audio channel."""

        with self._audio_channels_io_lock:
            for channel in self._audio_channels:
                if not channel:
                    channel.extend(OrientedSound(chunk, listener, source) for chunk in sound)
                    break

    def _sum_sounds(self, sounds: List[np.ndarray]) -> np.ndarray:
        """
        Add concurrent sounds together.
        For simplicity, sounds are expected to be of equal length.
        """

        if len(sounds) == 1:
            return sounds[0]

        return np.clip(np.add.reduce(sounds), self._MIN_SOUND_VAL, self._MAX_SOUND_VAL)

    def _process_sounds(self, sounds: List[OrientedSound]) -> List[np.ndarray]:
        """Apply positional effects to concurrent oriented sounds."""

        return [
            self._apply_positioning(
                self._apply_distance_attenuation(sound.sound, sound.get_distance()),
                sound.get_angle())
            for sound in sounds]

    def _apply_distance_attenuation(self, sound: np.ndarray, distance: float) -> np.ndarray:
        """Apply amplitude scaling and frequency attenuation."""

        return lfilter(*self._distance_filters[min(round(distance), self._max_distance)], sound, axis=1)

    def _apply_positioning(self, sound: np.ndarray, angle: float):
        """Convolve sound with a pair of HRIR filters corresponding to the given angle."""

        if angle is None:
            return np.pad(sound, ((0, 0), (127, 128)))

        angle *= self._DEG_TO_RAD_FACTOR

        if angle < 0.:
            angle += 360.

        angle = min(int(angle), 358)
        angle -= angle % 2

        return np.vstack([np.convolve(sound[ear], self._hrir_filters[angle][ear], mode='full') for ear in range(2)])


def get_mel_basis(sampling_rate: int = 44100, n_fft: int = 2048, n_mel: int = 64) -> np.ndarray:
    """
    Construct weights for resampling a spectrum on the mel scale.

    References:
    https://en.wikipedia.org/wiki/Mel_scale
    https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    https://librosa.org/doc/latest/generated/librosa.filters.mel.html#librosa.filters.mel
    """

    min_mel_freq = 0.
    max_mel_freq = 2595. * np.log10(1. + (sampling_rate/2.) / 700.)

    mel_freqs = np.linspace(min_mel_freq, max_mel_freq, n_mel + 2)
    hz_freqs = 700. * (10. ** (mel_freqs / 2595.) - 1.)
    fft_freqs = np.minimum(np.around((n_fft + 1) * hz_freqs / sampling_rate), float(n_fft//2))

    weights = np.zeros((n_mel, n_fft//2 + 1))

    # For each mel filter
    for i in range(1, n_mel + 1):
        freq_i_minus = int(fft_freqs[i - 1])
        freq_i = int(fft_freqs[i])
        freq_i_plus = int(fft_freqs[i + 1])

        # Construct left part of the triangular filter
        for freq_j in range(freq_i_minus, freq_i):
            weights[i - 1, freq_j] = (freq_j - fft_freqs[i - 1]) / (fft_freqs[i] - fft_freqs[i - 1])

        # Construct right part of the triangular filter
        for freq_j in range(freq_i, freq_i_plus + 1):
            weights[i - 1, freq_j] = (fft_freqs[i + 1] - freq_j) / (fft_freqs[i + 1] - fft_freqs[i])

    # Weights per filter should sum to 1 (const. area)
    # NOTE: If there are divide by zero warnings, it's a problem of discretisation
    # In that case, use more FFT points (`n_fft`) or fewer mel filters (`n_mel`)
    return weights / np.sum(weights, axis=1)[:, None]


def spectrify(
    sound: np.ndarray,
    mel_basis: Optional[np.ndarray] = None,
    window: Optional[np.ndarray] = None,
    sampling_rate: int = 44100,
    n_fft: int = 2048,
    n_mel: int = 64,
    eps: float = 1e-30
) -> np.ndarray:
    """
    Convert a stereo (2-channel) sound array into a pair of spectral vectors,
    using the mel frequency scale. `spectrify` is intended to be used on short
    chunks of sound, effectively constructing a larger spectrogram by pieces
    in real-time.

    References:
    https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html
    """

    if mel_basis is None:
        mel_basis = get_mel_basis(sampling_rate=sampling_rate, n_fft=n_fft, n_mel=n_mel)

    if window is None:
        window = np.hamming(sound.shape[1])[None, :]

    signal = window * sound / 2**15

    power_spectrum = np.power(np.abs(np.fft.rfft(signal, n_fft, axis=1)), 2)
    power_spectrum_mel = np.dot(mel_basis, power_spectrum.T)
    power_spectrum_db = 10.*np.log10(power_spectrum_mel + eps)

    return power_spectrum_db.T
