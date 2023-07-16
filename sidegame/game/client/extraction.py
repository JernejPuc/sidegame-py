"""Extract observations from recorded SDG demos."""

import argparse
import os
from collections import deque

import numpy as np
import h5py

from sidegame.game import GameID
from sidegame.game.client import SDGReplayClient
from sidegame.audio import get_mel_basis, spectrify
from sidegame.graphics import get_camera_warp


SAMPLING_RATE = 44100
TICK_RATE = 30.
CHUNK_SIZE = int(SAMPLING_RATE/TICK_RATE)
HRIR_SIZE = 255
N_FFT = 2048
N_MEL = 64
EPS = 1e-12

MEL_BASIS = get_mel_basis(sampling_rate=SAMPLING_RATE, n_fft=N_FFT, n_mel=N_MEL)
WINDOW = np.hamming(CHUNK_SIZE+HRIR_SIZE*2)[None, :]
REF = np.power(WINDOW.sum(), 2) / 2.

N_DELAYED_FRAMES = 6
MOUSE_BINS = np.array([0., 0.48, 1.19, 2.23, 3.78, 6.06, 9.44, 14.43, 21.82, 32.73, 48.87, 72.73, 108.])


class FocusSampler:
    """Deprecated remnants of a past approach to estimating focal coordinates."""

    REDUCTION_NONE = 0
    REDUCTION_STD = 1
    REDUCTION_TRUNC = 2

    FOCUS_HEIGHT = 72
    FOCUS_WIDTH = 128
    FOCUS_HOLD_PROBA = 0.8

    MAIN_HEIGHT = 108//2
    CHAT_WIDTH = 64//2
    FOCUS_HEIGHT = 144//2
    FOCUS_WIDTH = 256//2

    CHUNKS = (6, 1, 72, 128)

    def __init__(self, seed: int = None):
        self.rng: np.random.Generator = np.random.default_rng(seed=seed)

        self.pos_x, self.pos_y = np.meshgrid(np.arange(self.FOCUS_WIDTH), np.arange(self.FOCUS_HEIGHT))

        self.base_focus = np.zeros((self.FOCUS_HEIGHT, self.FOCUS_WIDTH))
        self.chat_focus = np.zeros((self.FOCUS_HEIGHT, self.FOCUS_WIDTH))
        self.hud_focus = np.zeros((self.FOCUS_HEIGHT, self.FOCUS_WIDTH))

        self.chat_focus[:, :self.CHAT_WIDTH] = \
            np.linspace(0., 1., self.FOCUS_HEIGHT)[:, None].repeat(self.CHAT_WIDTH, axis=1)
        self.hud_focus[self.MAIN_HEIGHT:, self.CHAT_WIDTH:] = 1.
        self.chat_focus /= self.chat_focus.sum()
        self.hud_focus /= self.hud_focus.sum()

        self.focus_indices = [
            np.unravel_index(idx, (self.FOCUS_HEIGHT, self.FOCUS_WIDTH))
            for idx in range(self.FOCUS_HEIGHT * self.FOCUS_WIDTH)]

        self.last_chat_lengths = deque(False for _ in range(N_DELAYED_FRAMES))

    def gaussian(self, y: float, x: float, std: float = 1., reduction: int = REDUCTION_STD) -> np.ndarray:
        """Generate a gaussian distribution around given coordinates."""

        res = np.exp(-0.5 * ((x - self.pos_x)**2 + (y - self.pos_y)**2)/std**2)

        if reduction == self.REDUCTION_TRUNC:
            res = np.where(res < 1e-3, 0., res)
            res /= res.sum()

        elif reduction == self.REDUCTION_STD:
            res *= 1. / (std**2 * 2.*np.pi)

        return res

    def get_focal_distribution(
        self,
        sdgr: SDGReplayClient,
        space_pressed: bool,
        message_received: bool,
        initial_frame: bool = False
    ) -> np.ndarray:
        """
        Get suggested focal distribution based on view, cursor coordinates,
        and entity positions.
        """

        # Get base highlights
        if initial_frame:
            focus = self.base_focus

        elif space_pressed or message_received:
            focus = self.chat_focus

        elif sdgr.sim.view == GameID.VIEW_WORLD:
            focus = self.hud_focus

        elif sdgr.sim.view == GameID.VIEW_STORE:
            focus = self.base_focus

        else:
            focus = self.chat_focus

        # Get cursor highlight
        std = 5. if sdgr.sim.view == GameID.VIEW_WORLD else 7.5
        focus = focus + self.gaussian(
            sdgr.sim.cursor_y//2., sdgr.sim.cursor_x//2., std=std, reduction=self.REDUCTION_TRUNC)

        if sdgr.sim.view != GameID.VIEW_WORLD or initial_frame:
            return focus / focus.sum()

        # Get entity highlights
        player = sdgr.session.players[sdgr.sim.observed_player_id]
        pos = tuple(player.pos)
        origin = tuple(player.d_pos_recoil + sdgr.sim.WORLD_FRAME_ORIGIN)
        angle = player.angle + np.pi/2. + player.d_angle_recoil
        world_warp = get_camera_warp(pos, angle, origin)

        ent_focus = self.base_focus

        for a_player in sdgr.session.players.values():
            if sdgr.sim.check_los(player, a_player):
                pos_x, pos_y = np.dot(world_warp, (*a_player.pos, 1.))

                if 0. <= pos_y <= 107. and 0. <= pos_x <= 191.:
                    ent_focus = ent_focus + self.gaussian(
                        pos_y//2., 32. + pos_x//2., std=1., reduction=self.REDUCTION_NONE)

        for an_object in sdgr.session.objects.values():
            if sdgr.sim.check_los(player, an_object):
                pos_x, pos_y = np.dot(world_warp, (*an_object.pos, 1.))

                if 0. <= pos_y <= 107. and 0. <= pos_x <= 191.:
                    ent_focus = ent_focus + self.gaussian(
                        pos_y//2., 32. + pos_x//2., std=1., reduction=self.REDUCTION_NONE)

        ent_focus_sum = ent_focus.sum()

        if ent_focus_sum:
            focus = focus + ent_focus / ent_focus_sum

        return focus / focus.sum()

    def get_focus(self, sdgr: SDGReplayClient, space: bool, frame_number: int) -> np.ndarray:
        """Get suggested focal distribution."""

        self.last_chat_lengths.append(len(sdgr.sim.chat) > self.last_chat_lengths[-1])

        return self.get_focal_distribution(
            sdgr, space, self.last_chat_lengths.popleft(), initial_frame=(frame_number < 2*N_DELAYED_FRAMES))

    def sample_focus(self, focus: np.ndarray, slice_length: int, batch_size: int) -> np.ndarray:
        """
        Get indices of focal points by sampling from suggested probability
        distributions. Sampled points can be held for a few consecutive frames.

        NOTE: Sampling can cause a noticeable bottleneck in a training process.
        Its complexity is determined by sequence length, batch size, and,
        most heavily, image dimensions.
        """

        # n, b, h, w -> n*b, h*w -> n*b, i -> n, b, i
        focus = focus.reshape(slice_length*batch_size, self.FOCUS_HEIGHT*self.FOCUS_WIDTH)
        focus = np.array([self.rng.choice(self.focus_indices, p=focus_i) for focus_i in focus])
        focus = focus.reshape((slice_length, batch_size, 2))

        for n in range(1, slice_length):
            mask = self.rng.uniform(size=batch_size) > self.FOCUS_HOLD_PROBA
            focus[n] = np.where(mask[:, None], focus[n], focus[n-1])

        return focus

    def dummy_sample(self, slice_length: int, batch_size: int):
        """
        Perform the same number of calls to the random number generator
        as if focus were actually sampled.
        """

        _ = [self.rng.choice(self.focus_indices) for _ in range(slice_length * batch_size)]

        for _ in range(1, slice_length):
            _ = self.rng.uniform(size=batch_size)


def extract(args: argparse.Namespace):
    args.focus_record = False
    args.time_scale = 1.  # TODO: Inferring time scale from replays is yet to be implemented

    assert os.path.exists(args.recording_path), 'No recording found on given path.'
    assert not os.path.exists(args.output_path), 'Output file already exists.'

    sdgr = SDGReplayClient(args, headless=True)
    logger = sdgr.logger

    input_path = args.recording_path
    output_path = args.output_path
    sequence_key = args.sequence_key
    frame_limit = args.frame_limit
    sub_px_threshold = args.sub_px_threshold
    rng = np.random.default_rng(seed=args.seed)

    frames = deque()
    spectra = deque()
    mkbd_states = deque()
    foci = deque()
    cursor_y = deque()
    cursor_x = deque()
    actions = deque()

    clk = None
    frame_number = 0

    while sdgr.recording and frame_number < frame_limit:
        clk = sdgr.manual_step(clk)

        # Progress
        last_tick_counter = sdgr._tick_counter - 1
        perc = last_tick_counter / sdgr.max_tick_counter * 100.

        print(f'\rProcessing tick {last_tick_counter} of {sdgr.max_tick_counter} ({perc:.2f}%)          ', end='')

        # Skip until match start
        if sdgr.sim.view == GameID.VIEW_LOBBY:
            sdgr.video_stream.clear()
            sdgr.audio_stream.clear()
            sdgr.action_stream.clear()
            continue

        # Extract action
        lbtn, rbtn, space, ekey, gkey, rkey, draw_id, crsr_x, crsr_y, ws1, da1, mwhl_y1, _, _, _, d_angle = \
            sdgr.action_stream.popleft()

        kbd_num = 0 if draw_id == GameID.NULL else sdgr.sim.inventory.get_item_by_id(draw_id).slot

        xrel = d_angle if sdgr.sim.view == GameID.VIEW_WORLD else ((crsr_x - cursor_x[-1]) if cursor_x else 0.)
        yrel = (crsr_y - cursor_y[-1]) if cursor_y else 0.

        # Bandaid for information loss due to rounding for int format
        if abs(xrel) == 1 and rng.random() > sub_px_threshold:
            xrel *= MOUSE_BINS[1]

        if abs(yrel) == 1 and rng.random() > sub_px_threshold:
            yrel *= MOUSE_BINS[1]

        xrel_idx = np.argmin(np.abs(np.abs(xrel) - MOUSE_BINS))
        yrel_idx = np.argmin(np.abs(np.abs(yrel) - MOUSE_BINS))

        # Mouse motion to [-1, 1] range
        mmot_xrel = np.log(MOUSE_BINS[xrel_idx] + 1.) / np.log(MOUSE_BINS[-1] + 1.) * np.sign(xrel)
        mmot_yrel = np.log(MOUSE_BINS[yrel_idx] + 1.) / np.log(MOUSE_BINS[-1] + 1.) * np.sign(yrel)

        wkey = int((ws1-1) > 0)
        skey = int((ws1-1) < 0)
        dkey = int((da1-1) > 0)
        akey = int((da1-1) < 0)

        wheel_y = mwhl_y1 - 1

        bkey = int(sdgr.sim.view == GameID.VIEW_STORE)
        xkey = int(sdgr.sim.view == GameID.VIEW_TERMS)
        ckey = int(sdgr.sim.view == GameID.VIEW_ITEMS)
        tab = int(sdgr.sim.view == GameID.VIEW_MAPSTATS)

        # Skip some actions to sync them with causing observations
        if frame_number >= N_DELAYED_FRAMES:
            action = [lbtn, rbtn, space, ekey, gkey, rkey, dkey, akey, wkey, skey, tab, xkey, ckey, bkey]

            num_cats = [0]*5

            if kbd_num != 0:
                num_cats[kbd_num-1] = 1

            mmot_xrel_cats = [0]*(2*len(MOUSE_BINS)-1)
            mmot_yrel_cats = [0]*(2*len(MOUSE_BINS)-1)
            mmot_xrel_cats[len(MOUSE_BINS)-1 + int(xrel_idx * np.sign(xrel))] = 1
            mmot_yrel_cats[len(MOUSE_BINS)-1 + int(yrel_idx * np.sign(yrel))] = 1

            wheel_y_cats = [0]*3
            wheel_y_cats[int(mwhl_y1)] = 1

            action = action + num_cats + mmot_yrel_cats + mmot_xrel_cats + wheel_y_cats
            actions.append(action)

            if sdgr.focus.mode:
                foci.append((sdgr.focus.y, sdgr.focus.x))

        # Image to [0, 1] range
        frames.append(sdgr.video_stream.popleft()[..., ::-1] / 255.)

        # Audio to [0, 1] range
        spectral_vectors = spectrify(
            sdgr.audio_stream.popleft(),
            mel_basis=MEL_BASIS,
            window=WINDOW,
            sampling_rate=SAMPLING_RATE,
            n_fft=N_FFT,
            n_mel=N_MEL,
            eps=EPS,
            ref=REF)

        spectral_vectors = spectral_vectors / (-10.*np.log10(EPS)) + 1.
        spectra.append(spectral_vectors)

        # Unchanged cursor data
        cursor_y.append(sdgr.sim.cursor_y)
        cursor_x.append(sdgr.sim.cursor_x)

        # Num key to [0, 1] range
        kbd_num /= 5

        # Cursor coordinates to [-1, 1] range
        crsr_y = (crsr_y - 53.5) / 51.5
        crsr_x = (crsr_x - 64. - 95.5) / 93.5

        # Observed mouse/keyboard state
        mkbd_state = [
                wkey, skey, dkey, akey, ekey, rkey, gkey, bkey, xkey, ckey, tab, space, lbtn, rbtn,
                kbd_num, wheel_y, mmot_yrel, mmot_xrel, crsr_y, crsr_x]

        mkbd_states.append(mkbd_state)

        frame_number += 1

    print()

    # Drop incomplete IO pairs
    min_len = min(len(buffer) for buffer in (frames, spectra, mkbd_states, foci, cursor_y, cursor_x, actions))

    for buffer in (frames, spectra, mkbd_states, foci, cursor_y, cursor_x, actions):
        while len(buffer) > min_len:
            buffer.pop()

    logger.info('Gathering arrays...')

    # Convert to arrays
    frames = np.array(frames, dtype=np.float32)
    spectra = np.array(spectra, dtype=np.float32)
    mkbd_states = np.array(mkbd_states, dtype=np.float32)
    foci = np.array(foci, dtype=np.int32)
    cursor_y = np.array(cursor_y, dtype=np.int32)
    cursor_x = np.array(cursor_x, dtype=np.int32)
    actions = np.array(actions, dtype=np.float32)

    # Reshape for batching and NCHW format
    frames = np.moveaxis(frames, 3, 1)[:, None]
    spectra = spectra[:, None]
    mkbd_states = mkbd_states[:, None]
    foci = foci[:, None]
    cursor_coords = np.vstack((cursor_y, cursor_x)).T[:, None]
    actions = actions[:, None]

    total_size = sum(arr.nbytes for arr in (frames, spectra, mkbd_states, foci, cursor_coords, actions))

    if total_size > 1e+9:
        logger.debug('Uncompressed size: %.2fGB', total_size / 1024**3)

    else:
        logger.debug('Uncompressed size: %.0fMB', total_size / 1024**2)

    logger.info('Compressing...')

    # NOTE: Chunks are built for fast access of temporal sub-sequences (indexing across the first/zeroth axis)
    # instead of compression efficiency, i.e. size on disk
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset('image', data=frames, compression='gzip', compression_opts=4, chunks=(2, 1, 3, 144, 256))
        hf.create_dataset('spectrum', data=spectra, compression='gzip', compression_opts=4, chunks=(12, 1, 2, 64))
        hf.create_dataset('mkbd', data=mkbd_states, compression='gzip', compression_opts=4, chunks=(12, 1, 20))
        hf.create_dataset('focus', data=foci, compression='gzip', compression_opts=4, chunks=(12, 1, 2))
        hf.create_dataset('cursor', data=cursor_coords, compression='gzip', compression_opts=4, chunks=(12, 1, 2))
        hf.create_dataset('action', data=actions, compression='gzip', compression_opts=4, chunks=(12, 1, 72))
        hf.attrs['src'] = os.path.split(input_path)[-1]
        hf.attrs['len'] = len(frames)
        hf.attrs['key'] = sequence_key

    sdgr.cleanup()

    logger.info('Done.')
