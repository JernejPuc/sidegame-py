#!/usr/bin/env python

"""Extract observations from recorded SDG demos."""

import os
import argparse
from logging import DEBUG
from collections import deque
import numpy as np
import h5py
from sidegame.game.shared import GameID
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

MAIN_HEIGHT = 108//2
CHAT_WIDTH = 64//2
FOCUS_HEIGHT = 144//2
FOCUS_WIDTH = 256//2
POS_X, POS_Y = np.meshgrid(np.arange(FOCUS_WIDTH), np.arange(FOCUS_HEIGHT))

BASE_FOCUS = np.zeros((FOCUS_HEIGHT, FOCUS_WIDTH))
CHAT_FOCUS = np.zeros((FOCUS_HEIGHT, FOCUS_WIDTH))
HUD_FOCUS = np.zeros((FOCUS_HEIGHT, FOCUS_WIDTH))
CHAT_FOCUS[:, :CHAT_WIDTH] = np.linspace(0., 1., FOCUS_HEIGHT)[:, None].repeat(CHAT_WIDTH, axis=1)
HUD_FOCUS[MAIN_HEIGHT:, CHAT_WIDTH:] = 1.
CHAT_FOCUS /= CHAT_FOCUS.sum()
HUD_FOCUS /= HUD_FOCUS.sum()

REDUCTION_NONE = 0
REDUCTION_STD = 1
REDUCTION_TRUNC = 2


def gaussian(y: float, x: float, std: float = 1., reduction: int = REDUCTION_STD) -> np.ndarray:
    """Generate a gaussian distribution around given coordinates."""

    res = np.exp(-0.5 * ((x - POS_X)**2 + (y - POS_Y)**2)/std**2)

    if reduction == REDUCTION_TRUNC:
        res = np.where(res < 1e-3, 0., res)
        res /= res.sum()

    elif reduction == REDUCTION_STD:
        res *= 1. / (std**2 * 2.*np.pi)

    return res


def get_focal_distribution(
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
        focus = BASE_FOCUS

    elif space_pressed or message_received:
        focus = CHAT_FOCUS

    elif sdgr.sim.view == GameID.VIEW_WORLD:
        focus = HUD_FOCUS

    elif sdgr.sim.view == GameID.VIEW_STORE:
        focus = BASE_FOCUS

    else:
        focus = CHAT_FOCUS

    # Get cursor highlight
    std = 5. if sdgr.sim.view == GameID.VIEW_WORLD else 7.5
    focus = focus + gaussian(sdgr.sim.cursor_y//2., sdgr.sim.cursor_x//2., std=std, reduction=REDUCTION_TRUNC)

    if sdgr.sim.view != GameID.VIEW_WORLD or initial_frame:
        return focus / focus.sum()

    # Get entity highlights
    player = sdgr.session.players[sdgr.sim.observed_player_id]
    pos = tuple(player.pos)
    origin = tuple(player.d_pos_recoil + sdgr.sim.WORLD_FRAME_ORIGIN)
    angle = player.angle + np.pi/2. + player.d_angle_recoil
    world_warp = get_camera_warp(pos, angle, origin)

    ent_focus = BASE_FOCUS

    for a_player in sdgr.session.players.values():
        if sdgr.sim.check_los(player, a_player):
            pos_x, pos_y = np.dot(world_warp, (*a_player.pos, 1.))

            if 0. <= pos_y <= 107. and 0. <= pos_x <= 191.:
                ent_focus = ent_focus + gaussian(pos_y//2., 32. + pos_x//2., std=1., reduction=REDUCTION_NONE)

    for an_object in sdgr.session.objects.values():
        if sdgr.sim.check_los(player, an_object):
            pos_x, pos_y = np.dot(world_warp, (*an_object.pos, 1.))

            if 0. <= pos_y <= 107. and 0. <= pos_x <= 191.:
                ent_focus = ent_focus + gaussian(pos_y//2., 32. + pos_x//2., std=1., reduction=REDUCTION_NONE)

    ent_focus_sum = ent_focus.sum()

    if ent_focus_sum:
        focus = focus + ent_focus / ent_focus_sum

    return focus / focus.sum()


def parse_args() -> argparse.Namespace:
    """Parse extraction args."""

    parser = argparse.ArgumentParser(description='Argument parser for data extraction from SDG demo files.')

    parser.add_argument(
        '-r', '--recording_path', type=str, required=True,
        help='Path to an existing recording of network data exchanged with the server.')
    parser.add_argument(
        '-o', '--output_path', type=str, required=True,
        help='Path to which the extracted data will be written.')
    parser.add_argument(
        '-k', '--sequence_key', type=int, required=True,
        help='Unique key by which the sequence source of the data will be identified.')

    parser.add_argument(
        '--seed', type=int, default=42,
        help='Seed for initialising random number generators.')
    parser.add_argument(
        '--frame_limit', type=int, default=1000000,
        help='Maximum number of ticks/frames that can be processed per extraction.')
    parser.add_argument(
        '--sub_px_threshold', type=float, default=0.75,
        help='RNG threshold to treat rounded 1px mouse movement as sub-1px instead.')

    parser.add_argument(
        '--logging_path', type=str, default=None,
        help='If given, execution logs are written to a file at the specified location instead of stdout.')
    parser.add_argument(
        '--logging_level', type=int, default=DEBUG,
        help='Threshold above the severity of which the runtime messages are logged or displayed.')
    parser.add_argument(
        '--show_fps', action='store_true',
        help='Print tracked frames-per-second to stdout.')
    parser.add_argument(
        '--render_scale', type=float, default=1,
        help='Factor by which the base render is upscaled. Determines the width and height of the window.')
    parser.add_argument(
        '--volume', type=float, default=1.,
        help='Initial factor by which in-game sound amplitudes are scaled.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

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
    last_chat_lengths = deque(False for _ in range(N_DELAYED_FRAMES))

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

            # Suggested focal distribution
            focus = get_focal_distribution(
                sdgr, bool(space), last_chat_lengths.popleft(), initial_frame=(frame_number < 2*N_DELAYED_FRAMES))

            foci.append(focus)
            last_chat_lengths.append(len(sdgr.sim.chat) > last_chat_lengths[-1])

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
    foci = np.array(foci, dtype=np.float32)
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
        hf.create_dataset('focus', data=foci, compression='gzip', compression_opts=4, chunks=(6, 1, 72, 128))
        hf.create_dataset('cursor', data=cursor_coords, compression='gzip', compression_opts=4, chunks=(12, 1, 2))
        hf.create_dataset('action', data=actions, compression='gzip', compression_opts=4, chunks=(12, 1, 72))
        hf.attrs['src'] = os.path.split(input_path)[-1]
        hf.attrs['len'] = len(frames)
        hf.attrs['key'] = sequence_key

    sdgr.cleanup()

    logger.info('Done.')
