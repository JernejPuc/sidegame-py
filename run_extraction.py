"""WIP script to extract observations from recorded SDG demos"""

import os
import sys
from collections import deque
from argparse import Namespace
import numpy as np
from sidegame.game.shared import GameID
from sidegame.game.client import SDGReplayClient
from sidegame.audio import get_mel_basis, spectrify


sampling_rate = 44100
tick_rate = 30.
chunk_size = int(sampling_rate/tick_rate)
hrir_size = 255
n_fft = 2048
n_mel = 64
eps = 1e-12

mel_basis = get_mel_basis(sampling_rate=sampling_rate, n_fft=n_fft, n_mel=n_mel)
window = np.hamming(chunk_size+hrir_size*2)[None, :]
ref = np.power(window.sum(), 2) / 2.

n_delayed_frames = 6
mouse_bins = np.array([0., 0.48, 1.19, 2.23, 3.78, 6.06, 9.44, 14.43, 21.82, 32.73, 48.87, 72.73, 108.])


if __name__ == '__main__':
    assert len(sys.argv) > 3, 'Not enough input arguments.'

    recording_path = sys.argv[1]
    output_path = sys.argv[2]
    sequence_key = int(sys.argv[3])
    frame_limit = int(sys.argv[4]) if len(sys.argv) > 4 else 1000000

    assert os.path.exists(recording_path), 'No recording found on given path.'
    assert not os.path.exists(output_path), 'Output file already exists.'

    args = Namespace()
    args.seed = 42
    args.recording_path = recording_path
    args.logging_path = None
    args.logging_level = 10
    args.show_fps = False
    args.volume = 1.
    args.render_scale = 1.

    sdgr = SDGReplayClient(args, headless=True)

    frames = deque()
    spectra = deque()
    cursor_y = deque()
    cursor_x = deque()
    mkbd_states = deque()
    actions = deque()

    clk = None
    frame_number = 0

    while sdgr.recording and frame_number < frame_limit:
        clk = sdgr.manual_step(clk)

        # Progress
        perc = sdgr._tick_counter / sdgr.max_tick_counter * 100.
        print(f'\rProcessing tick {sdgr._tick_counter} of {sdgr.max_tick_counter} ({perc:.2f}%)          ', end='')

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

        xrel_idx = np.argmin(np.abs(np.abs(xrel) - mouse_bins))
        yrel_idx = np.argmin(np.abs(np.abs(yrel) - mouse_bins))

        mmot_xrel = mouse_bins[xrel_idx] * np.sign(xrel)
        mmot_yrel = mouse_bins[yrel_idx] * np.sign(yrel)

        wkey = int((ws1-1) > 0)
        skey = int((ws1-1) < 0)
        dkey = int((da1-1) > 0)
        akey = int((da1-1) < 0)

        wheel_y = mwhl_y1 - 1

        bkey = int(sdgr.sim.view == GameID.VIEW_STORE)
        xkey = int(sdgr.sim.view == GameID.VIEW_TERMS)
        ckey = int(sdgr.sim.view == GameID.VIEW_ITEMS)
        tab = int(sdgr.sim.view == GameID.VIEW_MAPSTATS)

        # Mouse motion to [-1, 1] range
        mmot_xrel /= mouse_bins[-1]
        mmot_yrel /= mouse_bins[-1]

        # Inferred action
        action = [lbtn, rbtn, space, ekey, gkey, rkey, dkey, akey, wkey, skey, tab, xkey, ckey, bkey]

        num_cats = [0]*5

        if kbd_num != 0:
            num_cats[kbd_num-1] = 1

        mmot_xrel_cats = [0]*(2*len(mouse_bins)-1)
        mmot_yrel_cats = [0]*(2*len(mouse_bins)-1)
        mmot_xrel_cats[len(mouse_bins)-1 + int(xrel_idx * np.sign(xrel))] = 1
        mmot_yrel_cats[len(mouse_bins)-1 + int(yrel_idx * np.sign(yrel))] = 1

        wheel_y_cats = [0]*3
        wheel_y_cats[int(mwhl_y1)] = 1

        action = action + num_cats + mmot_yrel_cats + mmot_xrel_cats + wheel_y_cats
        actions.append(action)

        # Skip some observations to sync them with delayed actions
        if frame_number < n_delayed_frames:
            sdgr.video_stream.clear()
            sdgr.audio_stream.clear()

        else:
            # Image to [0, 1] range
            frames.append(sdgr.video_stream.popleft()[..., ::-1] / 255.)

            # Audio to [0, 1] range
            spectral_vectors = spectrify(
                sdgr.audio_stream.popleft(), mel_basis, window, sampling_rate, n_fft, n_mel, eps, ref)

            spectral_vectors = spectral_vectors / (-10.*np.log10(eps)) + 1.
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

    # Drop incomplete IO pairs
    min_len = min(len(buffer) for buffer in (frames, spectra, cursor_y, cursor_x, mkbd_states, actions))

    for buffer in (frames, spectra, cursor_y, cursor_x, mkbd_states, actions):
        while len(buffer) > min_len:
            buffer.pop()

    print('\nGathering into arrays...')

    # Convert to arrays
    frames = np.array(frames, dtype=np.float32)
    spectra = np.array(spectra, dtype=np.float32)
    cursor_y = np.array(cursor_y, dtype=np.float32)
    cursor_x = np.array(cursor_x, dtype=np.float32)
    mkbd_states = np.array(mkbd_states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)

    cursor = np.vstack((cursor_y, cursor_x)).T
    meta_key = np.array(sequence_key, ndmin=1, dtype=np.int32)

    print('Compressing...')

    np.savez_compressed(
        output_path,
        image=frames,
        spectrum=spectra,
        mkbd=mkbd_states,
        cursor=cursor,
        action=actions,
        meta=meta_key)

    sdgr.cleanup()

    print('Done.\n')
