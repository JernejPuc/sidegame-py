"""
Convert replays to separate video and audio recordings, which should then be
edited (merged) externally.
"""

import os
import argparse
from logging import DEBUG
from collections import deque
import wave
import cv2
import numpy as np
from sidegame.game.client import SDGReplayClient
from sidegame.audio import interleave


def parse_args() -> argparse.Namespace:
    """Parse arguments for the replay client and recorder."""

    parser = argparse.ArgumentParser(description='Argument parser for the SDG live client.')

    parser.add_argument(
        '--time_scale', type=float, default=1., help='Simulation time factor affecting movement and decay formulae.')
    parser.add_argument(
        '-x', '--render_scale', type=float, default=5,
        help='Factor by which the base render is upscaled. Determines the width and height of the window.')
    parser.add_argument(
        '--volume', type=float, default=1.,
        help='Initial factor by which in-game sound amplitudes are scaled.')
    parser.add_argument(
        '--seed', type=int, default=None, help='Seed for initialising random number generators.')

    parser.add_argument(
        '-i', '--init_perc', type=float, default=0.,
        help='Percentage of the replay, at which recording is started. Prior frames are simulated, but not recorded.')

    parser.add_argument(
        '-e', '--end_perc', type=float, default=100.,
        help='Percentage of the replay, at which recording is stopped. Any following frames are skipped.')

    parser.add_argument(
        '-f', '--video_fps', type=float, default=0,
        help='Target video framerate. Non-positive values specify FPS to be the same as the original tick rate.')
    parser.add_argument(
        '-c', '--codec', type=str, default='avc1', help='Video format in 4-character code.')
    parser.add_argument(
        '-a', '--audio_path', type=str, required=True, help='Audio output location (separate from video).')
    parser.add_argument(
        '-v', '--video_path', type=str, required=True, help='Video output location (separate from audio).')

    parser.add_argument(
        '-r', '--recording_path', type=str, required=True,
        help='Path to an existing recording of network data exchanged with the server.')
    parser.add_argument(
        '--focus_path', type=str, default=None, help='Path to a recording of focal coordinates.')
    parser.add_argument(
        '--logging_path', type=str, default=None,
        help='If given, execution logs are written to a file at the specified location instead of stdout.')
    parser.add_argument(
        '--logging_level', type=int, default=DEBUG,
        help='Threshold above the severity of which the runtime messages are logged or displayed.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.show_fps = False
    args.focus_record = False

    assert os.path.exists(args.recording_path), 'No recording found on given path.'
    assert len(args.codec) == 4, 'Invalid video format specification.'

    sdgr = SDGReplayClient(args, headless=True)

    video_fps = args.video_fps if args.video_fps > 0 else args.tick_rate
    video_out = cv2.VideoWriter(args.video_path, cv2.VideoWriter_fourcc(*args.codec), video_fps, sdgr.window_size)
    audio_queue = deque()

    clk = None

    while sdgr.recording:
        clk = sdgr.manual_step(clk)

        last_tick_counter = sdgr._tick_counter - 1
        perc = last_tick_counter / sdgr.max_tick_counter * 100.

        print(f'\rProcessing tick {last_tick_counter} of {sdgr.max_tick_counter} ({perc:.2f}%)          ', end='')

        sdgr.action_stream.clear()
        frame = sdgr.video_stream.popleft()
        sound = sdgr.audio_stream.popleft()

        if perc < args.init_perc:
            continue

        elif perc > args.end_perc:
            break

        video_out.write(frame)
        audio_queue.append(interleave(sound[..., 255:-255]).astype(np.int16).tobytes())

    video_out.release()

    with wave.open(args.audio_path, 'wb') as audio_out:
        audio_out.setframerate(sdgr.sim.audio_system.SAMPLING_RATE)
        audio_out.setnchannels(sdgr.sim.audio_system._N_CHANNELS)
        audio_out.setsampwidth(sdgr.sim.audio_system._SAMPLE_WIDTH)
        audio_out.writeframes(b''.join(audio_queue))

    print('\nDone.')
