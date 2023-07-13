#!/usr/bin/env python

"""Run `SDGLiveClient` with specified settings."""

import os
import sys
import json
import argparse
from logging import DEBUG
from sidegame.game.client import SDGLiveClient


DATA_DIR = os.path.abspath('user_data')


def parse_args() -> argparse.Namespace:
    """Parse client args."""

    config_parser = argparse.ArgumentParser(description='Config file parser.', add_help=False)
    parser = argparse.ArgumentParser(description='Argument parser for the SDG live client.')

    config_parser.add_argument(
        '--config_file', type=str, default=os.path.join(DATA_DIR, 'config.json'),
        help='Path to the configuration file.')
    config_parser.add_argument(
        '-c', '--config', type=str, default='ClientDefault',
        help='A specific config among presets in the configuration file.')

    parser.add_argument('-m', '--mode', type=str, default='client', help='Switch between scripts to run.')

    parser.add_argument(
        '--time_scale', type=float, default=1., help='Simulation time factor affecting movement and decay formulae.')
    parser.add_argument(
        '-t', '--tick_rate', type=float, default=64.,
        help='Rate of updating the local game state in ticks (frames) per second.')
    parser.add_argument(
        '--refresh_rate', type=float, default=64.,
        help='Rate of updating the image on screen in monitor refreshes per second.')
    parser.add_argument(
        '--polling_rate', type=float, default=64.,
        help='Rate of polling peripheral events in evaluations per second.')
    parser.add_argument(
        '--sending_rate', type=float, default=32.,
        help='Rate of sending messages to the server in packets per second.')

    parser.add_argument('--address', type=str, default='localhost', help='Server address.')
    parser.add_argument('--port', type=int, default=49152, help='Server port.')

    parser.add_argument(
        '--record', '--rec', action='store_true',
        help='Record network data exchanged with the server and save it at the end of runtime.')
    parser.add_argument(
        '-r', '--recording_path', type=str, default=None,
        help='Path to an existing recording of network data exchanged with the server.')
    parser.add_argument(
        '--logging_path', type=str, default=None,
        help='If given, execution logs are written to a file at the specified location instead of stdout.')
    parser.add_argument(
        '--logging_level', type=int, default=DEBUG,
        help='Threshold above the severity of which the runtime messages are logged or displayed.')
    parser.add_argument(
        '--track_stats', action='store_true', help='Keep track of accumulated in-game values, '
        'which can be used to analyse the success and tendencies of the player.')
    parser.add_argument(
        '-f', '--show_fps', action='store_true',
        help='Print tracked frames-per-second to stdout.')
    parser.add_argument(
        '-d', '--discretise_mouse', action='store_true',
        help='Round polled mouse movements to nearest discrete preset.')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Seed for initialising random number generators. Note that the process is still subject to '
        'the unpredictability of the network and OS scheduling.')

    parser.add_argument(
        '-x', '--render_scale', type=float, default=5,
        help='Factor by which the base render is upscaled. Determines the width and height of the window. '
        'Setting it to 0 will make the display mode fullscreen and infer the scale factor.')
    parser.add_argument(
        '-s', '--mouse_sensitivity', type=float, default=1.,
        help='Factor by which mouse movement, i.e. pixel distance traversed between updates, is multiplied.')
    parser.add_argument(
        '--volume', type=float, default=1.,
        help='Initial factor by which in-game sound amplitudes are scaled.')
    parser.add_argument(
        '--audio_device', type=int, default=0,
        help='Index of output audio device. Default (0) may not correspond to assumed priority.')
    parser.add_argument(
        '--interp_ratio', type=float, default=2.,
        help='Ratio between kept states for entity interpolation and the update rate of the server. '
        'Corresponds to the amount of artificial lag introduced to the client.')

    parser.add_argument(
        '--role_key', type=str, default='00000000',
        help='Role key used to introduce a client to the server. Used for authentication and to limit user privileges.')
    parser.add_argument(
        '-i', '--name', '--player_id', type=str, default='user',
        help='4-character name used to introduce a client to the server. Used to distinguish between clients.')
    parser.add_argument(
        '--mmr', type=float, default=0.,
        help='Matchmaking rating (MMR) used to route a client through the matchmaking server. '
        'If 0, the address and port are assumed to belong to a session server, which will be contacted directly.')

    parser.add_argument(
        '--focus_path', type=str, default=None, help='Path to a recording of focal coordinates.')
    parser.add_argument(
        '--focus_record', action='store_true', help='Whether to record or replay focal coordinates.')
    parser.add_argument(
        '--monitoring_path', type=str, default=None, help='Path where monitoring results will be saved to.')
    parser.add_argument(
        '--monitoring_rate', type=float, default=10.,
        help='Rate of updating monitoring data: FPS, CPU perc. utilisation, and memory usage. Off by default.')

    args, remaining_args = config_parser.parse_known_args()

    if args.config_file:
        with open(args.config_file, 'r') as json_file:
            config = json.load(json_file)

        config = config[args.config]
        parser.set_defaults(**config)

    return parser.parse_args(remaining_args)


if __name__ == '__main__':
    parsed_args = parse_args()

    if parsed_args.record:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        file_indices = [
            int(filename.split('_')[2][:-4]) for filename in os.listdir(DATA_DIR)
            if (filename.startswith(parsed_args.name) and filename.endswith('.sdg'))]

        file_idx = (max(file_indices)+1) if file_indices else 0
        file_name = f'{parsed_args.name}_demo-{parsed_args.tick_rate}_{file_idx:03d}.sdg'

        parsed_args.recording_path = os.path.join(DATA_DIR, file_name)

    client = SDGLiveClient(parsed_args)
    sys.exit(client.run())
