#!/usr/bin/env python

"""Run `SDGTrainedActor` with specified settings."""

import os
import sys
import json
import argparse
from logging import DEBUG
from sdgai.actor import SDGTrainedActor


DATA_DIR = os.path.abspath(os.path.join('user_data'))
MODEL_DIR = os.path.abspath(os.path.join('models'))


def parse_args() -> argparse.Namespace:
    """Parse client args."""

    config_parser = argparse.ArgumentParser(description='Config file parser.', add_help=False)
    parser = argparse.ArgumentParser(description='Argument parser for the SDG trained actor client.')

    # Common args
    config_parser.add_argument(
        '--config_file', type=str, default=os.path.join(DATA_DIR, 'config.json'),
        help='Path to the configuration file.')
    config_parser.add_argument(
        '-c', '--config', type=str, default='ActorDefault',
        help='A specific config among presets in the configuration file.')

    parser.add_argument(
        '-t', '--tick_rate', type=float, default=30.,
        help='Rate of updating the local game state in ticks (frames) per second.')
    parser.add_argument(
        '--refresh_rate', type=float, default=30.,
        help='Rate of updating the action queue in inference calls per second.')
    parser.add_argument(
        '--polling_rate', type=float, default=30.,
        help='Rate of accessing the action queue in actions per second.')
    parser.add_argument(
        '--sending_rate', type=float, default=30.,
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
        help='Has no effect: Polled mouse movements are always based on discrete presets.')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Seed for initialising random number generators. Note that the process is still subject to '
        'the unpredictability of the network and OS scheduling.')

    parser.add_argument(
        '-x', '--render_scale', type=float, default=1,
        help='Has no effect: The base render is never upscaled, i.e. its width and height are fixed.')
    parser.add_argument(
        '-s', '--mouse_sensitivity', type=float, default=1.,
        help='Has no effect: Mouse movement, i.e. pixel distance traversed between updates, is never augmented.')
    parser.add_argument(
        '--volume', type=float, default=1.,
        help='Initial factor by which in-game sound amplitudes are scaled.')
    parser.add_argument(
        '--interp_ratio', type=float, default=2.,
        help='Ratio between kept states for entity interpolation and the update rate of the server. '
        'Corresponds to the amount of artificial lag introduced to the client.')

    parser.add_argument(
        '--role_key', type=str, default='00000000',
        help='Role key used to introduce a client to the server. Used for authentication and to limit user privileges.')
    parser.add_argument(
        '-i', '--name', '--player_id', type=str, default='slai',
        help='4-character name used to introduce a client to the server. Used to distinguish between clients.')
    parser.add_argument(
        '-m', '--mmr', type=float, default=0.,
        help='Matchmaking rating (MMR) used to route a client through the matchmaking server. '
        'If 0, the address and port are assumed to belong to a session server, which will be contacted directly.')

    # Actor-specific args
    parser.add_argument('--device', type=str, default='cpu', help='Target inference device.')
    parser.add_argument('-n-', '--n_agents', type=int, default=1, help='Number of agent copies to spawn.')
    parser.add_argument(
        '-p', '--model_path', type=str, default=os.path.join(MODEL_DIR, 'pcnet-slx.pth'), help='Path to trained model.')
    parser.add_argument(
        '--sampling_proba', '--sp', type=float, default=0.01,
        help='Probability of sampling to get actions from probabilities instead of argmax. '
        '0 corresponds to argmax and 1 to sampling on every step.')
    parser.add_argument(
        '--sampling_thr', '--st', type=float, default=0.1,
        help='Determines the upper and lower probability thresholds between which actions can be sampled.')

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

    actor = SDGTrainedActor(parsed_args)
    sys.exit(actor.run())
