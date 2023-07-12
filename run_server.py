#!/usr/bin/env python

"""Run `SDGServer` or `SDGMatchmaker` with specified settings."""

import argparse
import json
import os
import sys
from logging import DEBUG

from sidegame.game.matchmaking import SDGMatchmaker
from sidegame.game.server import SDGServer


DATA_DIR = os.path.abspath('user_data')


def parse_args() -> argparse.Namespace:
    """
    Parse server args.

    NOTE: To be able to connect to the server on a local network,
    '' must be used for the address instead of 'localhost'. See:
    https://stackoverflow.com/questions/16130786/why-am-i-getting-the-error-connection-refused-in-python-sockets/16130819#16130819
    """

    config_parser = argparse.ArgumentParser(description='Config file parser.', add_help=False)
    parser = argparse.ArgumentParser(description='Argument parser for the SDG live server.')

    config_parser.add_argument(
        '--config_file', type=str, default=os.path.join(DATA_DIR, 'config.json'),
        help='Path to the configuration file.')
    config_parser.add_argument(
        '-c', '--config', type=str, default='ServerDefault',
        help='A specific config among presets in the configuration file.')

    parser.add_argument('-m', '--mode', type=str, default='server', help='Switch between scripts to run.')

    parser.add_argument(
        '--time_scale', type=float, default=1., help='Simulation time factor affecting movement and decay formulae.')
    parser.add_argument(
        '-t', '--tick_rate', type=float, default=64.,
        help='Rate of updating the game state in ticks per second.')
    parser.add_argument(
        '--update_rate', type=float, default=32.,
        help='Rate of sending messages to clients in packets per second.')

    parser.add_argument(
        '--ipconfig_file', type=str, default=os.path.join(DATA_DIR, 'config_ip.json'),
        help='Path to the IP configuration file.')
    parser.add_argument('--address', type=str, default='', help='Server address.')
    parser.add_argument(
        '--main_port', '--port', type=int, default=49152, help='Main port for the matchmaking or session server.')
    parser.add_argument(
        '-n', '--n_subports', '--n_procs', type=int, default=1,
        help='Number of ports following the main one, to be used for spawning session server processes.')
    parser.add_argument(
        '--lwtime_scale', type=float, default=8.5,
        help='Factor for weighting less waiting time (in seconds) wrt. MMR points when estimating a match.')
    parser.add_argument(
        '--limit_mmr', action='store_true',
        help='Use and gradually expand the maximum MMR difference allowed between players on the same team.')

    parser.add_argument(
        '--logging_path', type=str, default=None,
        help='If given, execution logs are written to a file at the specified location instead of stdout.')
    parser.add_argument(
        '--logging_level', type=int, default=DEBUG,
        help='Threshold above the severity of which the runtime messages are logged or displayed.')
    parser.add_argument(
        '-f', '--show_ticks', action='store_true',
        help='Print tracked ticks-per-second to stdout.')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Seed for initialising random number generators. Note that the process is still subject to '
        'the unpredictability of the network and OS scheduling.')

    parser.add_argument('-i', '--session_id', type=str, default=None, help='Session server identifier.')
    parser.add_argument(
        '--admin_key', type=str, default='00000000', help='Authentication key with admin privileges.')
    parser.add_argument(
        '--player_key', type=str, default='11111111', help='Authentication key with player privileges.')
    parser.add_argument(
        '--spectator_key', type=str, default='22222222', help='Authentication key with spectator privileges.')

    args, remaining_args = config_parser.parse_known_args()

    if args.config_file:
        with open(args.config_file, 'r') as json_file:
            config = json.load(json_file)

        config = config[args.config]
        parser.set_defaults(**config)

    return parser.parse_args(remaining_args)


if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'server':
        args.port = args.main_port

        if args.ipconfig_file:
            with open(args.ipconfig_file, 'r') as json_file:
                ip_config = json.load(json_file)

        else:
            ip_config = None

        server = SDGServer(args, ip_config=ip_config)

    elif args.mode == 'matchmaking':
        address = (args.address, args.main_port)
        subports = list(range(args.main_port+1, args.main_port+1+args.n_subports))

        server = SDGMatchmaker(args, address, subports, args.seed, args.lwtime_scale, args.limit_mmr)

    else:
        print(f'Mode {args.mode} not recognised.')
        raise SystemExit

    sys.exit(server.run())
