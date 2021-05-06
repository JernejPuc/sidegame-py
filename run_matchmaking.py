#!/usr/bin/env python

"""Run `SDGMatchmaker` with specified settings."""

import sys
from sidegame.game.matchmaking import SDGMatchmaker
from run_server import parse_args


if __name__ == '__main__':
    args = parse_args()

    address = (args.address, args.main_port)
    subports = list(range(args.main_port+1, args.main_port+1+args.n_subports))

    server = SDGMatchmaker(args, address, subports)
    sys.exit(server.run())
