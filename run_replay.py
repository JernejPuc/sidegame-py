#!/usr/bin/env python

"""Run `SDGReplayClient` with specified settings."""

import sys
from sidegame.game.client import SDGReplayClient
from run_client import parse_args


if __name__ == '__main__':
    client = SDGReplayClient(parse_args())
    sys.exit(client.run())
