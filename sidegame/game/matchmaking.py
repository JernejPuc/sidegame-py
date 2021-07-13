"""SDG matchmaking"""

import struct
import random
from argparse import Namespace
from collections import deque
from itertools import combinations
from typing import Dict, Iterable, Tuple
from sidegame.networking.matchmaking import Client, Matchmaker
from sidegame.game.server import SDGServer
from sidegame.game.shared.core import GameID


class SDGClient(Client):
    """
    A representation of a SDG client awaiting redirection to the SDG session
    server, with attributes to determine its priority.
    """

    def __init__(self, own_address: Tuple[str, int], timestamp: float):
        super().__init__(own_address, timestamp)

        self.mmr: float = None
        self.name: str = None
        self.team: int = None

    def unpack_client_data(self, data: bytes):
        request = SDGServer.unpack_single(data)[0].data
        self.mmr = struct.unpack('>f', struct.pack('>4B', *request[4:8]))[0]
        self.name = ''.join(chr(ordinal) for ordinal in request[:4])

    def pack_redirection_data(self, subport: int) -> bytes:
        return struct.pack(
            '>HhBLf12f3B', 0, 0, 0, 0, 0, float(subport), 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0, 0)

    def pack_confirmation_data(self, timestamp: float) -> bytes:
        return struct.pack('>HhBLf12f3B', 0, 0, 0, 0, 0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0, 0)


class SDGMatchmaker(Matchmaker):
    """
    Accepts client (actor) requests and manages a pool of subprocesses
    (using standard multiprocessing and sockets) to spawn SDG sessions.
    These detached environments act as authoritative servers for
    multiple agents interacting in a shared environment (e.g. game matches).

    Clients looking for a match send over their MMR values,
    which are used to set up balanced teams of 5v5 players.

    After the teams are set up, the agents are assigned to a session handler,
    which acts as the authoritative server for their match.
    """

    CLIENT_MESSAGE_SIZE: int = SDGServer.CLIENT_MESSAGE_SIZE
    N_PLAYERS_TO_MATCH = 10

    def __init__(
        self,
        session_args: Namespace,
        address: Tuple[str, int],
        subports: Iterable[int],
        seed: int = None,
        lwtime_to_mmr_scale: float = 8.5,
        expand_mmr_diff_limit: bool = True
    ):
        super().__init__(session_args, address, subports, self.CLIENT_MESSAGE_SIZE)
        self.no_matched_clients: Dict = {}
        self.ref_client: SDGClient = None

        # Each second of less waiting is equivalent to 8.5 MMR diff
        # Should allow a client that connects after 10 seconds with 50 MMR diff
        # to overshadow clients connected after 1 or 0 seconds with 100 MMR diff
        # (note that `get_max_mmr_diff` only allows 100 MMR diff after 10 seconds itself)
        self.lwtime_to_mmr_scale = lwtime_to_mmr_scale

        # If `False`, `get_max_mmr_diff` will not be called and no MMR restrictions will be applied
        # Thus, a match will be returned as soon as enough players connect, only subject to team balancing
        self.expand_mmr_diff_limit = expand_mmr_diff_limit

        # NOTE: Seeding random number generators is diminished by network and OS unpredictability
        random.seed(seed)

    def create_waiting_client(self, client_address: Tuple[str, int], timestamp: float) -> Client:
        return SDGClient(timestamp, client_address)

    def eval_waiting_clients(self, clients: Dict[Tuple[str, int], Client], timestamp: float) -> Iterable[Client]:
        """
        Try to compose a new match. Any data that is intended to determine
        how `run_session` should handle the matched clients should be attributed
        to them before they are returned and passed on.

        This evaluation simply makes up one team based on time spent waiting
        and closeness in MMR, then makes up the other by finding closest
        counterparts for each client in the made up team, then rechecks
        the combinations between them to fix inter-team balance.
        While it would be better to approach this differently, naive exhaustive
        search would not be worth the computational load and more sophisticated
        methods would be overkill at this point.

        NOTE: For AI setups, this would be the second tier of selection;
        first, the controller should determine which models and actors to spawn,
        these then connect to the matchmaking server and proceed from here.
        With that in mind, it would probably be best to define a new matchmaker,
        which would not do any selection of its own and strictly follow the
        instructions originating from the controller.
        """

        if len(clients) < self.N_PLAYERS_TO_MATCH:
            return self.no_matched_clients

        client_set = set(clients.values())

        # Get a reference based on time of first contact not to overlook longest waiting clients
        self.ref_client = min(client_set, key=self.time_key)
        time_of_first_contact = self.ref_client.time_of_first_contact

        # Long waiting clients can outweigh clients with close MMR in selection (param. by weighting)
        # If matching could be ignored when not ideal (e.g. MMR diffs too large in the end),
        # this would allow a match to be eventually found under increasingly less ideal conditions
        team_1 = set(sorted(client_set, key=self.dist_key)[:self.N_PLAYERS_TO_MATCH//2])
        team_2 = set()

        # Fill the remaining team to best match the counterparts in the full team
        for client in team_1:
            self.ref_client = client
            client_set.remove(client)

            counterpart = min(client_set, key=self.diff_key)
            client_set.remove(counterpart)
            team_2.add(counterpart)

        # Check other combinations to find most balanced teams in terms of overall MMR
        player_set = team_1 | team_2
        best_combos = deque()
        lowest_mmr_diff = float('inf')

        for combo in combinations(player_set, self.N_PLAYERS_TO_MATCH//2):
            team_1 = set(combo)
            team_2 = player_set - team_1

            team_1_mmr = sum(client.mmr for client in team_1)
            team_2_mmr = sum(client.mmr for client in team_2)
            mmr_diff = abs(team_1_mmr - team_2_mmr)

            # If a combo is clearly better, override current best
            if mmr_diff < lowest_mmr_diff:
                best_combos.clear()
                best_combos.append((team_1, team_2))
                lowest_mmr_diff = mmr_diff

            # If combos are equivalent, random choice will be performed on them later
            elif mmr_diff == lowest_mmr_diff:
                best_combos.append((team_1, team_2))

        # Note that options XY and YX share their score and either can be chosen
        team_1, team_2 = random.choice(best_combos)

        # Check max intra-team differences (assume positive MMR)
        # This way, the reception may wait to allow more ideal matchings to turn up
        # or max allowed diff to expand enough to allow less ideal matchings to pass
        # Otherwise, if no intra-team MMR diff checking were done at the end of matching,
        # no artificial waiting would be done: the teams would be determined as
        # soon as a session handler was available and at least 10 client requests were received
        max_intra_diff = max(
            max(team_1, key=self.mmr_key).mmr - min(team_1, key=self.mmr_key).mmr,
            max(team_2, key=self.mmr_key).mmr - min(team_2, key=self.mmr_key).mmr)

        if self.expand_mmr_diff_limit and max_intra_diff > self.get_max_mmr_diff(timestamp, time_of_first_contact):
            return self.no_matched_clients

        # Assign starting team IDs
        for client in team_1:
            client.team = GameID.GROUP_TEAM_T

        for client in team_2:
            client.team = GameID.GROUP_TEAM_CT

        return team_1 | team_2

    @staticmethod
    def time_key(client: SDGClient):
        """Allow sorting clients by time of first contact."""

        return client.time_of_first_contact

    @staticmethod
    def mmr_key(client: SDGClient):
        """Allow sorting clients by MMR."""

        return client.mmr

    def diff_key(self, client: SDGClient):
        """Allow sorting clients by absolute difference in MMR to a reference client."""

        return abs(client.mmr - self.ref_client.mmr)

    def dist_key(self, client: SDGClient):
        """Allow sorting clients by a composite of differences in MMR and waiting time."""

        return (
            (client.mmr - self.ref_client.mmr)**2 +
            ((client.time_of_first_contact - self.ref_client.time_of_first_contact)*self.lwtime_to_mmr_scale)**2)**0.5

    @staticmethod
    def get_max_mmr_diff(timestamp: float, time_of_first_contact: float) -> float:
        """
        Determine MMR-based sampling range.

        Examples of gradual expansion:
        25 MMR at 0s waiting time.
        50 MMR at 5s waiting time.
        100 MMR at 10s waiting time.
        550 MMR at 30s waiting time.
        1975 MMR at 1min waiting time (long overdue, should let anything pass).
        """

        t_wait = timestamp - time_of_first_contact

        return 25. + 2.5*t_wait + 0.5*t_wait**2

    @staticmethod
    def run_session(address: Tuple[str, int], args: Namespace, matched_clients: Iterable[SDGClient]):
        """Run a new SDG session server that should automatically conclude."""

        # session_args.address = session_address[0]
        args.port = address[1]
        args.session_id = 'Session ' + str(args.port - args.main_port)

        # NOTE: Needs unique names per client, otherwise some players might be left as spectators,
        # because max team size (5) could be exceeded
        assigned_teams = {client.name: client.team for client in matched_clients}

        server = SDGServer(args, assigned_teams=assigned_teams)
        server.run()
