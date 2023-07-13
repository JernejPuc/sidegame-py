"""Matchmaking around session servers"""

import socket
import select
import multiprocessing as mp
from argparse import Namespace
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Set, Tuple
from time import perf_counter_ns, sleep

from sidegame.utils import get_logger


class Client(ABC):
    """
    A representation of a client awaiting redirection to the session server,
    with attributes to determine its priority.
    """

    def __init__(self, own_address: Tuple[str, int], timestamp: float):
        self.own_address = own_address
        self.time_of_first_contact = timestamp
        self.time_of_last_contact = timestamp
        self.time_of_last_ping = timestamp
        self.session_subport: int = None

    @abstractmethod
    def unpack_client_data(self, data: bytes):
        """
        Extract client data that is relevant on first connection,
        e.g. matchmaking rating (MMR), timestamp, or side preference.
        """

    @abstractmethod
    def pack_redirection_data(self, subport: int) -> bytes:
        """Pack session server address to inform the waiting client."""

    @abstractmethod
    def pack_confirmation_data(self, timestamp: float) -> bytes:
        """
        Pack data that is sent to the client to confirm that the
        matchmaker knows about it, but that it is yet to be redirected.
        """


class Matchmaker(ABC):
    """
    Accepts client (actor) requests and manages a pool of subprocesses
    (using standard multiprocessing and sockets) to spawn simulation sessions.
    These detached environments act as authoritative servers for
    multiple agents interacting in a shared environment (e.g. game matches).

    NOTE: The maximum number of subprocesses is determined by available session
    (sub)ports.
    """

    CLIENT_TIMEOUT = 10.
    MIN_PING_INTERVAL = 1.

    def __init__(
        self,
        session_args: Namespace,
        address: Tuple[str, int],
        subports: Iterable[int],
        client_message_size: int
    ):
        self.session_args = session_args
        self.address = address
        self.subports = set(subports)
        self.client_message_size = client_message_size

        self.reception_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.reception_socket.bind(address)
        self.reception_socket.settimeout(0.)

        self.readable_sockets: List[socket.socket] = [self.reception_socket]
        self.placeholder_sockets: List[socket.socket] = []

        self.clock_reference: float = perf_counter_ns()
        self.logger = get_logger(name='Matchmaker')

    def clock(self) -> float:
        """Get the time (in seconds) since the matchmaker was initialised."""

        return (perf_counter_ns() - self.clock_reference) * 1e-9

    def sendall(self, data: bytes, address: Tuple[str, int]):
        """Ensure that the entirety of the data is sent."""

        len_sent = 0

        while len_sent != len(data):
            len_sent += self.reception_socket.sendto(data[len_sent:] if len_sent != 0 else data, address)

    @abstractmethod
    def create_waiting_client(self, client_address: Tuple[str, int], timestamp: float) -> Client:
        """
        Produce an instance of a waiting client and prepare it for evaluation
        by the matchmaker.
        """

    @abstractmethod
    def eval_waiting_clients(self, clients: Dict[Tuple[str, int], Client], timestamp: float) -> Iterable[Client]:
        """
        Try to compose a new match. Any data that is intended to determine
        how `run_session` should handle the matched clients should be attributed
        to them before they are returned and passed on.
        """

    @staticmethod
    @abstractmethod
    def run_session(address: Tuple[str, int], args: Namespace, matched_clients: Iterable[Client]):
        """
        Start a new session at the given address with given parameters
        and expected clients.
        """

    def run(self):
        """
        Run the main loop of the matchmaker.

        Essentially, it 1) accepts client addresses, 2) waits until conditions
        for setting up a session have been cleared, e.g. enough players
        of similar MMR are present or a session process is available,
        3) spawns a new session bound to the same address, but a different port,
        and 4) sends the matched clients information to redirect them
        to the address/port of their session.

        The state of session subprocesses is regularly checked.
        When a session is found to be finished, its resources are freed
        and made available for further use.

        NOTE: There are no explicit means to stop ongoing sessions. The `Matchmaker`
        simply stops its main loop and waits for ongoing sessions to conclude.
        """

        ctx = mp.get_context('spawn')

        running_sessions: List[Tuple[mp.Process, int]] = []
        subports_in_use: Set[int] = set()

        clients: Dict[Tuple[str, int], Client] = {}
        waiting_clients: Dict[Tuple[str, int], Client] = {}

        self.logger.info('Running...')

        while True:
            try:
                current_clock = self.clock()

                readable, _, _ = select.select(
                    self.readable_sockets, self.placeholder_sockets, self.placeholder_sockets, 0.)

                # Handle client requests
                if readable:
                    try:
                        client_data, client_address = self.reception_socket.recvfrom(self.client_message_size)

                    # NOTE: Apparently, an UDP packet not reaching its destination
                    # can cause an error (on Windows, at least)
                    # See the docstring under `networking.core::ServerSocket.recv`
                    # In the case of the matchmaker, this is to be expected
                    # due to intended redirections
                    except ConnectionResetError:
                        continue

                    # Add new waiting client
                    if client_address not in clients:
                        client = self.create_waiting_client(current_clock, client_address)
                        clients[client_address] = client
                        waiting_clients[client_address] = client

                        self.logger.debug('New client connected. Waiting clients: %d.', len(waiting_clients))

                    else:
                        client = clients[client_address]

                    # Update waiting client
                    client.unpack_client_data(client_data)
                    client.time_of_last_contact = current_clock

                else:
                    sleep(1e-2)

                # Ping or remove clients
                for client_address, client in tuple(clients.items()):
                    if current_clock - client.time_of_last_contact > self.CLIENT_TIMEOUT:
                        del clients[client_address]

                    elif client.session_subport is None:
                        if current_clock - client.time_of_last_ping > self.MIN_PING_INTERVAL:
                            client.time_of_last_ping = current_clock
                            self.sendall(client.pack_confirmation_data(current_clock), client_address)

                    elif current_clock - client.time_of_last_ping > self.MIN_PING_INTERVAL:
                        client.time_of_last_ping = current_clock
                        self.sendall(client.pack_redirection_data(client.session_subport), client_address)

                # Try to compose a new match if max number of running sessions is not exceeded
                available_subports = self.subports - subports_in_use

                if available_subports:
                    matched_clients = self.eval_waiting_clients(waiting_clients, current_clock)

                    # Start session if clients were successfully matched
                    if matched_clients:
                        subport = available_subports.pop()
                        session_address = (self.address[0], subport)

                        session = ctx.Process(
                            target=self.run_session,
                            args=(session_address, self.session_args, matched_clients),
                            daemon=True)

                        session.start()
                        subports_in_use.add(subport)
                        running_sessions.append((session, subport))

                        # Add redirection data to matched clients and mark them as such
                        for client in matched_clients:
                            client.session_subport = subport
                            del waiting_clients[client.own_address]

                        self.logger.debug('Started new session. Running sessions: %d.', len(running_sessions))

                # Prune terminated sessions and release subports
                if any((not session.is_alive()) for session, _ in running_sessions):
                    s_idx = 0

                    while s_idx < len(running_sessions):
                        session, subport = running_sessions[s_idx]

                        if session.is_alive():
                            s_idx += 1

                        else:
                            session.join()
                            session.close()
                            subports_in_use.remove(subport)
                            del running_sessions[s_idx]

                            self.logger.debug('Updated free sessions: %d.', (len(self.subports)-len(running_sessions)))

            except KeyboardInterrupt:
                break

        self.logger.info('Waiting for sessions to finish...')

        # Exit when all spawned processes finish
        force_termination = False
        join_timeout = 1.

        for session, _ in running_sessions:
            while session.exitcode is None:
                if force_termination:
                    session.terminate()

                try:
                    session.join(join_timeout)

                except KeyboardInterrupt:
                    if not force_termination:
                        self.logger.info('Forcing shutdown...')

                    force_termination = True
                    join_timeout = None

            session.close()

        self.reception_socket.close()

        self.logger.info('Stopped.')
