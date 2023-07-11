"""
Authoritative server in the predictive client-authoritative server model

References:
https://developer.valvesoftware.com/wiki/Source_Multiplayer_Networking
https://developer.valvesoftware.com/wiki/Latency_Compensating_Methods_in_Client/Server_In-game_Protocol_Design_and_Optimization
https://www.gabrielgambetta.com/client-server-game-architecture.html
"""

import logging
from collections import deque
from abc import ABC, abstractmethod
from typing import Any, Callable, Deque, Dict, Iterable, List, Tuple
from time import perf_counter
from sidegame.networking.core import Entry, Action, EventBase, Entity, Node, TickLimiter, StridedFunction, \
    ServerSocket, get_logger


class Server(ABC):
    """
    A single authority for the state of a simulation with multiple interacting
    agents and the central node in a network, based on the UDP socket,
    to which all clients must connect to.

    Some messages from either client or server should not be dropped or
    received out of order, e.g. events that affect the overall state of
    the simulation and not just individual entities. This is handled by
    exchanging local and global log request counters and attaching requested
    entries from local or global history records on each update.
    """

    MAX_N_LOGS_PER_UPDATE = 10

    def __init__(
        self,
        tick_rate: float,
        update_rate: float,
        socket_address: Tuple[str, int],
        client_message_size: int,
        server_message_size: int,
        logging_name: str = None,
        logging_path: str = None,
        logging_level: int = logging.DEBUG,
        show_ticks: bool = True
    ):
        self.entities: Dict[int, Entity] = {}

        self._socket = ServerSocket(socket_address, client_message_size, server_message_size, round(tick_rate))
        self._clients: Dict[Tuple[str, int], Node] = self._socket.nodes

        self._clock: Callable = perf_counter

        self._tick_limiter = TickLimiter(
            tick_rate,
            display_prefix='\rTICKS: ',
            display_suffix='  ',
            level=(TickLimiter.LEVEL_DISPLAY if show_ticks else TickLimiter.LEVEL_TRACK))

        self.logger: logging.Logger = get_logger(
            name=('Server' if logging_name is None else logging_name),
            path=logging_path,
            level=logging_level)

        self._global_log_counter = -1
        self._global_logs: Dict[Entry] = {}
        self._queued_local_logs: Deque[Action] = deque()
        self._queued_events: Deque[EventBase] = deque()

        self._tick_interval = 1. / tick_rate
        self._update_rate = update_rate

        # Detach sending updates from tick rate
        self._strided_send = StridedFunction(self._collect_and_send_data, stride=(tick_rate / update_rate))

        self.session_running: bool = None
        self.logger.debug('Accepting connections.')

    def run(self) -> int:
        """
        Run the main loop.

        NOTE: On end of session, `session_running` should be set (to `False`)
        externally.
        """

        self.logger.info('Running...')
        self.session_running = True

        previous_clock: float = None
        current_clock: float = None

        try:
            while self.session_running:
                # Update loop timekeeping
                current_clock = self._clock()
                dt_loop = (current_clock - previous_clock) if previous_clock is not None else 0.
                previous_clock = current_clock

                # Advance authoritative state
                self._step(dt_loop, current_clock)

                # Delay to target specified FPS
                self._tick_limiter.update_and_delay(self._clock() - current_clock, current_clock)

        except KeyboardInterrupt:
            self.logger.debug('Process ended by user.')

        else:
            self.logger.debug('Session ended.')

            # Explicitly send any final messages still in queues due to sending stride
            if current_clock is not None:
                self._collect_and_send_data(current_clock)

                # Include slight delay to allow them to reach clients before disconnecting
                self._tick_limiter.delay(0.5)

        # Cleanup
        self._socket.close()

        self.logger.info('Stopped.')

        return 0

    def _step(self, dt_loop: float, current_clock: float):
        """
        Advance the authoritative state (environment) by a single iteration.

        To mitigate network inconsistencies and tick (mis)alignment in tracking
        clients' latencies, the server employs moving average tracking with
        a constant offset, equal to subtracting half of its tick interval
        (where the bias of the tracked average should land between ticks).
        """

        # Check for client data
        new_clients = self._socket.recv(current_clock)

        # Add new clients
        if new_clients:
            for client in new_clients:
                self.entities[client.id] = self.create_entity(client.id)

                # Add interpolation window to client node and offset its latency tracker
                client_interp_ratio = self.unpack_connection_request(client.incoming_buffer[0], client.id)
                client.interp_window = client_interp_ratio / (self._update_rate - 1.)
                client.tracker.offset = -self._tick_interval / 2.

                initial_data = self.get_connection_reply(client.id, self._update_rate, current_clock)
                self._socket.sendall(initial_data, client.address)

                self.logger.info('Client %d connected.', client.id)

        # Unpack and evaluate client actions
        for client in self._clients.values():
            if client.incoming_buffer:
                entity = self.entities.get(client.id, None)

                # Ignore clients that were detached from entities
                if entity is None:
                    client.incoming_buffer.clear()
                    continue

                client_actions, entity.global_log_counter = self.unpack_client_data(client.incoming_buffer)
                client.incoming_buffer.clear()

                for client_action in client_actions:
                    # Update client's latency
                    client.tracker.update(current_clock - client_action.timestamp)

                    # Validate and apply client action
                    self._eval_action(entity, client_action, current_clock, client.tracker.value, client.interp_window)

        # Regular (client-independent) update
        new_global_logs = self.update_state(dt_loop, current_clock, self._queued_local_logs, self._queued_events)
        self._queued_local_logs.clear()
        self._queued_events.clear()

        # Update global log history
        for log_id, log_data in new_global_logs:
            self._global_log_counter += 1

            log = Entry(log_id, Entry.TYPE_LOG, self._global_log_counter, current_clock, log_data)
            self._global_logs[self._global_log_counter] = log

        # Send data to clients
        ids_to_remove = self._strided_send(current_clock)

        # Handle client disconnections
        if ids_to_remove:
            new_global_logs = self.handle_missing_entities(ids_to_remove)

            for log_id, log_data in new_global_logs:
                self._global_log_counter += 1

                log = Entry(log_id, Entry.TYPE_LOG, self._global_log_counter, current_clock, log_data)
                self._global_logs[self._global_log_counter] = log

            self.logger.debug(('Connection(s) ' + '%d, '*len(ids_to_remove) + 'timed out.'), *ids_to_remove)

    def _eval_action(self, entity: Entity, action: Action, timestamp: float, lag: float, lerp: float):
        """Validate and apply client action to the authoritative state."""

        # NOTE: Actions of TYPE_INIT are skipped

        if action.type == Action.TYPE_LOG and action.counter == entity.local_log_counter + 1:
            self._queued_local_logs.append(action)
            entity.local_log_counter += 1

        elif action.type == Action.TYPE_STATE and action.counter > entity.action_counter:
            self._queued_events.extend(self.apply_action(entity, action, timestamp, lag=lag, lerp=lerp))
            entity.action_counter = action.counter

    def _collect_and_send_data(self, timestamp: float) -> List[int]:
        """
        Determine state updates to be sent and send them per client.

        NOTE: For simplicity, `update_rate` was not made configurable per client,
        i.e. it is set by the server and is the same for all clients.
        """

        for client in self._clients.values():
            entity = self.entities[client.id]

            # Pack logs (up to a maximum from the requested index)
            if entity.global_log_counter <= self._global_log_counter:
                requested_logs = (
                    self._global_logs[req_ctr]
                    for req_ctr in range(
                        entity.global_log_counter,
                        min(self._global_log_counter+1, entity.global_log_counter+1+self.MAX_N_LOGS_PER_UPDATE)))

                packed_logs = self.pack_state_data(requested_logs, entity.local_log_counter+1)
                client.outgoing_buffer.extend(packed_logs)

            # Pack state data
            state_data = self.pack_state_data(self.gather_state_data(entity), entity.local_log_counter+1)
            client.outgoing_buffer.extend(state_data)

        return self._socket.send(timestamp)

    @abstractmethod
    def unpack_connection_request(self, data: bytes, client_id: int) -> float:
        """Extract `client_interp_ratio` from the initial request of client `client_id`."""

    @abstractmethod
    def get_connection_reply(self, client_id: int, update_rate: float, current_clock: float) -> bytes:
        """Create the first message to be sent to a connecting client."""

    @abstractmethod
    def create_entity(self, entity_id: int) -> Entity:
        """
        Produce an entity instance and prepare it
        for integration into the authoritative state.
        """

    @abstractmethod
    def handle_missing_entities(self, missing_entity_ids: List[int]) -> Iterable[Tuple[int, Any]]:
        """
        Remove entities with disconnected clients from local representations.

        NOTE: Automatically clearing all data and objects associated with
        disconnected clients might not always be intended, e.g. if clients
        are somehow allowed to reconnect.
        """

    @abstractmethod
    def unpack_client_data(self, data: Iterable[bytes]) -> Tuple[Iterable[Action], int]:
        """
        Unpack and possibly reduce or squash packets of client data
        into fewer actions and extract the requested global log counter.
        """

    @abstractmethod
    def pack_state_data(self, state: Iterable[Entry], local_log_request_counter: int) -> Iterable[bytes]:
        """Pack state updates and local log request counter."""

    @abstractmethod
    def gather_state_data(self, entity: Entity) -> Iterable[Entry]:
        """Gather state data which is exposed to the gathering `entity`."""

    @abstractmethod
    def apply_action(
        self,
        entity: Entity,
        action: Action,
        timestamp: float,
        lag: float,
        lerp: float
    ) -> Iterable[Any]:
        """
        Update the authoritative state according to an input from a client,
        with optional lag compensation.
        """

    @abstractmethod
    def update_state(
        self,
        dt: float,
        timestamp: float,
        queued_logs: Deque[Action],
        queued_events: Deque[EventBase]
    ) -> Iterable[Tuple[int, Any]]:
        """
        Consolidate the authoritative state wrt. logs and events produced
        by clients beforehand, returning any logs that are produced in turn.
        """
