"""
Predictive client in the predictive client-authoritative server model

References:
https://developer.valvesoftware.com/wiki/Source_Multiplayer_Networking
https://developer.valvesoftware.com/wiki/Latency_Compensating_Methods_in_Client/Server_In-game_Protocol_Design_and_Optimization
https://www.gabrielgambetta.com/client-server-game-architecture.html
"""

import logging
from collections import deque
from abc import ABC, abstractmethod
from typing import Any, Deque, Dict, Iterable, List, Tuple, Union
from time import perf_counter_ns

from sidegame.networking.core import Entry, Action, Entity, Recorder, ClientSocket
from sidegame.utils import MovingAverageTracker, TickLimiter, StridedFunction, get_logger


class ClientBase(ABC):
    """
    Defines the basic step of the main loop and the general pattern of its
    components, based on a record of known entities and how they are updated
    or interpolated at a certain point in time.

    Locally, the client works with its own timestamps, but converts them
    into the server's time before relaying them, so that the server can
    estimate its latency (the difference between the timestamps roughly
    corresponds to the round trip delay).

    This conversion is based on the difference between client and server clocks
    at a given tick (periodic time point of processing), synchronised in each
    available update to reflect the client's current latency.

    To mitigate network inconsistencies and tick (mis)alignment, both client
    and server employ moving average tracking with a constant offset,
    equal to adding or subtracting half of their tick interval, respectively
    (where the bias of the tracked average should land between ticks).
    """

    def __init__(
        self,
        id_: int,
        tick_rate: float,
        interp_window: float,
        init_clock_diff: float
    ):
        self.entities: Dict[int, Entity] = {}

        self.own_entity_id = id_
        self.own_entity = self.create_entity(self.own_entity_id)
        self.entities[self.own_entity_id] = self.own_entity

        self._interp_window = interp_window
        self._clock_diff_tracker = MovingAverageTracker(round(tick_rate))
        self._clock_diff_tracker.value = init_clock_diff
        self._clock_diff_tracker.offset = 1. / tick_rate / 2.

    @abstractmethod
    def run(self):
        """Run the main loop."""

    def step(self, dt_loop: float, local_clock: float):
        """Advance the local state (environment) by a single iteration."""

        # Check for and unpack server data
        server_data = self._get_server_data(local_clock)
        state_updates = self._get_state_updates(server_data) if server_data else None

        # Update local state wrt. state on the server
        self._eval_server_state(state_updates, local_clock)

        # Update local state wrt. user input
        user_input = self._get_user_input(local_clock)
        action = self._eval_input(user_input, local_clock) if user_input else None

        # Update foreign entities wrt. estimated server time
        self._interpolate_foreign_entities(local_clock + self._clock_diff_tracker.value - self._interp_window)

        # Record, send, and/or clean up
        self._relay(server_data, action, local_clock, self._clock_diff_tracker.value)

        # Produce new audio/video observation
        self.generate_output(dt_loop)

    @abstractmethod
    def _get_server_data(self, timestamp: float) -> Iterable[bytes]:
        """Receive or read server data."""

    @abstractmethod
    def _get_state_updates(self, data: Iterable[bytes]) -> Iterable[Entry]:
        """Unpack and interpret server data."""

    def _eval_server_state(self, state_updates: Union[Iterable[Entry], None], timestamp: float):
        """
        Evaluate the authoritative server state by updating the local state
        with partial updates, using counters for their validation.
        """

        if state_updates is None:
            return

        for state_entry in state_updates:
            # NOTE: Entries of TYPE_INIT are skipped

            if state_entry.type == Entry.TYPE_LOG and state_entry.counter == self.own_entity.global_log_counter + 1:
                self.own_entity.global_log_counter += 1
                self.handle_log(state_entry, timestamp)

            elif state_entry.type == Entry.TYPE_STATE:
                entity_id = state_entry.id

                # Add to local entities
                if entity_id not in self.entities:
                    self.entities[entity_id] = self.create_entity(entity_id)

                entity = self.entities[entity_id]

                # Validate state entry
                if state_entry.counter > entity.confirmed_action_counter:
                    entity.confirmed_action_counter = state_entry.counter

                    if entity_id == self.own_entity_id:
                        self.update_own_entity(state_entry, timestamp)
                        self._reconcile_state(entity.actions, state_entry.counter)

                        # Update server-client clock difference
                        self._clock_diff_tracker.update(state_entry.timestamp - timestamp)

                    else:
                        # Store data for entity interpolation
                        entity.states.append(state_entry)

    def _reconcile_state(self, actions: Deque[Action], last_confirmed_action_counter: int):
        """Remove past actions and re-predict the current state according to the rest."""

        idx = 0

        while idx < len(actions):
            action = actions[idx]

            if action.counter < last_confirmed_action_counter:
                del actions[idx]

            elif action.counter == last_confirmed_action_counter:
                idx += 1

            else:
                self.predict_state(action)
                idx += 1

    @abstractmethod
    def _get_user_input(self, timestamp: float) -> Union[Any, None]:
        """Poll or read user input data."""

    def _eval_input(self, user_input: Any, timestamp: float) -> Action:
        """Evaluate action corresponding to user input."""

        # Get time passed since previous action
        dt = (timestamp - self.own_entity.actions[-1].timestamp) if self.own_entity.actions else 0.

        self.own_entity.action_counter += 1
        action = Action(Action.TYPE_STATE, self.own_entity.action_counter, timestamp, user_input, dt=dt)

        # Update local state
        self.predict_state(action)

        # Add to action queue
        self.own_entity.actions.append(action)

        return action

    def _interpolate_foreign_entities(self, render_timestamp: float):
        """
        Interpolate locally known foreign entities, i.e. other than own entity,
        within the last interpolation window.
        """

        for entity in self.entities.values():
            if entity.id == self.own_entity_id:
                continue

            # Seek the pair of consecutive states between which the entity is to be interpolated,
            # removing those that have already been succeeded
            while len(entity.states) >= 2 and entity.states[1].timestamp < render_timestamp:
                entity.states.popleft()

            if len(entity.states) >= 2:
                state_0, state_1 = entity.states[0], entity.states[1]

                # Get state ratio wrt. render timestamp
                if state_0.timestamp <= render_timestamp <= state_1.timestamp:
                    state_ratio = (render_timestamp - state_0.timestamp) / (state_1.timestamp - state_0.timestamp)

                # If the states do not straddle the render timestamp, use the last one fully
                else:
                    state_ratio = 1.

                self.interpolate_foreign_entity(entity, state_ratio, state_0, state_1)

    @abstractmethod
    def _relay(
        self,
        server_data: Union[Iterable[bytes], None],
        action: Union[Action, None],
        timestamp: float,
        offset: float
    ):
        """Record, send, and/or clean up."""

    @abstractmethod
    def create_entity(self, entity_id: int) -> Entity:
        """
        Produce a local entity instance and prepare it
        for integration into the local state.
        """

    @abstractmethod
    def update_own_entity(self, state_entry: Entry, timestamp: float):
        """
        Synchronise own entity state with the state on the server,
        possibly resetting or correcting some of its predicted aspects.
        """

    @abstractmethod
    def handle_log(self, event_entry: Entry, timestamp: float):
        """Handle an unskippable, ordered event."""

    @abstractmethod
    def predict_state(self, action: Action):
        """
        Advance some aspects of the state that are liable to be reset or
        corrected by the server later on.

        NOTE: Some effects of prediction should only be triggered once,
        e.g. SFX/VFX, which can be done by checking `action.processed`.
        """

    @abstractmethod
    def interpolate_foreign_entity(self, entity: Entity, state_ratio: float, state1: Entry, state2: Entry):
        """Interpolate a foreign entity between two (past) states."""

    @abstractmethod
    def generate_output(self, dt: float):
        """Produce new audio/video observation."""


class LiveClient(ClientBase):
    """
    A client for live communication with the server based on the UDP socket.

    Some messages from either client or server should not be dropped or
    received out of order, e.g. events that affect the overall state of
    the simulation and not just individual entities. This is handled by
    exchanging local and global log request counters and attaching requested
    entries from local or global history records on each update.

    Frequency of client-server updates is determined by the client's own
    `sending_rate` and the server's `update_rate`.

    NOTE: For simplicity, `update_rate` was not made configurable per client,
    i.e. it is set by the server and is the same for all clients.

    NOTE: Frequency of data exchange affects the frequency of reconciliations.

    `update_rate`, in conjunction with `interp_ratio`, also determines the
    foreign entity interpolation window.

    NOTE: With `interp_ratio` of 1, the interpolation window would correspond
    to the last two updates. This brings the least of artifical latency,
    but can be most noticeably affected in case of packet loss.
    Using `interp_ratio` of 2, 1 consecutive packet drop can be gracefully
    handled, and so on, at the cost of inducing more artificial latency.

    NOTE: `interp_ratio` of 2 means that the render timestamp is set
    2 update (data receiving) intervals back from the current timestamp.
    With an `update_rate` of 30, the interpolation window will be about 70ms,
    about 35ms with 60 ticks, and so on.

    A `LiveClient` can also be used to record data for a `ReplayClient`:
    Storing all of the network data that it has exchanged with the server
    over the course of a session should allow complete reconstruction
    of the game experience from the client's perspective.
    """

    def __init__(
        self,
        tick_rate: float,
        polling_rate: float,
        sending_rate: float,
        server_address: Tuple[str, int],
        client_message_size: int,
        server_message_size: int,
        matchmaking: bool = False,
        recording_path: str = None,
        logging_path: str = None,
        logging_level: int = logging.DEBUG,
        show_fps: bool = True,
        interp_ratio: float = 2.
    ):
        self._clock_ref: int = perf_counter_ns()

        self._socket = ClientSocket(server_address, client_message_size, server_message_size)
        self._incoming_buffer: List[bytes] = self._socket.node.incoming_buffer
        self._outgoing_buffer: List[bytes] = self._socket.node.outgoing_buffer

        self._fps_limiter = TickLimiter(
            tick_rate,
            display_prefix='\rFPS: ',
            display_suffix='  ',
            level=(TickLimiter.LEVEL_DISPLAY if show_fps else TickLimiter.LEVEL_TRACK))

        self.logger: logging.Logger = get_logger(name='LiveClient', path=logging_path, level=logging_level)
        self.recorder = Recorder(file_path=recording_path)

        self._local_log_request_counter: int = 0
        self._local_logs: Dict[Action] = {}

        # Detach polling and sending from higher framerate
        self._strided_poll = StridedFunction(self.poll_user_input, stride=(tick_rate / polling_rate))
        self._strided_send = StridedFunction(self._send_client_data, stride=(tick_rate / sending_rate))

        # Establish connection (possibly through matchmaking server)
        request = self.get_connection_request(interp_ratio)

        if matchmaking:
            self._redirect_through_matchmaking(request)

        reply = self._connect_to_session(request)

        # Record and unpack connection data
        self.recorder.append(request, source=Recorder.SOURCE_CLIENT)
        self.recorder.append(reply, source=Recorder.SOURCE_SERVER)

        client_id, update_rate, server_clock = self.unpack_connection_reply(reply)

        interp_window = interp_ratio / update_rate
        init_clock_diff = server_clock - self._clock()

        super().__init__(client_id, tick_rate, interp_window, init_clock_diff)

        self.session_running: bool = None
        self.logger.debug('Connected to server.')

    def _log_error(self, msg: str):
        if self.logger.getEffectiveLevel() == logging.DEBUG:
            self.logger.error(msg, exc_info=True)

        else:
            self.logger.error(msg)

    def _clock(self) -> float:
        return (perf_counter_ns() - self._clock_ref) * 1e-9

    def _redirect_through_matchmaking(self, request: bytes):
        """
        Exchange pings with the matchmaking server until getting redirected
        to the session server address.
        """

        while True:
            try:
                reply = self._socket.exchange(request, self._clock())

            except (AssertionError, ConnectionError, ConnectionRefusedError, TimeoutError):
                self._log_error('Could not connect to the matchmaking server.')
                self._socket.close()
                raise SystemExit

            redirection_address = self.unpack_redirection_address(reply)

            if redirection_address is not None:
                self.logger.debug('Redirecting to the session server...')
                self._socket.redirect(redirection_address)

                # Give the server some time to establish itself, often got ConnectionResetError 10054 otherwise
                self._fps_limiter.delay(1.)
                break

            else:
                self._fps_limiter.delay(1.)

    def _connect_to_session(self, request: bytes) -> bytes:
        """Exchange initialisation data with the session server."""

        try:
            reply = self._socket.exchange(request, self._clock())

        except (AssertionError, ConnectionError, ConnectionRefusedError, TimeoutError):
            self._log_error('Could not connect to the session server.')
            self._socket.close()
            raise SystemExit

        return reply

    @abstractmethod
    def get_connection_request(self, interp_ratio: float) -> bytes:
        """
        Create the first (initialisation) message to establish contact
        with the server. It should include information for routing through
        matchmaking, as well.
        """

    @abstractmethod
    def unpack_redirection_address(self, message: bytes) -> Union[Tuple[str, int], None]:
        """
        Confirm and unpack redirection address from data sent by
        the matchmaking server.
        """

    @abstractmethod
    def unpack_connection_reply(self, message: bytes) -> Tuple[int, float, float]:
        """
        Interpret the first (initialisation) message sent back by the server.
        It should return the `client_id`, `update_rate`,
        and current `server_clock`.
        """

    def run(self) -> int:
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

                # Update recorder counter and timestamp
                self.recorder.update_meta(current_clock)

                # Advance local state
                self.step(dt_loop, current_clock)

                # Cache and squeeze records
                self.recorder.cache_chunks()
                self.recorder.squeeze()

                # Delay to target specified FPS
                self._fps_limiter.update_and_delay(self._clock() - current_clock, current_clock)

        except KeyboardInterrupt:
            self.logger.debug('Process ended by user.')

        except (ConnectionError, TimeoutError):
            self._log_error('Lost connection to the server.')

        else:
            self.logger.debug('Session ended.')

            # Explicitly send any final messages still in queue due to sending stride
            if current_clock is not None:
                self._send_client_data(current_clock)

                # Include slight delay to allow them to reach the server before disconnecting
                self._fps_limiter.delay(0.5)

        # Saving and cleanup
        self._socket.close()

        if self.recorder.file_path is not None:
            self.recorder.restore_chunks()
            self.recorder.squeeze(all_=True)
            self.recorder.save()

            self.logger.info("Recording saved to: '%s'.", self.recorder.file_path)

        self.cleanup()

        self.logger.info('Stopped.')

        return 0

    def _send_client_data(self, timestamp: float):
        """Send buffered data to server."""

        if self._outgoing_buffer:
            self._socket.send(timestamp=timestamp)

    def _get_server_data(self, timestamp: float) -> Iterable[bytes]:
        self._socket.recv(timestamp=timestamp)
        return self._incoming_buffer

    def _get_state_updates(self, data: Iterable[bytes]) -> Iterable[Entry]:
        server_state, self._local_log_request_counter = self.unpack_server_data(data)
        return server_state

    def _get_user_input(self, timestamp: float) -> Union[Any, None]:
        polled_input = self._strided_poll(timestamp)

        if polled_input is None:
            return None

        user_input, user_log = polled_input

        # Add to local log history
        if user_log is not None:
            self.own_entity.local_log_counter += 1

            log = Action(Action.TYPE_LOG, self.own_entity.local_log_counter, timestamp, user_log)
            self._local_logs[self.own_entity.local_log_counter] = log

        return user_input

    def _relay(
        self,
        server_data: Union[Iterable[bytes], None],
        action: Union[Action, None],
        timestamp: float,
        offset: float
    ):
        # Pack incoming data
        if self._incoming_buffer:
            self.recorder.extend(self._incoming_buffer, source=Recorder.SOURCE_SERVER)
            self._incoming_buffer.clear()

        # Pack logs
        if self._local_log_request_counter <= self.own_entity.local_log_counter:
            for request_counter in range(self._local_log_request_counter, self.own_entity.local_log_counter+1):
                requested_log = self._local_logs[request_counter]

                # Convert to server time
                local_timestamp = requested_log.timestamp
                requested_log.timestamp = local_timestamp + offset

                packed_log = self.pack_input_data(requested_log, self.own_entity.global_log_counter+1)

                # Convert back to local time
                requested_log.timestamp = local_timestamp

                self.recorder.append(packed_log, source=Recorder.SOURCE_CLIENT)
                self._outgoing_buffer.append(packed_log)

        # Pack action
        if action is not None:
            # Convert to server time
            local_timestamp = action.timestamp
            action.timestamp = local_timestamp + offset

            packed_action = self.pack_input_data(action, self.own_entity.global_log_counter+1)

            # Convert back to local time
            action.timestamp = local_timestamp

            self.recorder.append(packed_action, source=Recorder.SOURCE_CLIENT)
            self._outgoing_buffer.append(packed_action)

        # Send on condition
        self._strided_send(timestamp)

    @abstractmethod
    def unpack_server_data(self, data: Iterable[bytes]) -> Tuple[Iterable[Entry], int]:
        """Unpack state updates and local log request counter."""

    @abstractmethod
    def pack_input_data(self, action: Action, global_log_request_counter: int) -> bytes:
        """Pack user action and global log request counter."""

    @abstractmethod
    def poll_user_input(self, timestamp: float) -> Tuple[Any, Union[Any, None]]:
        """
        Poll or read peripheral events and interpret them
        as user input and optional local log data.
        """

    @abstractmethod
    def cleanup(self):
        """Gracefully close initialised subsystems."""


class ReplayClient(ClientBase):
    """
    A client which, instead of interacting with a server over the network,
    reconstructs the game experience from recorded data,
    i.e. all of the network data that a `LiveClient` has exchanged
    with the server over the course of a session.

    These recordings can be played (stepped) automatically or manually,
    which should allow game state data to be extracted and processed
    regardless of pace.

    Reference:
    https://docs.unrealengine.com/en-US/TestingAndOptimization/ReplaySystem/index.html
    """

    CMD_NONE = 0
    CMD_PAUSE = 1
    CMD_EXIT = 2

    def __init__(
        self,
        original_tick_rate: float,
        recording_path: str,
        logging_path: str = None,
        logging_level: int = logging.DEBUG,
        show_fps: bool = True
    ):
        self._clock_ref: int = perf_counter_ns()

        self._original_tick_rate = original_tick_rate
        self._tick_counter = 0
        self._tick_timestamp = 0.
        self._queued_state_updates: Deque[Entry] = deque()
        self._queued_actions: Deque[Action] = deque()

        self._fps_limiter = TickLimiter(
            original_tick_rate,
            display_prefix='\rFPS: ',
            display_suffix='  ',
            level=(TickLimiter.LEVEL_DISPLAY if show_fps else TickLimiter.LEVEL_TRACK))

        self.logger: logging.Logger = get_logger(name='ReplayClient', path=logging_path, level=logging_level)
        self.recorder = Recorder(file_path=recording_path)

        # Read recording
        self.recorder.read()
        self.recording = self.recorder.buffer

        # Extract initialisation data
        clk = self._get_next_batch()
        assert clk == 0., f'Expected initialisation batch at local time 0., got {clk}.'

        request = self._queued_actions.pop()
        reply = self._queued_state_updates.pop()

        client_id, interp_window, init_clock_diff = self._unpack_connection_exchange(request, reply)

        super().__init__(client_id, original_tick_rate, interp_window, init_clock_diff)

        self.logger.debug('Replay data loaded.')

    def _clock(self) -> float:
        return (perf_counter_ns() - self._clock_ref) * 1e-9

    def _get_next_batch(self) -> Union[float, None]:
        """Fill the update and action queues with data corresponding to the next tick."""

        clk = None
        self._queued_actions.clear()
        self._queued_state_updates.clear()

        while self.recording:
            (clk_, ctr, src), data = self.recorder.split_meta(self.recording[0])

            if ctr == self._tick_counter:
                if src == Recorder.SOURCE_CLIENT:
                    self._queued_actions.append(data)

                elif src == Recorder.SOURCE_SERVER:
                    self._queued_state_updates.append(data)

                clk = clk_
                self.recording.popleft()

            else:
                break

        self._tick_counter += 1
        return clk

    @abstractmethod
    def _unpack_connection_exchange(self, request: bytes, reply: bytes) -> Tuple[int, float, float]:
        """
        Get initialisation data from the initial exchange between
        a live client and server, i.e. `client_id`, `interp window`,
        and `init_clock_diff`.
        """

    def run(self) -> int:
        self.logger.info('Running...')

        previous_clock: float = None

        try:
            while self.recording:
                ref_clock = self._clock()

                # Apply external control
                if previous_clock is not None:
                    cmd, jumped_previous_clock = self.get_user_command(previous_clock)

                    if jumped_previous_clock is not None:
                        previous_clock = jumped_previous_clock

                else:
                    cmd = self.CMD_NONE

                # Exit or pause
                if cmd == self.CMD_EXIT:
                    raise KeyboardInterrupt

                elif cmd == self.CMD_PAUSE:
                    self._fps_limiter.delay(2e-1)

                else:
                    # Update loop timekeeping
                    current_clock = self._get_next_batch()

                    # Infer timestamp if tick has no update/polling data
                    if current_clock is None:
                        if previous_clock is None:
                            continue

                        else:
                            current_clock = previous_clock + 1. / self._original_tick_rate

                    dt_loop = (current_clock - previous_clock) if previous_clock is not None else 0.
                    previous_clock = current_clock

                    # Advance local state
                    self.step(dt_loop, current_clock)

                    # Imitate original (in)ability to keep up with target FPS
                    remaining_dt = max(0., dt_loop - (self._clock() - ref_clock) - 5e-4)
                    remaining_dt *= self._original_tick_rate / self._fps_limiter.tick_rate

                    self._fps_limiter.update(ref_clock)
                    self._fps_limiter.delay(remaining_dt)

        except KeyboardInterrupt:
            self.logger.debug('Process ended by user.')

        else:
            self.logger.debug('End of recording reached.')

        # Cleanup
        self.cleanup()

        self.logger.info('Stopped.')

        return 0

    def change_replay_speed(self, speedup: float = 1.):
        """Change the replay speed by the specified factor."""

        self._fps_limiter.set_tick_rate(self._original_tick_rate * speedup)

    def jump_to_timestamp(self, current_timestamp: float, jump_timestamp: float) -> Union[float, None]:
        """
        Jump forward or backward in time to the specified timestamp.

        NOTE: Jumping to past timestamps resimulates everything from the beginning.
        A possible optimisation would be to cache reference points with some
        interval, but this would probably be less general and possibly unnecessary.
        See: https://blog.counter-strike.net/index.php/2020/10/31790/
        """

        # Reinitialise
        if jump_timestamp < current_timestamp:
            self._reinit()
            self.reinit()

        else:
            self.pause_effects()

        # Step until reaching jump timestamp
        previous_clock: float = None

        while self.recording:
            current_clock = self._get_next_batch()

            # Infer timestamp if tick has no update/polling data
            if current_clock is None:
                if previous_clock is None:
                    continue

                else:
                    current_clock = previous_clock + 1. / self._original_tick_rate

            previous_clock = current_clock

            # Advance local state
            self._shortstep(current_clock)

            if current_clock >= jump_timestamp:
                break

        # Re-enable effects
        self.resume_effects()

        return previous_clock

    def _shortstep(self, local_clock: float):
        """Advance local state without relaying or generating observations."""

        # Check for and unpack server data
        server_data = self._get_server_data(local_clock)
        state_updates = self._get_state_updates(server_data) if server_data else None

        # Update local state wrt. state on the server
        self._eval_server_state(state_updates, local_clock)

        # Update local state wrt. user input
        user_input = self._get_user_input(local_clock)

        if user_input:
            self._eval_input(user_input, local_clock)

        # Update foreign entities wrt. estimated server time
        self._interpolate_foreign_entities(local_clock + self._clock_diff_tracker.value - self._interp_window)

    def _get_server_data(self, timestamp: float) -> Iterable[bytes]:
        return self._queued_state_updates

    def _get_state_updates(self, data: Iterable[bytes]) -> Iterable[Entry]:
        return LiveClient._get_state_updates(self, data)

    def _get_user_input(self, timestamp: float) -> Union[Any, None]:
        while self._queued_actions:
            action, _ = self.unpack_client_data(self._queued_actions.popleft())

            if action.type == Action.TYPE_STATE:
                return action.data

        return None

    def _relay(
        self,
        server_data: Union[Iterable[bytes], None],
        action: Union[Action, None],
        timestamp: float,
        offset: float
    ):
        pass

    @abstractmethod
    def unpack_server_data(self, data: Iterable[bytes]) -> Tuple[Iterable[Entry], int]:
        """Unpack state updates and local log request counter."""

    @abstractmethod
    def unpack_client_data(self, data: bytes) -> Tuple[Action, int]:
        """Unpack client data into an action and the requested global log counter."""

    def _reinit(self):
        """Reinitialise internal attributes."""

        self._tick_counter = 0
        self._tick_timestamp = 0.
        self._queued_state_updates.clear()
        self._queued_actions.clear()

        # Read recording
        self.recording.clear()
        self.recorder.read()

        # Extract initialisation data
        clk = self._get_next_batch()
        assert clk == 0., f'Expected initialisation batch at local time 0., got {clk}.'

        request = self._queued_actions.popleft()
        reply = self._queued_state_updates.popleft()

        _, _, init_clock_diff = self._unpack_connection_exchange(request, reply)

        self.entities.clear()
        self.own_entity = self.create_entity(self.own_entity_id)
        self.entities[self.own_entity_id] = self.own_entity

        self._clock_diff_tracker.reset()
        self._clock_diff_tracker.value = init_clock_diff

    @abstractmethod
    def reinit(self):
        """Reinitialise local state."""

    @abstractmethod
    def pause_effects(self):
        """(Temporarily) disable real-time effects."""

    @abstractmethod
    def resume_effects(self):
        """Re-enable real-time effects."""

    @abstractmethod
    def get_user_command(self, current_timestamp: float) -> Tuple[int, Union[float, None]]:
        """
        Get command for live replay control, i.e. pause, speed change, or jump.
        In the latter case, the return value of `jump_to_timestamp` should be
        propagated with the command.
        """

    @abstractmethod
    def cleanup(self):
        """Gracefully close initialised subsystems."""
