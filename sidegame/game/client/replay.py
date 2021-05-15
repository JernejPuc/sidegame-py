"""Interactive demo player for SDG"""

import os
import ctypes
import struct
import random
from argparse import Namespace
from typing import Callable, Deque, Iterable, Tuple, Union
from threading import Lock

import numpy as np
import cv2
import sdl2
import sdl2.ext

from sidegame.networking import Entry, Action, StridedFunction, ReplayClient
from sidegame.game.shared import GameID, Session
from sidegame.game.client.interface import SDGLiveClient
from sidegame.game.client.simulation import Simulation
from sidegame.game.client.tracking import DATA_DIR, StatTracker


class SDGReplayClient(ReplayClient):
    """
    A client which, instead of interacting with a SDG server over the network,
    reconstructs the game experience from recorded data,
    i.e. all of the network data that a SDG live client has exchanged
    with the server over the course of a session.

    It takes most of its methods directly from `SDGLiveClient`.

    Demos can be recorded by playing and spectating clients alike.
    When replaying recordings of the latter, the observed player and view
    can be controlled as if spectating live (because then all entities
    are interpolated and not dependant on client-side prediction).
    Otherwise, the recording player's experience is strictly followed.

    NOTE: For best real-time reproduction, the machine replaying at the
    specified render scale should be as capable as the recording machine
    at the original scale.

    NOTE: When no data is available for a certain tick (if polling rate was
    lower than tick rate), the missing loop time is assumed to be the ideal
    tick interval. Although this is a minor inconsistency, for accurate
    reproduction, the effective polling rate should match the tick rate.
    """

    ENV_ID = SDGLiveClient.ENV_ID

    SPEEDUPS = [0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2., 4., 8.]

    def __init__(self, args: Namespace, headless: bool = False):
        self.rng = np.random.default_rng(args.seed)
        random.seed(args.seed)

        # Set original tick rate
        args.tick_rate = float(args.recording_path.split('/')[-1].split('_')[-2].split('-')[-1])

        super().__init__(
            args.tick_rate,
            args.recording_path,
            args.logging_path,
            args.logging_level,
            args.show_fps)

        self.init_args = args
        self.last_lclick_status = 0
        self.last_space_status = 0
        self.space_time = 0.
        self.speedup_idx = 3
        self.paused = False

        self.sim = Simulation(args.tick_rate, args.volume, self.own_entity_id, rng=self.rng)
        self.session: Session = self.sim.session
        self.stats = StatTracker(self.session, self.own_entity)

        # Expose observations for external access
        self.headless = headless

        self.last_frame: np.ndarray = None
        self.audio_stream: Deque[np.ndarray] = self.sim.audio_system.external_buffer
        self.io_lock: Lock = self.sim.audio_system.external_buffer_io_lock

        self.window_size = (
                round(SDGLiveClient.RENDER_SIZE[0]*args.render_scale),
                round(SDGLiveClient.RENDER_SIZE[1]*args.render_scale))

        if headless:
            self.window = None
            self.window_array = None
            self.mouse_sensitivity = None
            self.strided_refresh = None

        else:
            sdl2.ext.init()
            self.sim.audio_system.start()

            self.window = sdl2.ext.Window(SDGLiveClient.WINDOW_NAME, size=self.window_size)
            self.window.show()
            self.window_array = sdl2.ext.pixels3d(sdl2.SDL_GetWindowSurface(self.window.window).contents)

            self.mouse_sensitivity = args.mouse_sensitivity / args.render_scale

            # Decoupled refresh allows the game state to be processed at higher framerate than e.g. 60Hz monitor limit
            # without the cost of actually rendering a frame (including fast-forwarding)
            self.strided_refresh: Callable = StridedFunction(
                self.refresh_display, self._fps_limiter.tick_rate / args.refresh_rate)

    def create_entity(self, entity_id):
        return SDGLiveClient.create_entity(self, entity_id)

    def update_own_entity(self, state_entry, timestamp):
        return SDGLiveClient.update_own_entity(self, state_entry, timestamp)

    def handle_log(self, event_entry, timestamp):
        return SDGLiveClient.handle_log(self, event_entry, timestamp)

    def remove_object_entities(self):
        """Remove client-side-only (object) entities from general entities."""
        return SDGLiveClient.remove_object_entities(self)

    def predict_state(self, action):
        return SDGLiveClient.predict_state(self, action)

    def interpolate_foreign_entity(self, entity, state_ratio, state1, state2):
        return SDGLiveClient.interpolate_foreign_entity(self, entity, state_ratio, state1, state2)

    def refresh_display(self):
        """Produce a new frame, upscale it, and update the image on screen."""

        frame = self.sim.get_frame()

        # Update externally accessible image and audio buffer
        if self.headless:
            with self.io_lock:
                self.last_frame = frame if self.init_args.render_scale == 1 else \
                    cv2.resize(frame, self.window_size, interpolation=cv2.INTER_NEAREST)

            self.sim.audio_system.step()

        # Upscale directly to window array memory and redraw
        else:
            frame = np.concatenate((frame, SDGLiveClient.ALPHA), axis=-1)
            cv2.resize(frame, self.window_size, dst=self.window_array.base, interpolation=cv2.INTER_NEAREST)
            self.window.refresh()

    def generate_output(self, dt: float):
        self.sim.eval_effects(dt)

        if self.headless:
            self.refresh_display()
        else:
            self.strided_refresh()

    def _unpack_connection_exchange(self, request: bytes, reply: bytes) -> Tuple[int, float, float]:
        req_data = struct.unpack('>HBLf13B2hf', request)
        rep_data = struct.unpack('>HhBLf12f3B', reply)

        interp_ratio = req_data[-1]
        client_id = rep_data[1]
        update_rate = rep_data[5]
        init_clock_diff = rep_data[6]

        return client_id, interp_ratio / (update_rate - 1.), init_clock_diff

    @staticmethod
    def unpack_single(data: bytes) -> Tuple[Entry, int]:
        """Unpack a single incoming data packet into a state entry and a log request counter."""
        return SDGLiveClient.unpack_single(data)

    def unpack_server_data(self, data: Iterable[bytes]) -> Tuple[Iterable[Entry], int]:
        return SDGLiveClient.unpack_server_data(self, data)

    def unpack_client_data(self, data: bytes) -> Tuple[Action, int]:
        """
        Unpack client data into an action and the requested global log counter.

        Because the live client handles some things within its interface,
        i.e. during `poll_user_input`, to recreate them, the packet must
        include all data necessary to infer the effects.

        `unpack_client_data` is then called in place of `poll_user_input`.

        NOTE: Actions are recorded when produced, not when they are sent to the
        server, so different tick and sending rates should not pose an issue.
        """

        packet = struct.unpack('>HBLf13B2hf', data)

        global_log_request_counter, action_type, counter, timestamp = packet[:4]
        action_data = packet[4:]

        if action_type == Action.TYPE_STATE:
            lclick_status = action_data[0]
            space_status = action_data[2]
            cursor_x = action_data[7]
            cursor_y = action_data[8]
            wheel_y = action_data[11] - 1
            view = action_data[12]
            hovered_entity_id = action_data[14]

            if not self.session.is_spectator(self.sim.own_player_id):
                # With the exception of `VIEW_LOBBY`, other views should be strictly followed on player replays
                if view != GameID.VIEW_LOBBY:
                    if self.sim.view != GameID.VIEW_WORLD and view == GameID.VIEW_WORLD:
                        if self.sim.view == GameID.VIEW_TERMS:
                            self.sim.create_log(GameID.EVAL_MSG_TERM)

                        elif self.sim.view == GameID.VIEW_ITEMS:
                            self.sim.create_log(GameID.EVAL_MSG_ITEM)

                        elif self.sim.view == GameID.VIEW_STORE:
                            self.sim.create_log(GameID.EVAL_BUY)

                    self.sim.view = view

                    if view == GameID.VIEW_MAPSTATS and self.last_lclick_status and not lclick_status:
                        self.sim.create_log(GameID.EVAL_MSG_MARK)

                # If dead, follow the originally observed player
                if self.session.is_dead_player(self.sim.own_player_id):
                    if view != GameID.VIEW_LOBBY and self.sim.view == GameID.VIEW_WORLD:
                        self.sim.observed_player_id = hovered_entity_id

                # When returning to life, switch back to own player
                elif self.sim.observed_player_id != self.sim.own_player_id:
                    self.sim.observed_player_id = self.sim.own_player_id

                # Follow cursor
                self.sim.cursor_x = cursor_x
                self.sim.cursor_y = cursor_y

                # Follow chat scroll
                if wheel_y < 0:
                    self.sim.wheel_y = max(wheel_y-1, 0)
                elif wheel_y > 0:
                    self.sim.wheel_y = min(wheel_y+1, max(len(self.sim.chat)-5, 0))

                # Follow send/clear
                if self.last_space_status and not space_status:
                    self.sim.create_log(GameID.EVAL_MSG_SEND)
                    self.space_time = 0.

                elif space_status:
                    if self.last_space_status:
                        if self.sim.message_draft and (timestamp - self.space_time) > 0.5:
                            self.sim.clear_message_draft()
                    else:
                        self.space_time = timestamp

                self.last_lclick_status = lclick_status
                self.last_space_status = space_status

        return Action(action_type, counter, timestamp, action_data), global_log_request_counter

    def reinit(self):
        self.sim.audio_system.stop()

        self.rng = np.random.default_rng(self.init_args.seed)
        random.seed(self.init_args.seed)

        self.sim = Simulation(
            self.init_args.tick_rate, self.init_args.volume, self.own_entity_id, rng=self.rng)
        self.session = self.sim.session
        self.stats = StatTracker(self.session, self.own_entity)

        self.audio_stream = self.sim.audio_system.external_buffer
        self.io_lock = self.sim.audio_system.external_buffer_io_lock

        if not self.headless:
            self.sim.audio_system.start()

    def get_user_command(self, current_timestamp: float) -> Tuple[int, Union[float, None]]:
        mmot_yrel = 0
        mmot_xrel = 0

        # Evaluate peripheral events
        event = sdl2.events.SDL_Event()
        event_ptr = ctypes.byref(event)

        while sdl2.events.SDL_PollEvent(event_ptr, 1):
            event_type = event.type

            # Quit
            if event_type == sdl2.SDL_QUIT:
                return self.CMD_EXIT, None

            elif event_type == sdl2.SDL_KEYDOWN:
                keysim = event.key.keysym.sym

                # Check map/scoreboard
                if keysim == sdl2.SDLK_TAB and self.session.phase and self.session.is_spectator(self.sim.own_player_id):
                    self.sim.view = GameID.VIEW_MAPSTATS

                # Volume
                elif keysim == sdl2.SDLK_UP:
                    self.sim.audio_system.volume = np.clip(self.sim.audio_system.volume + 0.05, 0., 1.)
                    self.logger.info('Volume increased to %.2f', self.sim.audio_system.volume)

                elif keysim == sdl2.SDLK_DOWN:
                    self.sim.audio_system.volume = np.clip(self.sim.audio_system.volume - 0.05, 0., 1.)
                    self.logger.info('Volume decreased to %.2f', self.sim.audio_system.volume)

            elif event_type == sdl2.SDL_KEYUP:
                keysim = event.key.keysym.sym

                # Check map/scoreboard
                if keysim == sdl2.SDLK_TAB and self.sim.view == GameID.VIEW_MAPSTATS:
                    if self.session.phase and self.session.is_spectator(self.sim.own_player_id):
                        self.sim.view = GameID.VIEW_WORLD

                # Take screenshot
                elif keysim == sdl2.SDLK_F12:
                    if not os.path.exists(DATA_DIR):
                        os.makedirs(DATA_DIR)

                    file_indices = [
                        int(filename.split('_')[1][:-4])
                        for filename in os.listdir(DATA_DIR) if filename.startswith('screenshot')]

                    file_idx = (max(file_indices)+1) if file_indices else 0
                    file_path = os.path.join(DATA_DIR, f'screenshot_{file_idx:03d}.png')

                    cv2.imwrite(file_path, self.window_array.base[..., :3])

                    self.logger.info("Screenshot saved to: '%s'.", file_path)

                # Quit
                elif keysim == sdl2.SDLK_ESCAPE:
                    return self.CMD_EXIT, None

                # (Un)Pause
                elif keysim == sdl2.SDLK_SPACE or keysim == sdl2.SDLK_k:
                    self.paused = not self.paused
                    self.logger.info('Replay paused.' if self.paused else 'Replay resumed.')

                # Speed up
                elif keysim == sdl2.SDLK_RIGHT:
                    self.speedup_idx = min(len(self.SPEEDUPS)-1, self.speedup_idx+1)
                    self.change_replay_speed(speedup=self.SPEEDUPS[self.speedup_idx])

                    self.strided_refresh: Callable = StridedFunction(
                        self.refresh_display, self._fps_limiter.tick_rate / self.init_args.refresh_rate)

                    self.logger.info('Replay speed changed to %.2f.', self.SPEEDUPS[self.speedup_idx])

                # Slow down
                elif keysim == sdl2.SDLK_LEFT:
                    self.speedup_idx = max(0, self.speedup_idx-1)
                    self.change_replay_speed(speedup=self.SPEEDUPS[self.speedup_idx])

                    self.strided_refresh: Callable = StridedFunction(
                        self.refresh_display, self._fps_limiter.tick_rate / self.init_args.refresh_rate)

                    self.logger.info('Replay speed changed to %.2f.', self.SPEEDUPS[self.speedup_idx])

                # Restart
                elif keysim == sdl2.SDLK_BACKSPACE or keysim == sdl2.SDLK_RETURN:
                    self.logger.info('Restarting replay...')
                    return self.CMD_NONE, self.jump_to_timestamp(current_timestamp, 0.)

                # Jump 10 secs ahead
                elif keysim == sdl2.SDLK_l:
                    self.logger.info('Forwarding 10 seconds ahead...')
                    return self.CMD_NONE, self.jump_to_timestamp(current_timestamp, current_timestamp + 10.)

                # Jump 10 secs back
                elif keysim == sdl2.SDLK_j:
                    self.logger.info('Rewinding 10 seconds back...')
                    return self.CMD_NONE, self.jump_to_timestamp(current_timestamp, max(0., current_timestamp - 10.))

            elif self.session.is_spectator(self.sim.own_player_id):

                # Cycle observed player
                if event_type == sdl2.SDL_MOUSEBUTTONUP:
                    if event.button.button == sdl2.SDL_BUTTON_LEFT:
                        self.sim.change_observed_player(rotate_upward=False)
                    elif event.button.button == sdl2.SDL_BUTTON_RIGHT:
                        self.sim.change_observed_player(rotate_upward=True)

                    self.logger.info('Observing player %d.', self.sim.observed_player_id)

                elif event_type == sdl2.SDL_MOUSEMOTION:
                    mmot_yrel += event.motion.yrel * self.mouse_sensitivity
                    mmot_xrel += event.motion.xrel * self.mouse_sensitivity

                # Scroll chat
                elif event_type == sdl2.SDL_MOUSEWHEEL:
                    if event.wheel.y > 0:
                        self.sim.wheel_y = max(self.sim.wheel_y-1, 0)
                    elif event.wheel.y < 0:
                        self.sim.wheel_y = min(self.sim.wheel_y+1, max(len(self.sim.chat)-5, 0))

        if self.session.is_spectator(self.sim.own_player_id):
            self.sim.cursor_y = np.clip(self.sim.cursor_y + mmot_yrel, 2., 105.)
            self.sim.cursor_x = np.clip(self.sim.cursor_x + mmot_xrel, 66., 253.)

        return (self.CMD_PAUSE if self.paused else self.CMD_NONE), None

    def cleanup(self):
        if self.headless:
            return

        self.sim.audio_system.stop()
        sdl2.SDL_Quit()

    def manual_step(self, previous_clock: Union[float, None]) -> Union[float, None]:
        """Manually step the simulation. Intended for use in tandem with headless mode."""

        if not self.recording:
            return None

        current_clock = self._get_next_batch()
        dt_loop = (current_clock - previous_clock) if previous_clock is not None else 0.

        # Advance local state
        self.step(dt_loop, current_clock)

        return current_clock
