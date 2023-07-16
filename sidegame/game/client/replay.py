"""Interactive demo player for SDG"""

from collections import deque
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

from sidegame.assets import Map
from sidegame.utils import StridedFunction
from sidegame.networking import Entry, Action, ReplayClient
from sidegame.game import GameID
from sidegame.game.shared import Session
from sidegame.game.client.interface import SDGLiveClient, _create_rgb_surface, _get_fullscreen_mode
from sidegame.game.client.simulation import Simulation
from sidegame.game.client.tracking import DATA_DIR, StatTracker, FocusTracker


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

    def __init__(
        self,
        args: Namespace,
        headless: bool = False,
        borderless: bool = True,
        vsync: bool = True
    ):
        self.rng = np.random.default_rng(args.seed)
        random.seed(args.seed)

        # Set original tick rate
        args.tick_rate = float(args.recording_path.split('/')[-1].split('_')[-2].split('-')[-1])
        self.time_scale = args.time_scale  # TODO: Inferring original time scale is yet to be implemented

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
        self.max_tick_counter = self.recorder.split_meta(self.recorder.buffer[-1])[0][1]

        self.session = Session(rng=self.rng)
        self.sim = Simulation(self.own_entity_id, args.tick_rate, args.audio_device, self.session)
        self.stats = StatTracker(self.session, self.own_entity)
        self.focus = FocusTracker(
            path=args.focus_path,
            mode=(FocusTracker.MODE_WRITE if args.focus_record else FocusTracker.MODE_READ),
            start_active=(not headless))

        # Expose observations for external access
        self.headless = headless

        self.video_stream: Deque[np.ndarray] = deque()
        self.audio_stream: Deque[np.ndarray] = self.sim.audio_system.external_buffer
        self.action_stream: Deque[Tuple[int, float]] = deque()
        self.io_lock: Lock = self.sim.audio_system.external_buffer_io_lock

        if not args.render_scale:
            fullscreen = True
            *self.window_size, _ = _get_fullscreen_mode()
            args.render_scale = self.window_size[0] / SDGLiveClient.RENDER_SIZE[0]

        else:
            fullscreen = False
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

            self.window = sdl2.ext.Window(
                SDGLiveClient.WINDOW_NAME,
                size=self.window_size,
                flags=(
                    sdl2.SDL_WINDOW_OPENGL
                    | sdl2.SDL_WINDOW_SHOWN
                    | (
                        sdl2.SDL_WINDOW_FULLSCREEN_DESKTOP
                        if fullscreen
                        else (sdl2.SDL_WINDOW_BORDERLESS if borderless else 0))))

            self.frame = _create_rgb_surface(*SDGLiveClient.RENDER_SIZE)
            self.frame_array = sdl2.ext.pixels3d(self.frame, transpose=False)
            self.frame_texture = None

            self.renderer = sdl2.ext.renderer.Renderer(
                self.window,
                backend='opengl',
                logical_size=SDGLiveClient.RENDER_SIZE,
                flags=sdl2.SDL_RENDERER_ACCELERATED | (sdl2.SDL_RENDERER_PRESENTVSYNC if vsync else 0))

            self.render = sdl2.render.SDL_RenderCopyEx if sdl2.dll.version < 2010 else sdl2.render.SDL_RenderCopyExF

            self.mouse_sensitivity = args.mouse_sensitivity / args.render_scale

            # Decoupled refresh allows the game state to be processed at higher framerate than e.g. 60Hz monitor limit
            # without the cost of actually rendering a frame (including fast-forwarding)
            self.strided_refresh: Callable = StridedFunction(
                self.refresh_display, self._fps_limiter.tick_rate / args.refresh_rate)

            if self.focus.mode == FocusTracker.MODE_WRITE:
                sdl2.SDL_SetRelativeMouseMode(sdl2.SDL_TRUE)

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

        # Add focus marker
        # NOTE: Tick counter was incremented after extracting data for the current tick, hence the -1
        self.focus.get(self._tick_counter-1)
        self.focus.register(frame, self._tick_counter-1)

        # Update externally accessible image and audio buffer
        if self.headless:
            with self.io_lock:
                self.video_stream.append(
                    frame if self.init_args.render_scale == 1 else
                    cv2.resize(frame, self.window_size, interpolation=cv2.INTER_NEAREST))

            self.sim.audio_system.step()

        # Upscale directly to window array memory and redraw
        else:
            # Add progress info
            width = round(self._tick_counter / self.max_tick_counter * 256)
            frame[-1, :width, 1] = 127

            np.copyto(self.frame_array, frame)

            frame_texture = sdl2.render.SDL_CreateTextureFromSurface(self.renderer.sdlrenderer, self.frame)

            self.render(self.renderer.sdlrenderer, frame_texture, None, None, 0, None, sdl2.render.SDL_FLIP_NONE)
            self.renderer.present()

            if self.frame_texture is not None:
                sdl2.render.SDL_DestroyTexture(self.frame_texture)
                self.frame_texture = frame_texture

    def generate_output(self, dt: float):
        self.sim.eval_effects(dt * self.time_scale)

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

        return client_id, interp_ratio / update_rate, init_clock_diff

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
            if self.headless:
                self.action_stream.append(action_data)

            lclick_status = action_data[0]
            space_status = action_data[2]
            cursor_x = action_data[7]
            cursor_y = action_data[8]
            wheel_y = action_data[11] - 1
            view = action_data[12]
            hovered_entity_id = action_data[14]

            if not self.session.is_spectator(self.sim.own_player_id):
                # If dead, follow the originally observed player
                if self.session.is_dead_player(self.sim.own_player_id) and view != GameID.VIEW_LOBBY:
                    if self.sim.view == GameID.VIEW_WORLD and hovered_entity_id != Map.PLAYER_ID_NULL:
                        self.sim.observed_player_id = hovered_entity_id

                # When returning to life, switch back to own player
                elif self.sim.observed_player_id != self.sim.own_player_id:
                    self.sim.observed_player_id = self.sim.own_player_id

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

                    # Follow cursor
                    self.sim.cursor_x = cursor_x

                    # Follow chat scroll
                    if wheel_y < 0:
                        self.sim.wheel_y = max(self.sim.wheel_y-1, 0)
                    elif wheel_y > 0:
                        self.sim.wheel_y = min(self.sim.wheel_y+1, max(len(self.sim.chat)-5, 0))

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

                self.sim.cursor_y = cursor_y
                self.last_lclick_status = lclick_status
                self.last_space_status = space_status

        return Action(action_type, counter, timestamp, action_data), global_log_request_counter

    def reinit(self):
        self.sim.audio_system.stop()
        self.sim.map.reset()

        self.rng = np.random.default_rng(self.init_args.seed)
        random.seed(self.init_args.seed)

        self.session = Session(rng=self.rng)
        assets = (self.sim.images, self.sim.sounds, self.sim.map, self.sim.inventory)

        self.sim = Simulation(
            self.own_entity_id, self.init_args.tick_rate, self.init_args.audio_device, self.session, assets)

        self.stats = StatTracker(self.session, self.own_entity)

        self.audio_stream = self.sim.audio_system.external_buffer
        self.io_lock = self.sim.audio_system.external_buffer_io_lock

    def pause_effects(self):
        self.sim.audio_system.paused = True

    def resume_effects(self):
        with self.sim.audio_system._audio_channels_io_lock:
            for channel in self.sim.audio_system._audio_channels:
                channel.clear()

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

                    cv2.imwrite(file_path, self.frame_array)

                    self.logger.info("Screenshot saved to: '%s'.", file_path)

                # Quit
                elif keysim == sdl2.SDLK_ESCAPE:
                    return self.CMD_EXIT, None

                # (Un)Pause
                elif keysim == sdl2.SDLK_SPACE or keysim == sdl2.SDLK_k:
                    self.paused = not self.paused

                    if self.focus.mode == FocusTracker.MODE_WRITE:
                        sdl2.SDL_SetRelativeMouseMode(sdl2.SDL_FALSE if self.paused else sdl2.SDL_TRUE)

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

                # Toggle labelling/display
                elif keysim == sdl2.SDLK_x:
                    if self.focus.mode != FocusTracker.MODE_NULL:
                        self.focus.active = not self.focus.active

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

            elif event_type == sdl2.SDL_MOUSEMOTION and self.focus.mode == FocusTracker.MODE_WRITE:
                mmot_yrel += event.motion.yrel * self.mouse_sensitivity
                mmot_xrel += event.motion.xrel * self.mouse_sensitivity

        if self.session.is_spectator(self.sim.own_player_id):
            self.sim.cursor_y = np.clip(self.sim.cursor_y + mmot_yrel, 2., 105.)
            self.sim.cursor_x = np.clip(self.sim.cursor_x + mmot_xrel, 66., 253.)

        if self.focus.mode == FocusTracker.MODE_WRITE and not self.paused:
            self.focus.update(mmot_yrel, mmot_xrel)

        return (self.CMD_PAUSE if self.paused else self.CMD_NONE), None

    def cleanup(self):
        self.focus.finish(self.logger)

        if self.headless:
            return

        self.sim.audio_system.stop()
        self.renderer.destroy()
        self.window.close()
        sdl2.SDL_Quit()

    def manual_step(self, previous_clock: Union[float, None], generate_obs: bool = True) -> Union[float, None]:
        """Manually step the simulation. Intended for use in tandem with headless mode."""

        if not self.recording:
            return None

        current_clock = self._get_next_batch()
        dt_loop = (current_clock - previous_clock) if previous_clock is not None else 0.

        # Advance local state
        if generate_obs:
            self.step(dt_loop, current_clock)

        else:
            # Check for and unpack server data
            server_data = self._get_server_data(current_clock)
            state_updates = self._get_state_updates(server_data) if server_data else None

            # Update local state wrt. state on the server
            self._eval_server_state(state_updates, current_clock)

            # Update local state wrt. user input
            user_input = self._get_user_input(current_clock)
            action = self._eval_input(user_input, current_clock) if user_input else None

            # Update foreign entities wrt. estimated server time
            self._interpolate_foreign_entities(current_clock + self._clock_diff_tracker.value - self._interp_window)

            # Record, send, and/or clean up
            self._relay(server_data, action, current_clock, self._clock_diff_tracker.value)

            # Generate min output
            self.sim.eval_effects(dt_loop * self.time_scale)
            self.focus.get(self._tick_counter-1)

            for channel in self.sim.audio_system._audio_channels:
                channel.clear()

        return current_clock
