"""Human UI version of the live client for SDG"""

import os
import ctypes
import struct
from argparse import Namespace
from typing import Any, Callable, Tuple, Union

import numpy as np
import cv2
import sdl2
import sdl2.ext

from sidegame.networking.core import StridedFunction
from sidegame.game.shared import GameID, Map, Item, Player
from sidegame.game.client.base import SDGLiveClientBase
from sidegame.game.client.tracking import DATA_DIR


class SDGLiveClient(SDGLiveClientBase):
    """
    A client for live communication with the SDG server.

    It uses `sdl2` to get user inputs from the mouse and keyboard
    and to update the image being displayed on screen,
    while a background thread feeds sound chunks into `pyaudio.Stream`.
    """

    WINDOW_NAME = 'SiDeGame v2021-05-15'
    RENDER_SIZE = (256, 144)

    # Tracked mouse/keyboard state indices
    MKBD_IDX_W = 0      # Forward
    MKBD_IDX_S = 1      # Backward
    MKBD_IDX_D = 2      # Rightward
    MKBD_IDX_A = 3      # Leftward
    MKBD_IDX_E = 4      # Use
    MKBD_IDX_SPACE = 5  # Send/clear message
    MKBD_IDX_LBTN = 6   # Attack
    MKBD_IDX_RBTN = 7   # Walk

    ALPHA = 255 * np.ones((*RENDER_SIZE[::-1], 1), dtype=np.uint8)

    def __init__(self, args: Namespace):
        sdl2.ext.init()

        super().__init__(args)
        self.sim.audio_system.start()

        self.mkbd_state = [0]*8
        self.space_time = 0.

        self.window_size = (round(self.RENDER_SIZE[0]*args.render_scale), round(self.RENDER_SIZE[1]*args.render_scale))
        self.window = sdl2.ext.Window(self.WINDOW_NAME, size=self.window_size)
        self.window.show()
        self.window_array = sdl2.ext.pixels3d(sdl2.SDL_GetWindowSurface(self.window.window).contents)

        sdl2.SDL_SetRelativeMouseMode(sdl2.SDL_TRUE)
        self.cursor_trapped = True

        self.mouse_sensitivity = args.mouse_sensitivity / args.render_scale
        self.discretise_mouse = args.discretise_mouse
        # self.mouse_bins = np.around(np.exp(np.arange(13.)/2.5579) - 1., 2)
        self.mouse_bins = np.array([0., 0.48, 1.19, 2.23, 3.78, 6.06, 9.44, 14.43, 21.82, 32.73, 48.87, 72.73, 108.])

        # Decoupled refresh allows the game state to be processed at higher framerate than e.g. 60Hz monitor limit
        # without the cost of actually rendering a frame
        self.strided_refresh: Callable = StridedFunction(self.refresh_display, args.tick_rate / args.refresh_rate)

    def poll_user_input(self, timestamp: float) -> Tuple[Any, Union[Any, None]]:
        """
        Poll or read peripheral events and interpret them as user input
        and optional local log data.

        NOTE: `pysdl2` intends for events to be obtained with `sdl2.ext.get_events()`,
        but there was an issue with events lagging in certain conditions,
        which switching to `SDL_PollEvent` seems to fix. See:
        https://discourse.urho3d.io/t/vsync-low-fps-input-lag-problem/733/2
        https://github.com/marcusva/py-sdl2/blob/master/sdl2/ext/common.py#L76
        """

        sim = self.sim
        session = self.session

        # Default values of untracked states
        mmot_yrel = 0
        mmot_xrel = 0
        mwhl_y = 0
        kbd_r = 0
        kbd_g = 0
        kbd_num = 0

        # Default (null) log
        log = None

        # Evaluate peripheral events
        event = sdl2.events.SDL_Event()
        event_ptr = ctypes.byref(event)

        while True:
            event_in_queue = sdl2.events.SDL_PollEvent(event_ptr, 1)

            if not event_in_queue:
                break

            event_type = event.type

            # Exit
            if event_type == sdl2.SDL_QUIT:
                raise KeyboardInterrupt

            elif event_type == sdl2.SDL_KEYDOWN:
                keysim = event.key.keysym.sym

                # General
                if keysim == sdl2.SDLK_UP:
                    sim.audio_system.volume = np.clip(sim.audio_system.volume + 0.05, 0., 2.)
                    self.logger.info('Volume increased to %.2f', sim.audio_system.volume)

                elif keysim == sdl2.SDLK_DOWN:
                    sim.audio_system.volume = np.clip(sim.audio_system.volume - 0.05, 0., 2.)
                    self.logger.info('Volume decreased to %.2f', sim.audio_system.volume)

                # In lobby
                elif sim.view == GameID.VIEW_LOBBY:
                    if keysim == sdl2.SDLK_ESCAPE:
                        continue

                    # Evaluate console command
                    elif keysim == sdl2.SDLK_RETURN:
                        if sim.console_text == 'exit':
                            raise KeyboardInterrupt

                        elif sim.console_text == 'mouse':
                            if self.cursor_trapped:
                                sdl2.SDL_SetRelativeMouseMode(sdl2.SDL_FALSE)
                            else:
                                sdl2.SDL_SetRelativeMouseMode(sdl2.SDL_TRUE)

                            self.cursor_trapped = not self.cursor_trapped
                            self.logger.debug('Relative mouse mode set to %s.', self.cursor_trapped)

                        elif sim.console_text == 'stats' and self.stats is not None:
                            print_string = '\n\n'

                            for stat_key, stat_val in self.stats.summary():
                                print_string += f'{stat_key}: ' + (
                                    f'{stat_val}\n' if isinstance(stat_val, int) else f'{stat_val:.2f}\n')

                            print(print_string)

                        elif sim.console_text.startswith('set name'):
                            name = [
                                min(ord(m), 255) for m in sim.console_text[9:13] + ' '*(4-len(sim.console_text[9:13]))]
                            log = [sim.own_player_id, GameID.CMD_SET_NAME, *name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.]

                        elif sim.console_text.startswith('set team'):
                            words = sim.console_text.split(' ')

                            if len(words) == 4:
                                moved_player_id = int(words[2])

                                if words[3] == 't':
                                    team = GameID.GROUP_TEAM_T
                                elif words[3] == 'ct':
                                    team = GameID.GROUP_TEAM_CT
                                elif words[3] == 's':
                                    team = GameID.GROUP_SPECTATORS
                                else:
                                    team = GameID.NULL

                            elif len(words) >= 3:
                                moved_player_id = sim.own_player_id

                                if words[2] == 't':
                                    team = GameID.GROUP_TEAM_T
                                elif words[2] == 'ct':
                                    team = GameID.GROUP_TEAM_CT
                                elif words[2] == 's':
                                    team = GameID.GROUP_SPECTATORS
                                else:
                                    team = GameID.NULL

                            else:
                                team = GameID.NULL

                            if team != GameID.NULL:
                                log = [
                                    sim.own_player_id, GameID.CMD_SET_TEAM, moved_player_id, team,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.]

                        elif sim.console_text.startswith('set role'):
                            self.role_key = '0x' + sim.console_text.split(' ')[-1]

                            try:
                                role_key = int(self.role_key, 16)

                                fractured_role_key = struct.unpack('>4B', struct.pack('>L', role_key))
                                log = [
                                    sim.own_player_id, GameID.CMD_SET_ROLE, *fractured_role_key,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0.]

                            except ValueError:
                                self.logger.warning('Invalid role key.')

                        elif sim.console_text in sim.CONSOLE_COMMANDS:
                            cmd = sim.CONSOLE_COMMANDS[sim.console_text]
                            log = [sim.own_player_id, cmd, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.]

                        sim.console_text = ''

                    # Remove character from text
                    elif keysim == sdl2.SDLK_BACKSPACE:
                        sim.console_text = sim.console_text[:-1]

                    # Add character to text
                    elif len(sim.console_text) < 23 and (0 <= keysim <= 1114111):
                        sim.console_text += chr(keysim)

                # Spectator in world view
                elif session.is_spectator(sim.own_player_id):
                    if keysim == sdl2.SDLK_TAB and session.phase:
                        sim.view = GameID.VIEW_MAPSTATS

                # Player in world view
                elif session.phase:
                    if keysim == sdl2.SDLK_w:
                        self.mkbd_state[self.MKBD_IDX_W] = 1
                    elif keysim == sdl2.SDLK_s:
                        self.mkbd_state[self.MKBD_IDX_S] = -1
                    elif keysim == sdl2.SDLK_d:
                        self.mkbd_state[self.MKBD_IDX_D] = 1
                    elif keysim == sdl2.SDLK_a:
                        self.mkbd_state[self.MKBD_IDX_A] = -1
                    elif keysim == sdl2.SDLK_e:
                        self.mkbd_state[self.MKBD_IDX_E] = 1

                    elif keysim == sdl2.SDLK_x and sim.view != GameID.VIEW_TERMS:
                        sim.view = GameID.VIEW_TERMS
                        sim.cursor_y, sim.cursor_x = sim.WORLD_FRAME_CENTRE

                    elif keysim == sdl2.SDLK_c and sim.view != GameID.VIEW_ITEMS:
                        sim.view = GameID.VIEW_ITEMS
                        sim.cursor_y, sim.cursor_x = sim.WORLD_FRAME_CENTRE

                    elif keysim == sdl2.SDLK_b and sim.view != GameID.VIEW_STORE:
                        can_view_store = sim.observed_player_id == sim.own_player_id and \
                            session.check_player_buy_eligibility(sim.own_player_id)

                        if can_view_store:
                            sim.view = GameID.VIEW_STORE
                            sim.cursor_y, sim.cursor_x = sim.WORLD_FRAME_CENTRE

                    elif keysim == sdl2.SDLK_TAB:
                        sim.view = GameID.VIEW_MAPSTATS

                    elif keysim == sdl2.SDLK_r:
                        kbd_r = 1
                    elif keysim == sdl2.SDLK_g:
                        kbd_g = 1

                    elif keysim == sdl2.SDLK_1:
                        kbd_num = 1
                    elif keysim == sdl2.SDLK_2:
                        kbd_num = 2
                    elif keysim == sdl2.SDLK_3:
                        kbd_num = 3
                    elif keysim == sdl2.SDLK_4:
                        kbd_num = 4
                    elif keysim == sdl2.SDLK_5:
                        kbd_num = 5

                    # NOTE: Apparently keydown keeps triggering on hold...
                    elif keysim == sdl2.SDLK_SPACE:
                        if self.mkbd_state[self.MKBD_IDX_SPACE]:
                            if sim.message_draft and (timestamp - self.space_time) > 0.5:
                                sim.clear_message_draft()
                        else:
                            self.mkbd_state[self.MKBD_IDX_SPACE] = 1
                            self.space_time = timestamp

            elif event_type == sdl2.SDL_KEYUP:
                keysim = event.key.keysym.sym

                # Take screenshot
                if keysim == sdl2.SDLK_F12:
                    if not os.path.exists(DATA_DIR):
                        os.makedirs(DATA_DIR)

                    file_indices = [
                        int(filename.split('_')[1][:-4])
                        for filename in os.listdir(DATA_DIR) if filename.startswith('screenshot')]

                    file_idx = (max(file_indices)+1) if file_indices else 0
                    file_path = os.path.join(DATA_DIR, f'screenshot_{file_idx:03d}.png')

                    cv2.imwrite(file_path, self.window_array.base[..., :3])

                    self.logger.info("Screenshot saved to: '%s'.", file_path)

                elif keysim == sdl2.SDLK_w:
                    self.mkbd_state[self.MKBD_IDX_W] = 0
                elif keysim == sdl2.SDLK_s:
                    self.mkbd_state[self.MKBD_IDX_S] = 0
                elif keysim == sdl2.SDLK_d:
                    self.mkbd_state[self.MKBD_IDX_D] = 0
                elif keysim == sdl2.SDLK_a:
                    self.mkbd_state[self.MKBD_IDX_A] = 0
                elif keysim == sdl2.SDLK_e:
                    self.mkbd_state[self.MKBD_IDX_E] = 0

                elif keysim == sdl2.SDLK_ESCAPE:
                    if sim.view == GameID.VIEW_LOBBY:
                        sim.enter_world()
                    else:
                        sim.exit_world()

                # Add term to message
                elif keysim == sdl2.SDLK_x and sim.view == GameID.VIEW_TERMS:
                    sim.view = GameID.VIEW_WORLD
                    log = sim.create_log(GameID.EVAL_MSG_TERM)

                # Add item to message
                elif keysim == sdl2.SDLK_c and sim.view == GameID.VIEW_ITEMS:
                    sim.view = GameID.VIEW_WORLD
                    log = sim.create_log(GameID.EVAL_MSG_ITEM)

                # Send message
                elif keysim == sdl2.SDLK_SPACE and sim.view != GameID.VIEW_LOBBY and (
                    not session.is_spectator(sim.own_player_id)
                ):
                    log = sim.create_log(GameID.EVAL_MSG_SEND)
                    self.mkbd_state[self.MKBD_IDX_SPACE] = 0
                    self.space_time = 0.

                # Buy an item
                elif keysim == sdl2.SDLK_b and sim.view == GameID.VIEW_STORE:
                    sim.view = GameID.VIEW_WORLD
                    log = sim.create_log(GameID.EVAL_BUY)

                elif keysim == sdl2.SDLK_TAB and sim.view == GameID.VIEW_MAPSTATS:
                    sim.view = GameID.VIEW_WORLD

            # Update scroll position, enforcing valid range
            elif event_type == sdl2.SDL_MOUSEWHEEL:
                if event.wheel.y > 0:
                    sim.wheel_y = max(sim.wheel_y-1, 0)
                    mwhl_y -= 1
                elif event.wheel.y < 0:
                    sim.wheel_y = min(sim.wheel_y+1, max(len(sim.chat)-5, 0))
                    mwhl_y += 1

            elif event_type == sdl2.SDL_MOUSEBUTTONDOWN:
                if event.button.button == sdl2.SDL_BUTTON_LEFT:
                    self.mkbd_state[self.MKBD_IDX_LBTN] = 1
                elif event.button.button == sdl2.SDL_BUTTON_RIGHT:
                    self.mkbd_state[self.MKBD_IDX_RBTN] = 1

            elif event_type == sdl2.SDL_MOUSEBUTTONUP:
                if event.button.button == sdl2.SDL_BUTTON_LEFT:
                    self.mkbd_state[self.MKBD_IDX_LBTN] = 0
                    sim.change_observed_player(rotate_upward=False)

                    if sim.view == GameID.VIEW_MAPSTATS:
                        log = sim.create_log(GameID.EVAL_MSG_MARK)

                elif event.button.button == sdl2.SDL_BUTTON_RIGHT:
                    self.mkbd_state[self.MKBD_IDX_RBTN] = 0
                    sim.change_observed_player(rotate_upward=True)

            elif event_type == sdl2.SDL_MOUSEMOTION:
                mmot_yrel += event.motion.yrel * self.mouse_sensitivity
                mmot_xrel += event.motion.xrel * self.mouse_sensitivity

        # Discretise mouse motion (for AI demos)
        if self.discretise_mouse:
            mmot_yrel = self.mouse_bins[np.argmin(np.abs(np.abs(mmot_yrel) - self.mouse_bins))] * np.sign(mmot_yrel)
            mmot_xrel = self.mouse_bins[np.argmin(np.abs(np.abs(mmot_xrel) - self.mouse_bins))] * np.sign(mmot_xrel)

        # Update cursor and angle movement
        sim.cursor_y = np.clip(sim.cursor_y + mmot_yrel, 2., 105.)

        if sim.view == GameID.VIEW_WORLD and not session.is_dead_or_spectator(sim.own_player_id):
            sim.cursor_x = 159.5
            d_angle = float(mmot_xrel)
        else:
            sim.cursor_x = np.clip(sim.cursor_x + mmot_xrel, 66., 253.)
            d_angle = 0.

        # Get drawn item id from key num
        draw_id = self.get_next_item_by_slot(kbd_num)

        # Get hovered (entity) ID
        hovered_id = GameID.NULL
        hovered_entity_id = Map.PLAYER_ID_NULL

        if sim.view == GameID.VIEW_WORLD:
            if session.is_dead_or_spectator(sim.own_player_id):
                hovered_entity_id = sim.observed_player_id
            else:
                _, hovered_entity_id, hovered_id = sim.get_cursor_obs_from_world_view()

        elif sim.view == GameID.VIEW_TERMS:
            hovered_id = sim.get_cursor_obs_from_code_view(sim.code_view_terms)

        elif sim.view == GameID.VIEW_ITEMS:
            hovered_id = sim.get_cursor_obs_from_code_view(sim.code_view_items)

        elif sim.view == GameID.VIEW_STORE:
            if self.own_entity.team == GameID.GROUP_TEAM_T:
                hovered_id = sim.get_cursor_obs_from_code_view(sim.code_view_store_t)
            else:
                hovered_id = sim.get_cursor_obs_from_code_view(sim.code_view_store_ct)

        state = [
            self.mkbd_state[self.MKBD_IDX_LBTN], self.mkbd_state[self.MKBD_IDX_RBTN],
            self.mkbd_state[self.MKBD_IDX_SPACE], self.mkbd_state[self.MKBD_IDX_E],
            kbd_g, kbd_r, draw_id, round(sim.cursor_x), round(sim.cursor_y),
            self.mkbd_state[self.MKBD_IDX_W] + self.mkbd_state[self.MKBD_IDX_S] + 1,
            self.mkbd_state[self.MKBD_IDX_D] + self.mkbd_state[self.MKBD_IDX_A] + 1,
            max(-1, min(1, mwhl_y)) + 1, sim.view, hovered_id, hovered_entity_id, d_angle]

        return state, log

    def get_next_item_by_slot(self, slot: int, subslot: int = 0) -> int:
        """
        Get the ID of the item at the specified slot. If the player is holding
        a utility, which can share its slot with multiple others, the utilities
        are cycled through to the next.
        """

        if not slot or not self.session.phase:
            return GameID.NULL

        player: Player = self.session.players.get(self.sim.own_player_id, None)

        if player is None:
            return GameID.NULL

        if slot == Item.SLOT_UTILITY and any(player.slots[Item.SLOT_UTILITY:]):
            if slot == player.held_object.item.slot:
                subslot = 0 if player.held_object.item.subslot == 3 else (player.held_object.item.subslot + 1)

            while player.slots[slot + subslot] is None:
                subslot = 0 if subslot == 3 else (subslot + 1)

        obj = player.slots[slot + subslot]

        return obj.item.id if obj is not None else GameID.NULL

    def generate_output(self, dt: float):
        self.sim.eval_effects(dt)
        self.strided_refresh()

    def refresh_display(self):
        """
        Produce a new frame, upscale it, and update the image on screen.

        NOTE: While resource demanding, resizing and copying are only necessary
        in human interfaces, but an AI interface would have bottlenecks
        of its own, as well.
        """

        frame = self.sim.get_frame()
        frame = np.concatenate((frame, self.ALPHA), axis=-1)

        # Upscale directly to window array memory
        cv2.resize(frame, self.window_size, dst=self.window_array.base, interpolation=cv2.INTER_NEAREST)

        # Redraw
        self.window.refresh()

    def cleanup(self):
        if self.stats is not None:
            path_to_stats = self.stats.save()

            self.logger.info('Stats saved to: %s', path_to_stats)

        self.sim.audio_system.stop()
        sdl2.SDL_Quit()
