"""Main and supporting methods for client-side world simulation of SDG"""

import math

import numpy as np
from numpy import ndarray
from numba import jit

from sidegame.assets import ImageBank, SoundBank, MapID, MAP_WARP
from sidegame.utils_jit import vec2_norm2, vec2_rot_, fix_angle_range, get_disk_indices, F_PI2, F_2PI, RAD_DIV_DEG
from sidegame.audio import PlanarAudioSystem as AudioSystem, DummyAudioSystem
from sidegame.physics import trace_sight, MAX_VIEW_RANGE
from sidegame.effects import Effect
from sidegame.graphics import (
    draw_image, draw_text, draw_number, draw_colour, draw_overlay, draw_muted,
    get_camera_warp, get_inverse_warp, warp_position, project_into_view, mask_view)
from sidegame.game import GameID
from sidegame.game.shared import Message, Item, Inventory, Object, Player, Session


class Simulation:
    """Generates and holds the state of the game world from the client's perspective."""

    AUDIO_MAX_DISTANCE = 108.
    AUDIO_DISTANCE_SCALING = 512./108.
    AUDIO_BASE_VOLUME = 0.25

    WORLD_FRAME_CENTRE = (108./2. + 1., 64. + 192./2.)  # Shift by 1px; centre in terms/items/store images is wrong
    WORLD_FRAME_ORIGIN = (95.5, 106.5)
    WORLD_FRAME_SIZE = (192, 108)
    WORLD_BOUNDS = WORLD_FRAME_SIZE[::-1]

    RING1_INDICES = np.array((-1, 0, 0, 1)), np.array((0, -1, 1, 0))
    RING2_INDICES = np.array((-1, -1, 1, 1)), np.array((-1, 1, -1, 1))

    SPRITE_CAP_INDICES = np.array((1, 1, 2, 2)), np.array((1, 2, 1, 2))
    SPRITE_AURA_INDICES = get_disk_indices(2)
    SPRITE_AURA_INDICES = SPRITE_AURA_INDICES[0]+1, SPRITE_AURA_INDICES[1]+1

    FOV_STD = 106.3
    FOV_SCOPED = 25.
    ANGLE_LIM_STD = FOV_STD / 2. * RAD_DIV_DEG
    ANGLE_LIM_SCOPED = FOV_SCOPED / 2. * RAD_DIV_DEG

    CONSOLE_COMMANDS: dict[str, int] = {
        'start': GameID.CMD_START_MATCH,
        'stop': GameID.CMD_END_MATCH,
        'quit': GameID.CMD_END_SESSION,
        'set team': GameID.CMD_SET_TEAM,
        'set name': GameID.CMD_SET_NAME,
        'set role': GameID.CMD_SET_ROLE,
        'ping': GameID.CMD_GET_LATENCY,
        'rundown': GameID.CHEAT_END_ROUND,
        'dev mode': GameID.CHEAT_DEV_MODE,
        'max money': GameID.CHEAT_MAX_MONEY,
        'add bot': GameID.CMD_ADD_BOT,
        'kick': GameID.CMD_KICK}

    DEFAULT_OBS = (np.inf, MapID.PLAYER_ID_NULL, GameID.NULL)

    EQUIPMENT_DRAW_PARAMS = (
        (Item.SLOT_PRIMARY, 112, 187),
        (Item.SLOT_PISTOL, 112, 205),
        (Item.SLOT_OTHER, 112, 241),
        (Item.SLOT_UTILITY + Item.SUBSLOT_FLASH, 128, 187),
        (Item.SLOT_UTILITY + Item.SUBSLOT_EXPLOSIVE, 128, 205),
        (Item.SLOT_UTILITY + Item.SUBSLOT_INCENDIARY, 128, 223),
        (Item.SLOT_UTILITY + Item.SUBSLOT_SMOKE, 128, 241))

    def __init__(
        self,
        own_player_id: int,
        tick_rate: float,
        audio_device: int = 0,
        session: Session = None,
        assets: tuple[ImageBank, SoundBank, Inventory] = None
    ):
        self.session = Session() if session is None else session
        self.map = self.session.map

        if assets is None:
            self.images = ImageBank()
            self.sounds = SoundBank(tick_rate)
            self.inventory = Inventory(self.images, self.sounds.item_sounds)

        else:
            self.images, self.sounds, self.inventory = assets

        self.audio_system = DummyAudioSystem(step_freq=tick_rate) if audio_device == -1 else AudioSystem(
            step_freq=tick_rate,
            max_distance=self.AUDIO_MAX_DISTANCE,
            distance_scaling=self.AUDIO_DISTANCE_SCALING,
            base_volume=self.AUDIO_BASE_VOLUME,
            out_device_idx=audio_device)

        self.effects: dict[int, Effect] = {}
        self.own_player_id = own_player_id
        self.observed_player_id = own_player_id
        self.observer_lock_time = 0.

        self.cursor_y, self.cursor_x = self.WORLD_FRAME_CENTRE
        self.wheel_y = 0
        self.console_text = ''
        self.message_draft: list[tuple[int, float]] = []
        self.chat: list[Message] = []
        self.view: int = GameID.VIEW_LOBBY

        # View-related
        self.code_view_terms = self.images['code_view_terms']
        self.code_view_items = self.images['code_view_items']
        self.code_view_store_t = self.images['code_view_store_t']
        self.code_view_store_ct = self.images['code_view_store_ct']

        self.overlay_terms = self.images['overlay_terms']
        self.overlay_items = self.images['overlay_items']
        self.overlay_store_t = self.images['overlay_store_t']
        self.overlay_store_ct = self.images['overlay_store_ct']
        self.overlay_mapstats = self.images['overlay_mapstats']

        self.window_base_world = self.images['window_base_world']
        self.window_base_lobby = self.images['window_base_lobby']

        self.last_world_frame: ndarray = None
        self.fx_canvas = self.map.fx_canvas
        self.fx_ctr_map = self.map.fx_ctr_map

        self.line_sight_indices = np.arange(0, 106), np.repeat(64 + 96, 106)
        self.standard_fov_endpoints = self.get_view_endpoints(self.FOV_STD, 108.)
        self.scoped_fov_endpoints = self.get_view_endpoints(self.FOV_SCOPED, 108.)

        # Sound bank
        self.movements = self.sounds.movements
        self.keypresses = [self.sounds.item_sounds['c4'][f'press{i}'] for i in range(1, 8)]
        self.footsteps = self.sounds.footsteps

        # Image bank
        self.icon_console_pointer = self.images['pointer_console']
        self.icon_cursor = self.images['pointer_cursor']
        self.icon_selected = self.images['pointer_item']
        self.icon_reset = self.images['phase_reset']
        self.icon_store = self.images['phase_buy']
        self.icon_kill = self.images['term_kill']
        self.icon_c4 = self.images['item_c4']
        self.icon_placeholder = self.images['term_stop']

        self.characters = self.images.characters
        self.digits = self.images.digits
        self.addressable_icons = self.images.id_icons
        self.colours = self.images.COLOURS
        self.player_colours = self.images.player_colours
        self.other_colours = self.images.other_colours
        self.sprites = self.images.sprites

    def get_view_endpoints(self, fov_deg: float, radius: float) -> tuple[ndarray, ...]:
        """
        Given a field of view and limited radius, get the endpoints for rays that
        define the viewable area.
        """

        assert fov_deg <= 180., f'Max. field of view range exceeded: {fov_deg:.2f}'

        # Get endpoint image source
        endpoint_image = self.images['endpoints']

        # Convert FOV to radians
        fov_rad = fov_deg * RAD_DIV_DEG

        # Mask away the endpoints below the bottom threshold determined by FOV
        threshold_y = math.ceil(radius * (1. - math.cos(fov_rad/2.)))
        endpoint_image[threshold_y:] = 0

        # Split the image into left and right halves
        left_half_image, right_half_image = np.hsplit(endpoint_image, 2)

        # Get endpoints (indices)
        left_half_idy, left_half_idx = np.nonzero(left_half_image)
        right_half_idy, right_half_idx = np.nonzero(right_half_image)

        # Correct right half indices after splitting
        right_half_idx += endpoint_image.shape[1] // 2

        return left_half_idy, left_half_idx, right_half_idy, right_half_idx

    def get_cursor_obs_from_code_view(self, code_view: ndarray) -> int:
        """Get the ID corresponding to the position above which the cursor is hovering."""

        return code_view[round(self.cursor_y), round(self.cursor_x)-64]

    def draw_cursor_obs_from_code_view(self, window: ndarray, code_view: ndarray):
        """Draw the icon of the observed term or item lying under the cursor."""

        obs_id = self.get_cursor_obs_from_code_view(code_view)

        if obs_id:
            draw_image(window, self.addressable_icons[obs_id], 112, 85)

    def get_cursor_obs_from_world_view(self) -> tuple[float, int, int]:
        """
        Get the IDs corresponding to the entity above which the cursor is
        roughly hovering (along with the distance to the cursor).
        """

        player: Player = self.session.players.get(self.own_player_id, None)

        if player is None or not player.health:
            return self.DEFAULT_OBS

        # Transform cursor to world frame (rather than the other way around)
        cursor_dist = MAX_VIEW_RANGE - self.cursor_y
        cursor_pos = player.pos + np.array((math.cos(player.angle), math.sin(player.angle))) * cursor_dist

        # Iter over players, get argmin wrt. cursor position
        closest_player_obs = min((
            (vec2_norm2(a_player.pos - cursor_pos), a_player.id, a_player.position_id)
            for a_player in self.session.players.values() if a_player.team != GameID.GROUP_SPECTATORS
            and self.check_los(player, a_player)),
            default=self.DEFAULT_OBS)

        # Iter over objects, get argmin wrt. cursor position
        # NOTE: For the sake of gameplay, C4 can be seen through smoke zones
        closest_object_obs = min((
            (vec2_norm2(an_object.pos - cursor_pos), an_object.id, an_object.item.id)
            for an_object in self.session.objects.values()
            if (an_object.lifetime == np.inf and self.check_los(player, an_object))
            or (an_object.item.id == GameID.ITEM_C4 and self.check_los(player, an_object, ignore_zone=True))),
            default=self.DEFAULT_OBS)

        # Get closest from both players and objects
        closest_obs = min(closest_player_obs, closest_object_obs)
        closest_dist = closest_obs[0]

        # Check if distance to cursor is valid
        return self.DEFAULT_OBS if closest_dist > 3. else closest_obs

    def draw_cursor_obs_from_world_view(self, window: ndarray):
        """Draw the icon of the observed entity lying roughly under the cursor."""

        _, _, closest_id = self.get_cursor_obs_from_world_view()

        if closest_id:
            draw_image(window, self.addressable_icons[closest_id], 112, 85)

    def draw_focus_line(self, window: ndarray):
        """Trace a green ray to the farthest unobscured point directly in front of the player."""

        player: Player = self.session.players[self.own_player_id]
        map_ = self.map

        if not player.health:
            return

        endpoint = player.get_focal_point(map_.wall, map_.zone, map_.player_id, max_range=MAX_VIEW_RANGE)

        # Use preset indices up to threshold index
        if any(endpoint):
            thr_idx = round(MAX_VIEW_RANGE - vec2_norm2(endpoint - player.pos))
            line_sight_indices = self.line_sight_indices[0][thr_idx:], self.line_sight_indices[1][thr_idx:]

        else:
            line_sight_indices = self.line_sight_indices

        draw_colour(window, line_sight_indices, self.colours['green'], opacity=0.1)

    def eval_effects(self, dt: float):
        """Iterate over all active effects and step them, clearing those that expire."""

        # Death "animation" buffer
        if self.observer_lock_time:
            self.observer_lock_time = max(0., self.observer_lock_time - dt)

        # Add ambient sound
        # NOTE: By interacting with audio channels before evaluating new effects,
        # the same channel should always be reserved
        if not self.audio_system._audio_channels[0]:
            own_player = self.session.players.get(self.own_player_id, None)

            if own_player is not None:
                self.audio_system.queue_sound(self.sounds['ambient'], own_player, own_player)

        expired_effects = [key_effect for key_effect in self.effects.items() if not key_effect[1].update(dt)]

        for key, effect in expired_effects:
            if effect.type == Effect.TYPE_COLOUR:
                self.fx_ctr_map[effect.world_indices] -= 1

            del self.effects[key]

    def clear_effects(self):
        """Reset buffer, canvas, and counter map."""

        self.effects.clear()
        self.fx_ctr_map.fill(0)

    def get_frame(self) -> ndarray:
        """Get the image corresponding to the current frame."""

        # Draw window content
        if not self.session.phase or self.view == GameID.VIEW_LOBBY or self.own_player_id not in self.session.players:
            window = self.lobby()

        else:
            window = self.main()

            if self.view == GameID.VIEW_MAPSTATS:
                window = self.mapstats(window)

            else:
                window = self.world(window)

                # Add view overlay according to observed player and view
                if self.view == GameID.VIEW_TERMS:
                    window[:108, 64:] = draw_overlay(window[:108, 64:], self.overlay_terms, opacity=0.65)

                    # Draw hovered term
                    self.draw_cursor_obs_from_code_view(window, self.code_view_terms)

                elif self.view == GameID.VIEW_ITEMS:
                    window[:108, 64:] = draw_overlay(window[:108, 64:], self.overlay_items, opacity=0.65)

                    # Draw hovered item
                    self.draw_cursor_obs_from_code_view(window, self.code_view_items)

                elif self.view == GameID.VIEW_STORE:
                    player: Player = self.session.players[self.own_player_id]

                    # Draw hovered item
                    if player.team == GameID.GROUP_TEAM_T:
                        window[:108, 64:] = draw_overlay(
                            window[:108, 64:], self.overlay_store_t, opacity=0.65)

                        self.draw_cursor_obs_from_code_view(window, self.code_view_store_t)

                    else:
                        window[:108, 64:] = draw_overlay(
                            window[:108, 64:], self.overlay_store_ct, opacity=0.65)

                        self.draw_cursor_obs_from_code_view(window, self.code_view_store_ct)

                # Own world view
                elif self.observed_player_id == self.own_player_id:
                    self.draw_cursor_obs_from_world_view(window)
                    self.draw_focus_line(window)

                # Draw observed player icon in spectated world view
                else:
                    observed_player: Player = self.session.players[self.observed_player_id]
                    draw_image(window, self.addressable_icons[observed_player.position_id], 112, 85)

        # Draw cursor
        # NOTE: Cursor values are enforced to lie within valid range when they are externally updated
        draw_image(window, self.icon_cursor, round(self.cursor_y)-2, round(self.cursor_x)-2)

        return window

    def lobby(self) -> ndarray:
        """Draw the console and players connected to the session."""

        window: ndarray = self.window_base_lobby.copy()
        null_char = self.characters['null']

        # Display game status
        draw_text(window, self.characters, null_char, 'in match' if self.session.phase else 'waiting to start', 2, 142)

        # Draw spectators
        for i, player in enumerate(self.session.spectators.values()):
            pos_y = 12 + i*8
            draw_number(window, self.digits, player.id, pos_y, 4)
            draw_image(window, self.addressable_icons[GameID.GROUP_SPECTATORS], pos_y - 3, 7)
            draw_text(window, self.characters, null_char, player.name, pos_y, 19)

        # Draw Ts
        for i, player in enumerate(self.session.players_t.values()):
            pos_y = 40 + i*8
            draw_number(window, self.digits, player.id, pos_y, 138)
            draw_image(window, self.addressable_icons[player.position_id], pos_y - 4, 144)
            draw_text(window, self.characters, null_char, player.name, pos_y, 108)

        # Draw CTs
        for i, player in enumerate(self.session.players_ct.values()):
            pos_y = 40 + i*8
            draw_number(window, self.digits, player.id, pos_y, 183)
            draw_image(window, self.addressable_icons[player.position_id], pos_y - 4, 164)
            draw_text(window, self.characters, null_char, player.name, pos_y, 190)

        # Draw console
        draw_text(window, self.characters, null_char, self.console_text, 128, 69, spacing=3)
        draw_image(window, self.icon_console_pointer, 124, 69 + min(len(self.console_text), 22)*8)

        return window

    def main(self) -> ndarray:
        """Draw the main HUD with information on current inventory, chat, and match state."""

        window = self.window_base_world.copy()
        digits = self.digits
        addressable_icons = self.addressable_icons

        player: Player = self.session.players[self.own_player_id]
        slots: list[Object | None] = player.slots

        # Draw equipped inventory
        for slot_idx, pos_y, pos_x in self.EQUIPMENT_DRAW_PARAMS:
            obj = slots[slot_idx]

            if obj is not None:
                draw_image(window, obj.item.icon, pos_y, pos_x)

        # Draw frame around held (selected) item
        held_object = player.held_object

        if held_object is not None:
            slot = held_object.item.slot
            subslot = held_object.item.subslot

            pos_y = 110 if slot != Item.SLOT_UTILITY else 126
            pos_x = 185 + 18 * ((slot-1) if slot != Item.SLOT_UTILITY else subslot)

            draw_image(window, self.icon_selected, pos_y, pos_x)

        # Display health, armour, money
        if player.health and player.team != GameID.GROUP_SPECTATORS:
            draw_number(window, digits, round(player.health), 137, 75)

        armour = slots[Item.SLOT_ARMOUR]

        if armour is not None:
            draw_number(window, digits, round(armour.durability), 137, 93)

        draw_number(window, digits, player.money, 120, 129)

        # Display magazine (or reloading/drawing or carried utility) and reserve
        if held_object is not None:
            if player.time_until_drawn or player.time_to_reload:
                draw_image(window, self.icon_reset, 129, 125)

            elif held_object.item.magazine_cap:
                draw_number(window, digits, held_object.magazine, 129, 129)

            elif held_object.item.id != GameID.ITEM_KNIFE and held_object.item.slot != Item.SLOT_OTHER:
                draw_number(window, digits, held_object.carrying, 129, 129)

            if held_object.item.reserve_cap:
                draw_number(window, digits, held_object.reserve, 137, 129)

        # Display match state
        session = self.session
        phase = session.phase
        current_round = session.rounds_won_t + session.rounds_won_ct + (1 if phase != GameID.PHASE_RESET else 0)

        draw_number(window, digits, session.rounds_won_t, 119, 150)
        draw_number(window, digits, session.rounds_won_ct, 119, 169)
        draw_number(window, digits, current_round, 130, 166)
        draw_number(window, digits, int(session.time // 60.), 138, 162)
        draw_number(window, digits, int(session.time % 60.), 138, 172, min_n_digits=2)

        if phase == GameID.PHASE_BUY:
            draw_image(window, self.icon_store, 134, 148)

        elif phase == GameID.PHASE_DEFUSE:
            draw_image(window, self.icon_c4, 130, 144)

        elif phase == GameID.PHASE_RESET:
            draw_image(window, self.icon_reset, 134, 148)

        # Display own icon
        if player.team == GameID.GROUP_SPECTATORS:
            draw_image(window, addressable_icons[GameID.GROUP_SPECTATORS], 116, 67)

        else:
            draw_image(window, addressable_icons[player.position_id], 116, 67)

        # Display chat
        # NOTE: The scroll wheel update enforces its range between `0` and `len(self.chat)-5`
        chat = self.chat[self.wheel_y:self.wheel_y+5]
        pos_y = 2
        pos_x = 4

        for message in chat:
            draw_number(window, digits, message.round, pos_y, pos_x)
            draw_number(window, digits, int(message.time // 60.), pos_y, pos_x + 12)
            draw_number(window, digits, int(message.time % 60.), pos_y, pos_x + 22, min_n_digits=2)
            draw_image(window, addressable_icons[message.position_id], pos_y - 3, pos_x + 45)

            for i, word in enumerate(message.words):
                if not word:
                    break

                draw_image(window, addressable_icons[word], pos_y + 8, pos_x - 3 + 16*i)

            pos_y += 24

        # Add scrollbar
        width = int(math.ceil(min(5, len(self.chat)) / max(len(self.chat), 1) * 63))
        offset = int(self.wheel_y / max(len(self.chat), 1) * 63)
        window[121:123, offset:offset+width] = 127

        # Display current message draft
        for i, (word, _, _) in enumerate(self.message_draft):
            draw_image(window, addressable_icons[word], 128, 1 + 16*i)

        return window

    def mapstats(self, window: ndarray) -> ndarray:
        """Draw player positions on the map besides player statistics."""

        window[:108, 64:] = self.overlay_mapstats
        digits = self.digits
        addressable_icons = self.addressable_icons

        player = self.session.players[self.observed_player_id]
        own_player = self.session.players.get(self.own_player_id, None)
        observer_team = GameID.GROUP_SPECTATORS if own_player is None else own_player.team

        # Get kills-to-deaths ratios and other stats to display
        stats_t = sorted((
            (a_player.kills / (a_player.deaths if a_player.deaths else 1.),
                a_player.position_id, a_player.health, a_player.money)
            for a_player in self.session.players.values() if a_player.team == GameID.GROUP_TEAM_T),
            reverse=True)

        stats_ct = sorted((
            (a_player.kills / (a_player.deaths if a_player.deaths else 1.),
                a_player.position_id, a_player.health, a_player.money)
            for a_player in self.session.players.values() if a_player.team == GameID.GROUP_TEAM_CT),
            reverse=True)

        # Draw stat table
        for i, (kdr, pos_id, health, money) in enumerate(stats_t):
            if not health:
                draw_image(window, self.icon_kill, 6 + i*9, 67)

            draw_image(window, addressable_icons[pos_id], 6 + i*9, 83)
            draw_number(window, digits, int(kdr), 10 + i*9, 104)
            draw_number(window, digits, int(10*(kdr - int(kdr))), 10 + i*9, 110)

            if observer_team != GameID.GROUP_TEAM_CT:
                draw_number(window, digits, money, 10 + i*9, 138)

        for i, (kdr, pos_id, health, money) in enumerate(stats_ct):
            if not health:
                draw_image(window, self.icon_kill, 60 + i*9, 67)

            draw_image(window, addressable_icons[pos_id], 61 + i*9, 83)
            draw_number(window, digits, int(kdr), 65 + i*9, 104)
            draw_number(window, digits, int(10*(kdr - int(kdr))), 65 + i*9, 110)

            if observer_team != GameID.GROUP_TEAM_T:
                draw_number(window, digits, money, 65 + i*9, 138)

        # Transform global coordinates into map view coordinates
        map_warp = MAP_WARP[self.map.id]

        player_pos_id = [
            (warp_position(map_warp, a_player.pos) + (64, 0), a_player.position_id, a_player.health)
            for a_player in self.session.players.values() if a_player.team == player.team]

        # Pings already in map view coordinates
        ping_pos_id = [
            ((effect.pos_x, effect.pos_y), effect.associated_id, effect.cover_indices, effect.opacity)
            for effect in self.effects.values() if effect.type == Effect.TYPE_MARK]

        # Draw player locations for own team
        # NOTE: "Smooth" outlines differentiate between player positions and ping marks
        # (dead player marks have black outline, observed player has white, others go by their colours)
        for (pos_x, pos_y), pos_id, health in player_pos_id:
            pos_y = round(pos_y)
            pos_x = round(pos_x)
            window[pos_y, pos_x] = self.player_colours[pos_id]

            ring1_indices = self.RING1_INDICES[0] + pos_y, self.RING1_INDICES[1] + pos_x
            ring2_indices = self.RING2_INDICES[0] + pos_y, self.RING2_INDICES[1] + pos_x

            if not health:
                window[ring1_indices] = np.uint8(self.other_colours['dead'] * 0.6 + window[ring1_indices] * 0.4)
                window[ring2_indices] = np.uint8(self.other_colours['dead'] * 0.3 + window[ring2_indices] * 0.7)

            elif pos_id == player.position_id:
                window[ring1_indices] = np.uint8(self.other_colours['self'] * 0.6 + window[ring1_indices] * 0.4)
                window[ring2_indices] = np.uint8(self.other_colours['self'] * 0.3 + window[ring2_indices] * 0.7)

            else:
                window[ring1_indices] = np.uint8(self.player_colours[pos_id] * 0.6 + window[ring1_indices] * 0.4)
                window[ring2_indices] = np.uint8(self.player_colours[pos_id] * 0.3 + window[ring2_indices] * 0.7)

        # Display pings for own team
        # NOTE: The other team's pings should not be received at all
        for (pos_x, pos_y), pos_id, cover_indices, fading in ping_pos_id:
            pos_y = round(pos_y)
            pos_x = round(pos_x)
            colour = self.player_colours[pos_id - 120]
            draw_colour(window, cover_indices, colour, opacity=fading, pos_y=pos_y, pos_x=pos_x)

        # Get cursor obs
        if self.cursor_x >= 84.:

            # Iter over players, get argmin wrt. cursor position
            if player_pos_id:
                closest_player = min(
                    (vec2_norm2(pos - (self.cursor_x, self.cursor_y)), pos_id)
                    for pos, pos_id, _ in player_pos_id)
            else:
                closest_player = (np.inf, None)

            # Iter over pings, get argmin wrt. cursor position
            if ping_pos_id:
                closest_ping = min(
                    (vec2_norm2(np.array(pos) - (self.cursor_x, self.cursor_y)), pos_id)
                    for pos, pos_id, _, _ in ping_pos_id)
            else:
                closest_ping = (np.inf, None)

            # Get argmin of both
            closest_dist, closest_id = min(closest_player, closest_ping)

            if closest_dist < 3.:
                draw_image(window, addressable_icons[closest_id], 112, 85)

        return window

    def world(self, window: ndarray) -> ndarray:
        """Draw the in-game world with players, objects, and effects in line of sight."""

        players = self.session.players
        player = players[self.observed_player_id]
        map_ = self.map
        world_map = map_.world

        # Get recoil-affected position and angle
        # NOTE: Recoil (moving viewpoint/origin) can cause the own sprite to
        # not always be drawn at the same (expected) position
        # (but it looks natural enough if only negative position offsets are used)
        pos = tuple(player.pos)
        origin = tuple(player.d_pos_recoil + self.WORLD_FRAME_ORIGIN)
        angle = player.angle + F_PI2 + player.d_angle_recoil

        # Transform world into local frame wrt. observed entity
        world = project_into_view(world_map, pos, angle, origin, self.WORLD_FRAME_SIZE)

        # To transform inhabiting effects, objects, and players, their positions need to be warped, as well
        world_warp = get_camera_warp(pos, angle, origin)
        inv_warp = get_inverse_warp(pos, angle, origin)

        # Hide information from dead observed player
        observed_alive = player.health or self.session.is_spectator(self.own_player_id)

        # Iter over colour effects
        if observed_alive:
            map_bounds = map_.bounds

            for effect in self.effects.values():
                if effect.type == Effect.TYPE_COLOUR:
                    draw_colour(
                        self.fx_canvas, effect.world_indices, effect.colour, effect.opacity,
                        bounds=map_bounds, background=world_map)

        # Draw effects and render view mask
        # Differentiate between normal and scoped endpoints wrt. held item
        endpoints = (
            self.scoped_fov_endpoints
            if player.held_object is not None and player.held_object.item.scoped
            else self.standard_fov_endpoints)

        view_mask = mask_view(
            player.id if observed_alive else MapID.PLAYER_ID_NULL,
            map_.code_map, map_.id_map, self.fx_canvas, world, inv_warp, endpoints)

        # Draw sprites etc.
        # NOTE: 1.03125 is used (instead of 1.) to round down the observed player position after warp to proper place

        # Draw dead players
        for a_player in players.values():
            if not a_player.health and self.check_los(player, a_player):
                pos_x, pos_y = warp_position(world_warp, a_player.pos)
                pos_x, pos_y = round(pos_x - 1.03125), round(pos_y - 1.03125)
                sprite = self.sprites[a_player.team, -1]

                draw_colour(
                    world, self.SPRITE_AURA_INDICES, self.player_colours[a_player.position_id], opacity=0.3,
                    pos_y=pos_y, pos_x=pos_x, bounds=self.WORLD_BOUNDS)
                draw_image(world, sprite, pos_y, pos_x)

        # Draw persistent objects (with infinite lifetime)
        for an_object in self.session.objects.values():
            if an_object.lifetime == np.inf and self.check_los(player, an_object):
                pos_x, pos_y = warp_position(world_warp, an_object.pos)
                pos_x, pos_y = round(pos_x), round(pos_y)

                if 0 <= pos_y <= 107 and 0 <= pos_x <= 191:
                    world[pos_y, pos_x] = self.other_colours['obj_reg']

                    # Emphasise C4
                    if 1 <= pos_y <= 106 and 1 <= pos_x <= 190 and an_object.item.id == GameID.ITEM_C4:
                        ring1_indices = self.RING1_INDICES[0] + pos_y, self.RING1_INDICES[1] + pos_x
                        ring2_indices = self.RING2_INDICES[0] + pos_y, self.RING2_INDICES[1] + pos_x

                        world[ring1_indices] = np.uint8(self.other_colours['obj_reg']*0.6 + world[ring1_indices]*0.4)
                        world[ring2_indices] = np.uint8(self.other_colours['obj_reg']*0.3 + world[ring2_indices]*0.7)

        # Draw alive players
        for a_player in players.values():
            if a_player.health and self.check_los(player, a_player):
                pos_x, pos_y = warp_position(world_warp, a_player.pos)
                pos_x, pos_y = round(pos_x - 1.03125), round(pos_y - 1.03125)
                angle = get_relative_sprite_index(player.angle, a_player.angle)
                sprite = self.sprites[a_player.team, angle]

                draw_image(world, sprite, pos_y, pos_x)
                draw_colour(
                    world, self.SPRITE_CAP_INDICES, self.player_colours[a_player.position_id], opacity=0.75,
                    pos_y=pos_y, pos_x=pos_x, bounds=self.WORLD_BOUNDS)

        # Draw transient objects (with finite lifetime)
        for an_object in self.session.objects.values():
            if an_object.lifetime != np.inf and self.check_los(player, an_object):
                pos_x, pos_y = warp_position(world_warp, an_object.pos)
                pos_x, pos_y = round(pos_x), round(pos_y)

                if 0 <= pos_y <= 107 and 0 <= pos_x <= 191:
                    world[pos_y, pos_x] = self.other_colours['obj_fuse']

                    # Emphasise C4
                    if 1 <= pos_y <= 106 and 1 <= pos_x <= 190 and an_object.item.id == GameID.ITEM_C4:
                        ring1_indices = self.RING1_INDICES[0] + pos_y, self.RING1_INDICES[1] + pos_x
                        ring2_indices = self.RING2_INDICES[0] + pos_y, self.RING2_INDICES[1] + pos_x

                        world[ring1_indices] = np.uint8(self.other_colours['obj_fuse']*0.6 + world[ring1_indices]*0.4)
                        world[ring2_indices] = np.uint8(self.other_colours['obj_fuse']*0.3 + world[ring2_indices]*0.7)

        # Needed for residual effects
        self.last_world_frame = world

        # Mask with visibility factors
        world = (view_mask[..., None] * world.astype(np.float32)).astype(np.uint8)

        # Draw overlays
        if observed_alive:
            for effect in self.effects.values():
                if effect.type == Effect.TYPE_OVERLAY:
                    world = draw_overlay(world, effect.overlay, effect.opacity)

        # Draw muted if own player is dead
        own_player = player if self.observed_player_id == self.own_player_id else players[self.own_player_id]

        if not own_player.health and own_player.team != GameID.GROUP_SPECTATORS:
            world = draw_muted(world)

        window[:108, 64:] = world

        return window

    def check_los(
        self,
        observer: Player,
        entity: Object | Player,
        ignore_zone: bool = False
    ) -> bool:
        """
        Confirm line of sight from the observing player to an entity
        by tracing a ray between the points until succeeding or reaching
        an obstruction, such as a wall, a smoke zone, or another player.
        """

        # Can't see spectators
        if isinstance(entity, Player) and entity.team == GameID.GROUP_SPECTATORS:
            return False

        # Can see oneself
        if observer.id == entity.id:
            return True

        # Can't see if observer entity dead and own entity isn't a spectator
        if not observer.health and not self.session.is_spectator(self.own_player_id):
            return False

        observer_pos = observer.pos
        entity_pos = entity.pos

        # Check distance
        if vec2_norm2(observer_pos - entity_pos) > MAX_VIEW_RANGE:
            return False

        # Check angle
        relative_x, relative_y = vec2_rot_(entity_pos - observer_pos, -observer.angle)
        relative_angle = math.atan2(-relative_y, relative_x)

        if abs(relative_angle) > (self.ANGLE_LIM_SCOPED if observer.held_object.item.scoped else self.ANGLE_LIM_STD):
            return False

        map_ = self.map
        zone_map = map_.zone_null if ignore_zone else map_.zone

        pos_x, pos_y = trace_sight(observer.id, observer_pos, entity_pos, map_.wall, zone_map, map_.player_id_null)

        if pos_x or pos_y:
            return self.map.player_id[round(pos_y), round(pos_x)] == entity.id

        return True

    def add_chat_entry(self, message: Message):
        """Add message to chat history and increment the scroll wheel (unless viewing older messages)."""

        if len(self.chat) >= 5 and self.wheel_y >= (len(self.chat) - 5):
            self.wheel_y += 1

        self.chat.append(message)

    def add_effect(self, effect: Effect):
        """Give the effect an identifer and add it to tracked effects."""

        effect_id = (max(self.effects.keys()) + 1) if self.effects else 0
        self.effects[effect_id] = effect

        if effect.type == Effect.TYPE_COLOUR:
            self.fx_ctr_map[effect.world_indices] += 1

    def create_log(self, eval_type: int) -> list[int | float] | None:
        """
        Evaluate player action wrt. given evaluation type, cursor position,
        and view. The evaluation can result in the generation of a (local) log,
        e.g. when attempting to purchase an item or send a message.
        """

        player: Player = self.session.players.get(self.own_player_id, None)

        # Spectators can't chat or buy
        if player is None or player.team == GameID.GROUP_SPECTATORS:
            return None

        if eval_type == GameID.EVAL_BUY:
            if player.team == GameID.GROUP_TEAM_T:
                hover_id = self.get_cursor_obs_from_code_view(self.code_view_store_t)

            else:
                hover_id = self.get_cursor_obs_from_code_view(self.code_view_store_ct)

            if hover_id:
                if player.money >= self.inventory.get_item_by_id(hover_id).price or player.dev_mode:
                    self.audio_system.queue_sound(self.sounds['buy'], player, player)

                else:
                    self.audio_system.queue_sound(self.sounds['no_buy'], player, player)

                return [self.own_player_id, GameID.LOG_BUY, hover_id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.]

        # Add term id to message buffer, obtained from hover id
        elif eval_type == GameID.EVAL_MSG_TERM:
            if len(self.message_draft) < 4:
                hover_id = self.get_cursor_obs_from_code_view(self.code_view_terms)

                if hover_id:
                    self.message_draft.append((hover_id, 0, 0))
                    self.audio_system.queue_sound(self.sounds['word_added'], player, player)

        elif eval_type == GameID.EVAL_MSG_ITEM:
            if len(self.message_draft) < 4:
                hover_id = self.get_cursor_obs_from_code_view(self.code_view_items)

                if hover_id:
                    self.message_draft.append((hover_id, 0, 0))
                    self.audio_system.queue_sound(self.sounds['word_added'], player, player)

        # Add ping/mark to message buffer, obtained from cursor indices
        elif eval_type == GameID.EVAL_MSG_MARK:
            if len(self.message_draft) < 4 and self.cursor_x > 144.:
                # NOTE: 120 moves e.g. PLAYER_T1 to MARK_T1
                self.message_draft.append((player.position_id + 120, round(self.cursor_x), round(self.cursor_y)))
                self.audio_system.queue_sound(self.sounds['word_added'], player, player)

        # Send characters from message buffer and clear it
        elif eval_type == GameID.EVAL_MSG_SEND:
            if not self.message_draft:
                return None

            msg = [self.own_player_id, GameID.LOG_MESSAGE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.]

            for i in range(len(self.message_draft)):
                word, pos_x, pos_y = self.message_draft[i]
                msg[2+i] = word
                msg[6+2*i] = int(pos_x)
                msg[6+2*i+1] = int(pos_y)

            self.message_draft.clear()
            self.audio_system.queue_sound(self.sounds['msg_sent'], player, player)

            return msg

        return None

    def clear_message_draft(self):
        """Clear message draft and emit SFX."""
        if not self.message_draft:
            return

        player: Player = self.session.players[self.own_player_id]
        self.message_draft.clear()
        self.audio_system.queue_sound(self.sounds['msg_deleted'], player, player)

    def enter_world(self):
        """Set observed player etc. upon entering the game world."""

        if not self.session.phase:
            return

        if self.own_player_id in self.session.players_t or self.own_player_id in self.session.players_ct:
            self.observed_player_id = self.own_player_id
            self.view = GameID.VIEW_WORLD
            self.cursor_x = 159.5

        elif self.session.players_t or self.session.players_ct:
            self.view = GameID.VIEW_WORLD
            self.change_observed_player()

        else:
            self.exit_world()

    def exit_world(self):
        """Set observed player etc. upon exiting the game world."""

        self.observed_player_id = self.own_player_id
        self.view = GameID.VIEW_LOBBY

    def change_observed_player(self, rotate_upward: bool = None):
        """
        Determine the next observed player after the currently observed one
        was rotated, changed teams, or was disconnected, based on the
        specified direction.
        """

        own_player: Player = self.session.players.get(self.own_player_id, None)

        # If unable to infer player pools, return to lobby
        if own_player is None:
            self.exit_world()
            return

        # Must be spectator or dead
        if own_player.team != GameID.GROUP_SPECTATORS and own_player.health:
            return

        # Death "animation" buffer
        if self.observer_lock_time:
            return

        # Limit observable pool
        pool_t = tuple(self.session.players_t.keys())
        pool_ct = tuple(self.session.players_ct.keys())

        if own_player.team == GameID.GROUP_TEAM_T:
            player_pool = pool_t
        elif own_player.team == GameID.GROUP_TEAM_CT:
            player_pool = pool_ct
        else:
            player_pool = pool_t + pool_ct

        # If no eligible players to observe, return to lobby
        if not player_pool:
            self.exit_world()
            return

        # Select next observed player based on specified direction
        if self.observed_player_id not in player_pool or rotate_upward is None:
            next_idx = 0
        else:
            next_idx = player_pool.index(self.observed_player_id) + (1 if rotate_upward else -1)

            if next_idx >= len(player_pool):
                next_idx = 0

        self.observed_player_id = player_pool[next_idx]


@jit('float64(float64, float64)', nopython=True, nogil=True, cache=True)
def get_relative_sprite_index(observer_angle: float, player_angle: float) -> float:
    """Get index from 0 to (incl.) 15 corresponding to a sprite with the angle as seen by the observer."""

    rel_angle = fix_angle_range(player_angle - observer_angle)

    if rel_angle < 0.:
        rel_angle += F_2PI

    rel_angle *= 16. / F_2PI

    return round(rel_angle) if rel_angle < 15.5 else 0
