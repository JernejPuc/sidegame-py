"""Main and supporting methods for client-side world simulation of SDG"""

import os
from collections import deque
from typing import Deque, Dict, List, Tuple, Union
import numpy as np
import cv2
from sidegame.ext import sdglib
from sidegame.audio import PlanarAudioSystem as AudioSystem
from sidegame.effects import Effect
from sidegame.graphics import draw_image, draw_text, draw_number, draw_colour, draw_overlay, draw_muted, \
    get_camera_warp, project_into_view, get_view_endpoints, render_view
from sidegame.game.shared import ASSET_DIR, GameID, Map, Message, Item, Inventory, Object, Player, Session


class Simulation:
    """Generates and holds the state of the game world from the client's perspective."""

    AUDIO_MAX_DISTANCE = 108.
    AUDIO_DISTANCE_SCALING = 512./108.
    AUDIO_BASE_VOLUME = 0.25

    WORLD_FRAME_CENTRE = (108./2. + 1., 64. + 192./2.)  # Shift by 1px; centre in terms/items/store images is wrong
    WORLD_FRAME_ORIGIN = (95.5, 106.5)
    WORLD_FRAME_SIZE = (192, 108)
    WORLD_BOUNDS = WORLD_FRAME_SIZE[::-1]

    COLOUR_BLACK = np.array([0, 0, 0], dtype=np.uint8)
    COLOUR_WHITE = np.array([255, 255, 255], dtype=np.uint8)
    COLOUR_RED = np.array([0, 0, 255], dtype=np.uint8)
    COLOUR_GREEN = np.array([0, 255, 0], dtype=np.uint8)
    COLOUR_BLUE = np.array([255, 0, 0], dtype=np.uint8)
    COLOUR_ITEM_INF = np.array([140, 80, 140], dtype=np.uint8)
    COLOUR_ITEM_FUSE = np.array([127, 127, 191], dtype=np.uint8)

    RING1_INDICES = (np.array([-1, 0, 0, 1], dtype=np.int16), np.array([0, -1, 1, 0], dtype=np.int16))
    RING2_INDICES = (np.array([-1, -1, 1, 1], dtype=np.int16), np.array([-1, 1, -1, 1], dtype=np.int16))

    # Mapping between world pos and map view pos
    # y: 535 to x: 90, y: 103 to x: 185 || x: 20 to y: 1, x: 603 to y: 106
    MAP_WARP = np.array([[0, -95./432., 185. + 103.*95./432.], [105./583., 0, 1. - 105.*20./583.]])

    FOV_MAIN = 106.3
    FOV_SCOPED = 25.

    CONSOLE_COMMANDS: Dict[str, int] = {
        'start': GameID.CMD_START_MATCH,
        'stop': GameID.CMD_END_MATCH,
        'quit': GameID.CMD_END_SESSION,
        'set team': GameID.CMD_SET_TEAM,
        'set name': GameID.CMD_SET_NAME,
        'set role': GameID.CMD_SET_ROLE,
        'ping': GameID.CMD_GET_LATENCY,
        'rundown': GameID.CHEAT_END_ROUND,
        'global buy': GameID.CHEAT_GLOBAL_BUY,
        'max money': GameID.CHEAT_MAX_MONEY}

    DEFAULT_OBS = (np.Inf, Map.PLAYER_ID_NULL, GameID.NULL)

    def __init__(self, tick_rate: float, volume: float, own_player_id: int, rng: np.random.Generator = None):
        self.audio_system = AudioSystem(
            step_freq=tick_rate,
            max_distance=self.AUDIO_MAX_DISTANCE,
            distance_scaling=self.AUDIO_DISTANCE_SCALING,
            load_attenuation=self.AUDIO_BASE_VOLUME,
            volume=volume)

        self.effects: Dict[int, Effect] = {}
        self.inventory = Inventory(self.load_image, self.load_sound)
        self.session = Session(rng=rng)
        self.own_player_id = own_player_id
        self.observed_player_id = own_player_id

        self.cursor_y, self.cursor_x = self.WORLD_FRAME_CENTRE
        self.wheel_y = 0
        self.console_text = ''
        self.message_draft: Deque[Tuple[int, float]] = deque()
        self.chat: List[Message] = []
        self.view: int = GameID.VIEW_LOBBY

        # View-related
        self.window_base_lobby = self.load_image('views', 'lobby.png')
        self.window_base_world = self.load_image('views', 'main.png')
        self.overlay_mapstats = self.load_image('views', 'mapstats.png')
        self.overlay_terms = self.load_image('views', 'terms.png')
        self.overlay_items = self.load_image('views', 'items.png')
        self.overlay_store_t = self.load_image('views', 'store_t.png')
        self.overlay_store_ct = self.load_image('views', 'store_ct.png')

        self.code_view_terms = self.load_image('views', 'code_terms.png', mono=True)
        self.code_view_items = self.load_image('views', 'code_items.png', mono=True)
        self.code_view_store_t = self.load_image('views', 'code_store_t.png', mono=True)
        self.code_view_store_ct = self.load_image('views', 'code_store_ct.png', mono=True)

        self.last_world_frame: np.ndarray = None
        self.null_rot_entity_map = np.empty(self.WORLD_BOUNDS, dtype=np.int16)
        self.null_rot_entity_map.fill(Map.PLAYER_ID_NULL)
        self.null_rot_zone_map = np.zeros(self.WORLD_BOUNDS, dtype=np.uint8)

        self.line_sight_indices = (np.arange(0, 106), np.repeat(64 + 96, 106))
        self.standard_fov_endpoints = get_view_endpoints(self.FOV_MAIN, 108.)
        self.scoped_fov_endpoints = get_view_endpoints(self.FOV_SCOPED, 108.)

        # Soundbank
        self.sounds: Dict[str, List[np.ndarray]] = {
            'clip_low': self.load_sound('sounds', 'general', 'lowammo_01.wav'),
            'clip_empty': self.load_sound('sounds', 'general', 'clipempty_rifle.wav'),
            'msg_sent': self.load_sound('sounds', 'general', 'playerping.wav', base_volume=0.5),
            'msg_deleted': self.load_sound('sounds', 'general', 'menu_accept.wav', base_volume=0.5),
            'word_added': self.load_sound('sounds', 'general', 'counter_beep.wav', base_volume=0.5),
            'msg_received': self.load_sound('sounds', 'general', 'lobby_notification_chat.wav', base_volume=0.5),
            'mark_received': self.load_sound('sounds', 'general', 'ping_alert_01.wav', base_volume=0.5),
            'reset_round': self.load_sound('sounds', 'general', 'pl_respawn.wav', base_volume=0.5),
            'reset_side': self.load_sound('sounds', 'general', 'bonus_alert_start.wav', base_volume=0.5),
            'planted': self.load_sound('sounds', 'general', 'bombpl_mod.wav', base_volume=0.5),
            'defused': self.load_sound('sounds', 'general', 'bombdef_mod.wav', base_volume=0.5),
            'ct_win': self.load_sound('sounds', 'general', 'ctwin_mod.wav', base_volume=0.5),
            't_win': self.load_sound('sounds', 'general', 'terwin_mod.wav', base_volume=0.5),
            'get': self.load_sound('sounds', 'general', 'pickup_weapon_01.wav'),
            'drop': self.load_sound('sounds', 'grenades', 'grenade_throw.wav'),
            'buy': self.load_sound('sounds', 'general', 'radial_menu_buy_02.wav'),
            'no_buy': self.load_sound('sounds', 'general', 'weapon_cant_buy.wav'),
            'death': self.load_sound('sounds', 'player', 'death1.wav', base_volume=0.5),
            'hit': self.load_sound('sounds', 'player', 'kevlar5.wav'),
            'sine_max': self.load_sound('sounds', 'grenades', 'flashbang_sine1_new.wav'),
            'sine_mid': self.load_sound('sounds', 'grenades', 'flashbang_sine2_new.wav'),
            'sine_min': self.load_sound('sounds', 'grenades', 'flashbang_sine3_new.wav')}

        self.movements: List[List[np.ndarray]] = [
            self.load_sound('sounds', 'player', 'movement1.wav', base_volume=0.125),
            self.load_sound('sounds', 'player', 'movement2.wav', base_volume=0.125),
            self.load_sound('sounds', 'player', 'movement3.wav', base_volume=0.125)]

        self.keypresses: List[List[np.ndarray]] = [self.inventory.c4.sounds[f'press{i}'] for i in range(1, 8)]
        self.footsteps: Dict[int, List[List[np.ndarray]]] = {}

        terrain_keys = [Map.SOUND_CONCRETE, Map.SOUND_DIRT, Map.SOUND_WOOD, Map.SOUND_METAL, Map.SOUND_TILE]
        terrains = ['concrete', 'dirt', 'wood', 'metal', 'tile']

        for terrain_key, terrain in zip(terrain_keys, terrains):
            terrain_path = os.path.join(ASSET_DIR, 'sounds', 'player', terrain)

            self.footsteps[terrain_key] = [
                self.load_sound('sounds', 'player', terrain, soundfile, base_volume=0.125)
                for soundfile in os.listdir(terrain_path)]

        # Specific icons
        self.icon_console_pointer = self.load_image('icons', 'pointer_console.png')
        self.icon_cursor = self.load_image('icons', 'pointer_cursor.png')
        self.icon_selected = self.load_image('icons', 'pointer_item.png')
        self.icon_reset = self.load_image('icons', 'phase_reset.png')
        self.icon_store = self.load_image('icons', 'phase_buy.png')
        self.icon_kill = self.load_image('icons', 'term_kill.png')

        self.addressable_icons: Dict[int, np.ndarray] = {
            GameID.NULL: self.load_image('icons', 'team_spectator.png'),
            GameID.PLAYER_T1: self.load_image('icons', 'agent_t_0.png'),
            GameID.PLAYER_T2: self.load_image('icons', 'agent_t_1.png'),
            GameID.PLAYER_T3: self.load_image('icons', 'agent_t_2.png'),
            GameID.PLAYER_T4: self.load_image('icons', 'agent_t_3.png'),
            GameID.PLAYER_T5: self.load_image('icons', 'agent_t_4.png'),
            GameID.PLAYER_CT1: self.load_image('icons', 'agent_ct_0.png'),
            GameID.PLAYER_CT2: self.load_image('icons', 'agent_ct_1.png'),
            GameID.PLAYER_CT3: self.load_image('icons', 'agent_ct_2.png'),
            GameID.PLAYER_CT4: self.load_image('icons', 'agent_ct_3.png'),
            GameID.PLAYER_CT5: self.load_image('icons', 'agent_ct_4.png'),
            GameID.MARK_T1: self.load_image('icons', 'ping_t_0.png'),
            GameID.MARK_T2: self.load_image('icons', 'ping_t_1.png'),
            GameID.MARK_T3: self.load_image('icons', 'ping_t_2.png'),
            GameID.MARK_T4: self.load_image('icons', 'ping_t_3.png'),
            GameID.MARK_T5: self.load_image('icons', 'ping_t_4.png'),
            GameID.MARK_CT1: self.load_image('icons', 'ping_ct_0.png'),
            GameID.MARK_CT2: self.load_image('icons', 'ping_ct_1.png'),
            GameID.MARK_CT3: self.load_image('icons', 'ping_ct_2.png'),
            GameID.MARK_CT4: self.load_image('icons', 'ping_ct_3.png'),
            GameID.MARK_CT5: self.load_image('icons', 'ping_ct_4.png'),
            GameID.TERM_MOVE: self.load_image('icons', 'term_move.png'),
            GameID.TERM_HOLD: self.load_image('icons', 'term_hold.png'),
            GameID.TERM_SEE: self.load_image('icons', 'term_see.png'),
            GameID.TERM_STOP: self.load_image('icons', 'term_fullstop.png'),
            GameID.TERM_EXCLAME: self.load_image('icons', 'term_exclamation.png'),
            GameID.TERM_ASK: self.load_image('icons', 'term_question.png'),
            GameID.TERM_KILL: self.icon_kill,
            GameID.ITEM_ARMOUR: self.inventory.armour.icon,
            GameID.ITEM_RIFLE_T: self.inventory.rifle_t.icon,
            GameID.ITEM_RIFLE_CT: self.inventory.rifle_ct.icon,
            GameID.ITEM_SMG_T: self.inventory.smg_t.icon,
            GameID.ITEM_SMG_CT: self.inventory.smg_ct.icon,
            GameID.ITEM_SHOTGUN_T: self.inventory.shotgun_t.icon,
            GameID.ITEM_SHOTGUN_CT: self.inventory.shotgun_ct.icon,
            GameID.ITEM_SNIPER: self.inventory.sniper.icon,
            GameID.ITEM_PISTOL_T: self.inventory.pistol_t.icon,
            GameID.ITEM_PISTOL_CT: self.inventory.pistol_ct.icon,
            GameID.ITEM_KNIFE: self.inventory.knife.icon,
            GameID.ITEM_DKIT: self.inventory.dkit.icon,
            GameID.ITEM_C4: self.inventory.c4.icon,
            GameID.ITEM_FLASH: self.inventory.flash.icon,
            GameID.ITEM_EXPLOSIVE: self.inventory.explosive.icon,
            GameID.ITEM_INCENDIARY_T: self.inventory.incendiary_t.icon,
            GameID.ITEM_INCENDIARY_CT: self.inventory.incendiary_ct.icon,
            GameID.ITEM_SMOKE: self.inventory.smoke.icon,
            GameID.GROUP_TEAM_T: self.load_image('icons', 'team_t.png'),
            GameID.GROUP_TEAM_CT: self.load_image('icons', 'team_ct.png')}

        # Associate colours with player position ids
        self.colours: Dict[int, np.ndarray] = {
            'dead': np.array([0, 0, 0], dtype=np.uint8),
            'self': np.array([255, 255, 255], dtype=np.uint8),
            GameID.PLAYER_T1: np.array([160, 64, 160], dtype=np.uint8),
            GameID.PLAYER_T2: np.array([16, 192, 255], dtype=np.uint8),
            GameID.PLAYER_T3: np.array([160, 96, 255], dtype=np.uint8),
            GameID.PLAYER_T4: np.array([32, 32, 160], dtype=np.uint8),
            GameID.PLAYER_T5: np.array([96, 128, 160], dtype=np.uint8),
            GameID.PLAYER_CT1: np.array([32, 224, 192], dtype=np.uint8),
            GameID.PLAYER_CT2: np.array([224, 192, 32], dtype=np.uint8),
            GameID.PLAYER_CT3: np.array([32, 224, 16], dtype=np.uint8),
            GameID.PLAYER_CT4: np.array([224, 96, 96], dtype=np.uint8),
            GameID.PLAYER_CT5: np.array([224, 196, 160], dtype=np.uint8)}

        # Team and angle specific sprites
        self.sprites: Dict[Tuple[int, int], np.ndarray] = {}

        for i in range(16):
            self.sprites[(GameID.GROUP_TEAM_T, i)] = self.load_image('sprites', f't_{i}.png')
            self.sprites[(GameID.GROUP_TEAM_CT, i)] = self.load_image('sprites', f'ct_{i}.png')

        self.sprites[(GameID.GROUP_TEAM_T, -1)] = self.load_image('sprites', 't_dead.png')
        self.sprites[(GameID.GROUP_TEAM_CT, -1)] = self.load_image('sprites', 'ct_dead.png')

    def load_image(self, *image_path: Union[str, Tuple[str]], mono: bool = False) -> np.ndarray:
        """Wrapper around `cv2.imread` to minimise path specification."""
        return cv2.imread(
            os.path.join(ASSET_DIR, *image_path), flags=cv2.IMREAD_GRAYSCALE if mono else cv2.IMREAD_UNCHANGED)

    def load_sound(self, *sound_path: Union[str, Tuple[str]], base_volume: float = None) -> List[np.ndarray]:
        """Wrapper around `audio::PlanarAudioSystem.load_sound` to minimise path specification."""
        return self.audio_system.load_sound(os.path.join(ASSET_DIR, *sound_path), base_volume=base_volume)

    def get_cursor_obs_from_code_view(self, code_view: np.ndarray) -> int:
        """Get the ID corresponding to the position above which the cursor is hovering."""
        return code_view[round(self.cursor_y), round(self.cursor_x)-64]

    def draw_cursor_obs_from_code_view(self, window: np.ndarray, code_view: np.ndarray):
        """Draw the icon of the observed term or item lying under the cursor."""
        obs_id = self.get_cursor_obs_from_code_view(code_view)

        if obs_id:
            draw_image(window, self.addressable_icons[obs_id], 112, 85)

    def get_cursor_obs_from_world_view(self) -> Tuple[float, int, int]:
        """
        Get the IDs corresponding to the entity above which the cursor is
        roughly hovering (along with the distance to the cursor).
        """

        player: Player = self.session.players.get(self.own_player_id, None)

        if player is None or not player.health:
            return self.DEFAULT_OBS

        # Transform cursor to world frame (rather than the other way around)
        cursor_dist = Player.MAX_VIEW_RANGE - self.cursor_y
        cursor_pos = player.pos + np.array([np.cos(player.angle), np.sin(player.angle)]) * cursor_dist

        # Iter over players, get argmin wrt. cursor position
        closest_player_obs = min((
            (np.linalg.norm(a_player.pos - cursor_pos), a_player.id, a_player.position_id)
            for a_player in self.session.players.values() if a_player.team != GameID.GROUP_SPECTATORS
            and self.check_los(player, a_player)),
            default=self.DEFAULT_OBS)

        # Iter over objects, get argmin wrt. cursor position
        # NOTE: For the sake of gameplay, C4 can be seen through smoke zones
        closest_object_obs = min((
            (np.linalg.norm(an_object.pos - cursor_pos), an_object.id, an_object.item.id)
            for an_object in self.session.objects.values()
            if (an_object.lifetime == np.Inf and self.check_los(player, an_object))
            or (an_object.item.id == GameID.ITEM_C4 and self.check_los(player, an_object, ignore_zone=True))),
            default=self.DEFAULT_OBS)

        # Get closest from both players and objects
        closest_obs = min(closest_player_obs, closest_object_obs)
        closest_dist = closest_obs[0]

        # Check if distance to cursor is valid
        return self.DEFAULT_OBS if closest_dist > 3. else closest_obs

    def draw_cursor_obs_from_world_view(self, window: np.ndarray):
        """Draw the icon of the observed entity lying roughly under the cursor."""
        _, _, closest_id = self.get_cursor_obs_from_world_view()

        if closest_id:
            draw_image(window, self.addressable_icons[closest_id], 112, 85)

    def draw_focus_line(self, window: np.ndarray):
        """Trace a green ray to the farthest unobscured point directly in front of the player."""
        player: Player = self.session.players[self.own_player_id]
        map_: Map = self.session.map

        if not player.health:
            return

        endpoint = player.get_focal_point(map_.height, map_.player_id, map_.zone, max_range=Player.MAX_VIEW_RANGE)

        # Use preset indices up to threshold index
        if any(endpoint):
            thr_idx = round(Player.MAX_VIEW_RANGE - np.linalg.norm(endpoint - player.pos))
            line_sight_indices = self.line_sight_indices[0][thr_idx:], self.line_sight_indices[1][thr_idx:]

        else:
            line_sight_indices = self.line_sight_indices

        draw_colour(window, line_sight_indices, self.COLOUR_GREEN, opacity=0.1)

    def draw_carried_item(self, window: np.ndarray, obj: Object, pos_y: int, pos_x: int):
        """Draw the item icon of a carried object."""
        if obj is not None:
            draw_image(window, obj.item.icon, pos_y, pos_x)

    def get_relative_sprite_index(self, observer: Player, player: Player) -> Tuple[float]:
        """Get index from 0 to (incl.) 15 corresponding to a sprite with the angle as seen by the observer."""

        rel_angle = player.fix_angle_range(player.angle - observer.angle)

        if rel_angle < 0.:
            rel_angle += Player.F_2PI

        rel_angle *= 16. / Player.F_2PI

        return round(rel_angle) if rel_angle < 15.5 else 0

    def eval_effects(self, dt: float):
        """Iterate over all active effects and step them, clearing those that expire."""
        expired_effect_keys = [effect_key for effect_key, effect in self.effects.items() if not effect.update(dt)]

        for effect_key in expired_effect_keys:
            del self.effects[effect_key]

    def get_frame(self) -> np.ndarray:
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
                        window[:108, 64:] = draw_overlay(window[:108, 64:], self.overlay_store_t, opacity=0.65)
                        self.draw_cursor_obs_from_code_view(window, self.code_view_store_t)

                    else:
                        window[:108, 64:] = draw_overlay(window[:108, 64:], self.overlay_store_ct, opacity=0.65)
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

    def lobby(self) -> np.ndarray:
        """Draw the console and players connected to the session."""
        window: np.ndarray = self.window_base_lobby.copy()

        # Display game status
        draw_text(window, 'in match' if self.session.phase else 'waiting to start', 2, 142)

        # Draw spectators
        for i, player in enumerate(self.session.spectators.values()):
            pos_y = 12 + i*8
            draw_number(window, player.id, pos_y, 4)
            draw_image(window, self.addressable_icons[GameID.NULL], pos_y - 3, 7)
            draw_text(window, player.name, pos_y, 19)

        # Draw Ts
        for i, player in enumerate(self.session.players_t.values()):
            pos_y = 40 + i*8
            draw_number(window, player.id, pos_y, 138)
            draw_image(window, self.addressable_icons[player.position_id], pos_y - 4, 144)
            draw_text(window, player.name, pos_y, 108)

        # Draw CTs
        for i, player in enumerate(self.session.players_ct.values()):
            pos_y = 40 + i*8
            draw_number(window, player.id, pos_y, 183)
            draw_image(window, self.addressable_icons[player.position_id], pos_y - 4, 164)
            draw_text(window, player.name, pos_y, 190)

        # Draw console
        draw_text(window, self.console_text, 128, 69, spacing=3)
        draw_image(window, self.icon_console_pointer, 124, 69 + min(len(self.console_text), 22)*8)

        return window

    def main(self) -> np.ndarray:
        """Draw the main HUD with information on current inventory, chat, and match state."""
        window = self.window_base_world.copy()
        player: Player = self.session.players[self.own_player_id]
        slots: List[Union[Object, None]] = player.slots

        # Draw equipped inventory
        self.draw_carried_item(window, slots[Item.SLOT_PRIMARY], 112, 187)
        self.draw_carried_item(window, slots[Item.SLOT_PISTOL], 112, 205)
        self.draw_carried_item(window, slots[Item.SLOT_OTHER], 112, 241)
        self.draw_carried_item(window, slots[Item.SLOT_UTILITY + Item.SUBSLOT_FLASH], 128, 187)
        self.draw_carried_item(window, slots[Item.SLOT_UTILITY + Item.SUBSLOT_EXPLOSIVE], 128, 205)
        self.draw_carried_item(window, slots[Item.SLOT_UTILITY + Item.SUBSLOT_INCENDIARY], 128, 223)
        self.draw_carried_item(window, slots[Item.SLOT_UTILITY + Item.SUBSLOT_SMOKE], 128, 241)

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
            draw_number(window, round(player.health), 137, 75)

        armour = slots[Item.SLOT_ARMOUR]

        if armour is not None:
            draw_number(window, round(armour.durability), 137, 93)

        draw_number(window, player.money, 120, 129)

        # Display magazine (or reloading/drawing or carried utility) and reserve
        if held_object is not None:
            if player.time_until_drawn or player.time_to_reload:
                draw_image(window, self.icon_reset, 129, 125)
            elif held_object.item.magazine_cap:
                draw_number(window, held_object.magazine, 129, 129)
            elif held_object.item.id != GameID.ITEM_KNIFE and held_object.item.slot != Item.SLOT_OTHER:
                draw_number(window, held_object.carrying, 129, 129)

            if held_object.item.reserve_cap:
                draw_number(window, held_object.reserve, 137, 129)

        # Display match state
        session = self.session
        phase = session.phase

        draw_number(window, session.rounds_won_t, 119, 150)
        draw_number(window, session.rounds_won_ct, 119, 169)
        draw_number(
            window,
            session.rounds_won_t + session.rounds_won_ct + (1 if phase != GameID.PHASE_RESET else 0),
            130, 166)
        draw_number(window, int(session.time // 60.), 138, 162)
        draw_number(window, int(session.time % 60.), 138, 172, min_n_digits=2)

        if phase == GameID.PHASE_BUY:
            draw_image(window, self.icon_store, 134, 148)
        elif phase == GameID.PHASE_DEFUSE:
            draw_image(window, self.inventory.c4.icon, 130, 144)
        elif phase == GameID.PHASE_RESET:
            draw_image(window, self.icon_reset, 134, 148)

        # Display own icon
        if player.team == GameID.GROUP_SPECTATORS:
            draw_image(window, self.addressable_icons[GameID.NULL], 116, 67)
        else:
            draw_image(window, self.addressable_icons[player.position_id], 116, 67)

        # Display chat
        # NOTE: The scroll wheel update enforces its range between `0` and `len(self.chat)-5`
        chat = self.chat[self.wheel_y:self.wheel_y+5]
        pos_y = 2
        pos_x = 4

        for message in chat:
            draw_number(window, message.round, pos_y, pos_x)
            draw_number(window, int(message.time // 60.), pos_y, pos_x + 12)
            draw_number(window, int(message.time % 60.), pos_y, pos_x + 22, min_n_digits=2)
            draw_image(window, self.addressable_icons[message.position_id], pos_y - 3, pos_x + 45)

            for i, word in enumerate(message.words):
                if not word:
                    break

                draw_image(window, self.addressable_icons[word], pos_y + 8, pos_x - 3 + 16*i)

            pos_y += 24

        # Add scrollbar
        width = int(np.ceil(min(5, len(self.chat)) / max(len(self.chat), 1) * 63))
        offset = int(self.wheel_y / max(len(self.chat), 1) * 63)
        window[121:123, offset:offset+width] = 127

        # Display current message draft
        for i, (word, _, _) in enumerate(self.message_draft):
            draw_image(window, self.addressable_icons[word], 128, 1 + 16*i)

        return window

    def mapstats(self, window: np.ndarray) -> np.ndarray:
        """Draw player positions on the map besides player statistics."""
        window[:108, 64:] = self.overlay_mapstats

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

            draw_image(window, self.addressable_icons[pos_id], 6 + i*9, 83)
            draw_number(window, int(kdr), 10 + i*9, 104)
            draw_number(window, round(10*(kdr - int(kdr))), 10 + i*9, 110)

            if observer_team != GameID.GROUP_TEAM_CT:
                draw_number(window, money, 10 + i*9, 138)

        for i, (kdr, pos_id, health, money) in enumerate(stats_ct):
            if not health:
                draw_image(window, self.icon_kill, 60 + i*9, 67)

            draw_image(window, self.addressable_icons[pos_id], 61 + i*9, 83)
            draw_number(window, int(kdr), 65 + i*9, 104)
            draw_number(window, round(10*(kdr - int(kdr))), 65 + i*9, 110)

            if observer_team != GameID.GROUP_TEAM_T:
                draw_number(window, money, 65 + i*9, 138)

        # Transform global coordinates into map view coordinates
        player_pos_id = [
            (np.dot(self.MAP_WARP, (*a_player.pos, 1.)) + (64, 0), a_player.position_id, a_player.health)
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
            window[pos_y, pos_x] = self.colours[pos_id]

            ring1_indices = self.RING1_INDICES[0] + pos_y, self.RING1_INDICES[1] + pos_x
            ring2_indices = self.RING2_INDICES[0] + pos_y, self.RING2_INDICES[1] + pos_x

            if not health:
                window[ring1_indices] = np.uint8(self.colours['dead'] * 0.6 + window[ring1_indices] * 0.4)
                window[ring2_indices] = np.uint8(self.colours['dead'] * 0.3 + window[ring2_indices] * 0.7)
            elif pos_id == player.position_id:
                window[ring1_indices] = np.uint8(self.colours['self'] * 0.6 + window[ring1_indices] * 0.4)
                window[ring2_indices] = np.uint8(self.colours['self'] * 0.3 + window[ring2_indices] * 0.7)
            else:
                window[ring1_indices] = np.uint8(self.colours[pos_id] * 0.6 + window[ring1_indices] * 0.4)
                window[ring2_indices] = np.uint8(self.colours[pos_id] * 0.3 + window[ring2_indices] * 0.7)

        # Display pings for own team
        # NOTE: The other team's pings should not be received at all
        for (pos_x, pos_y), pos_id, cover_indices, fading in ping_pos_id:
            pos_y = round(pos_y)
            pos_x = round(pos_x)
            colour = self.colours[pos_id - 120]
            draw_colour(window, cover_indices, colour, opacity=fading, pos_y=pos_y, pos_x=pos_x)

        # Get cursor obs
        if self.cursor_x >= 84.:

            # Iter over players, get argmin wrt. cursor position
            if player_pos_id:
                closest_player = min(
                    (np.linalg.norm(pos - (self.cursor_x, self.cursor_y)), pos_id)
                    for pos, pos_id, _ in player_pos_id)
            else:
                closest_player = (np.Inf, None)

            # Iter over pings, get argmin wrt. cursor position
            if ping_pos_id:
                closest_ping = min(
                    (np.linalg.norm(np.array(pos) - (self.cursor_x, self.cursor_y)), pos_id)
                    for pos, pos_id, _, _ in ping_pos_id)
            else:
                closest_ping = (np.Inf, None)

            # Get argmin of both
            closest_dist, closest_id = min(closest_player, closest_ping)

            if closest_dist < 3.:
                draw_image(window, self.addressable_icons[closest_id], 112, 85)

        return window

    def world(self, window: np.ndarray) -> np.ndarray:
        """Draw the in-game world with players, objects, and effects in line of sight."""
        player = self.session.players[self.observed_player_id]
        map_ = self.session.map

        # Get recoil-affected position and angle
        # NOTE: Recoil (moving viewpoint/origin) can cause the own sprite to
        # not always be drawn at the same (expected) position
        # (but it looks natural enough if only negative position offsets are used)
        pos = tuple(player.pos)
        origin = tuple(player.d_pos_recoil + self.WORLD_FRAME_ORIGIN)
        angle = player.angle + np.pi/2. + player.d_angle_recoil

        # Transform world into local frame wrt. observed entity
        world = project_into_view(map_.world, pos, angle, origin, self.WORLD_FRAME_SIZE)

        # To transform inhabiting effects, objects, and players, their positions need to be warped, as well
        world_warp = get_camera_warp(pos, angle, origin)

        # Transform reference map layers into local frame
        rot_height_map = project_into_view(map_.height, pos, angle, origin, self.WORLD_FRAME_SIZE, preserve_values=True)

        # Hide information from dead observed player
        dead_observant = not player.health and not self.session.is_spectator(self.own_player_id)

        if dead_observant:
            rot_entity_map = self.null_rot_entity_map
            rot_zone_map = self.null_rot_zone_map

        else:
            rot_entity_map = project_into_view(
                map_.player_id, pos, angle, origin, self.WORLD_FRAME_SIZE, preserve_values=True)
            rot_zone_map = project_into_view(map_.zone, pos, angle, origin, self.WORLD_FRAME_SIZE, preserve_values=True)

            # Iter over colour effects and display them
            for effect in self.effects.values():
                if effect.type == Effect.TYPE_COLOUR:
                    cover_indices = np.dot(world_warp, np.vstack((
                        effect.cover_indices[1] + effect.pos_x,
                        effect.cover_indices[0] + effect.pos_y,
                        np.ones_like(effect.cover_indices[0]))))

                    cover_indices = np.around(cover_indices).astype(np.int16)
                    cover_indices = (cover_indices[1], cover_indices[0])

                    draw_colour(world, cover_indices, effect.colour, effect.opacity, bounds=self.WORLD_BOUNDS)

        # Draw sprites etc.
        # NOTE: 1.03125 is used (instead of 1.) to round down the observed player position after warp to proper place

        # Draw dead players
        for a_player in self.session.players.values():
            if not a_player.health and self.check_los(player, a_player):
                pos_x, pos_y = np.dot(world_warp, (*a_player.pos, 1.))
                pos_x, pos_y = round(pos_x - 1.03125), round(pos_y - 1.03125)
                sprite = self.sprites[a_player.team, -1]

                draw_image(world, sprite, pos_y, pos_x, bounds=self.WORLD_BOUNDS)

        # Draw persistent objects (with infinite lifetime)
        for an_object in self.session.objects.values():
            if an_object.lifetime == np.Inf and self.check_los(player, an_object):
                pos_x, pos_y = np.dot(world_warp, (*an_object.pos, 1.))
                pos_x, pos_y = round(pos_x), round(pos_y)

                if 0 <= pos_y <= 107 and 0 <= pos_x <= 191:
                    world[pos_y, pos_x] = self.COLOUR_ITEM_INF

                    # Emphasise C4
                    if 1 <= pos_y <= 106 and 1 <= pos_x <= 190 and an_object.item.id == GameID.ITEM_C4:
                        ring1_indices = self.RING1_INDICES[0] + pos_y, self.RING1_INDICES[1] + pos_x
                        ring2_indices = self.RING2_INDICES[0] + pos_y, self.RING2_INDICES[1] + pos_x

                        world[ring1_indices] = np.uint8(self.COLOUR_ITEM_INF * 0.6 + world[ring1_indices] * 0.4)
                        world[ring2_indices] = np.uint8(self.COLOUR_ITEM_INF * 0.3 + world[ring2_indices] * 0.7)

        # Draw alive players
        for a_player in self.session.players.values():
            if a_player.health and self.check_los(player, a_player):
                pos_x, pos_y = np.dot(world_warp, (*a_player.pos, 1.))
                pos_x, pos_y = round(pos_x - 1.03125), round(pos_y - 1.03125)
                angle = self.get_relative_sprite_index(player, a_player)
                sprite = self.sprites[a_player.team, angle]

                draw_image(world, sprite, pos_y, pos_x, bounds=self.WORLD_BOUNDS)

        # Draw transient objects (with finite lifetime)
        for an_object in self.session.objects.values():
            if an_object.lifetime != np.Inf and self.check_los(player, an_object):
                pos_x, pos_y = np.dot(world_warp, (*an_object.pos, 1.))
                pos_x, pos_y = round(pos_x), round(pos_y)

                if 0 <= pos_y <= 107 and 0 <= pos_x <= 191:
                    world[pos_y, pos_x] = self.COLOUR_ITEM_FUSE

                    # Emphasise C4
                    if 1 <= pos_y <= 106 and 1 <= pos_x <= 190 and an_object.item.id == GameID.ITEM_C4:
                        ring1_indices = self.RING1_INDICES[0] + pos_y, self.RING1_INDICES[1] + pos_x
                        ring2_indices = self.RING2_INDICES[0] + pos_y, self.RING2_INDICES[1] + pos_x

                        world[ring1_indices] = np.uint8(self.COLOUR_ITEM_FUSE * 0.6 + world[ring1_indices] * 0.4)
                        world[ring2_indices] = np.uint8(self.COLOUR_ITEM_FUSE * 0.3 + world[ring2_indices] * 0.7)

        # Needed for residual effects
        self.last_world_frame = world

        # Differentiate between normal and scoped endpoints wrt. held item
        if self.observed_player_id == self.own_player_id \
                and player.held_object is not None and player.held_object.item.scoped:
            world = render_view(
                world, rot_height_map, rot_entity_map, rot_zone_map, self.scoped_fov_endpoints, observer_id=player.id)
        else:
            world = render_view(
                world, rot_height_map, rot_entity_map, rot_zone_map, self.standard_fov_endpoints, observer_id=player.id)

        # Draw overlays
        if not dead_observant:
            for effect in self.effects.values():
                if effect.type == Effect.TYPE_OVERLAY:
                    world = draw_overlay(world, effect.overlay, effect.opacity)

        # Draw muted if own player is dead
        if self.observed_player_id != self.own_player_id:
            own_player = self.session.players[self.own_player_id]
        else:
            own_player = player

        if not own_player.health and own_player.team != GameID.GROUP_SPECTATORS:
            world = draw_muted(world)

        window[:108, 64:] = world

        return window

    def check_los(
        self,
        observer: Player,
        entity: Union[Object, Player],
        ignore_zone: bool = False,
        ignore_players: bool = True
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
        if np.linalg.norm(observer_pos - entity_pos) > Player.MAX_VIEW_RANGE:
            return False

        # Check angle
        recentred_entity_pos = entity_pos - observer_pos
        angle = -observer.angle

        rotmat = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]])

        relative_x, relative_y = np.dot(rotmat, recentred_entity_pos)
        relative_angle = np.arctan2(-relative_y, relative_x)

        if isinstance(entity, Player) and (entity.held_object is not None and observer.held_object.item.scoped):
            angle_lim = self.FOV_SCOPED / 2. * np.pi / 180.
        else:
            angle_lim = self.FOV_MAIN / 2. * np.pi / 180.

        if abs(relative_angle) > angle_lim:
            return False

        # Trace path
        map_ = self.session.map
        zone_map = map_.zone_null if ignore_zone else map_.zone
        player_id_map = map_.player_id_null if ignore_players else map_.player_id

        barred_pos = sdglib.trace_sight(observer.id, observer_pos, entity_pos, map_.height, player_id_map, zone_map)

        if any(barred_pos):
            pos_x, pos_y = barred_pos
            return map_.player_id[int(pos_y), int(pos_x)] == entity.id

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

    def create_log(self, eval_type: int) -> Union[List[Union[int, float]], None]:
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
                if player.money >= self.inventory.get_item_by_id(hover_id).price:
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

        # Must be in world view
        if self.view != GameID.VIEW_WORLD:
            return

        own_player: Player = self.session.players.get(self.own_player_id, None)

        # If unable to infer player pools, return to lobby
        if own_player is None:
            self.exit_world()
            return

        # Must be spectator or dead
        if own_player.team != GameID.GROUP_SPECTATORS and own_player.health:
            return

        # Limit observable pool
        pool_t = list(self.session.players_t.keys())
        pool_ct = list(self.session.players_ct.keys())

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
