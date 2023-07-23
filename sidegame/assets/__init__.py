import os
from enum import IntEnum

import numpy as np
import cv2

from sidegame.audio import SoundBank as SoundBankBase
from sidegame.game import GameID, MapID


ASSET_DIR = os.path.dirname(__file__)


# Mapping between world pos and map view pos
# y: 535 to x: 90, y: 103 to x: 185 || x: 20 to y: 1, x: 603 to y: 106
MAP_WARP = np.array((
    ((1., 0., 0.), (0., 1., 0.)),
    ((1., 0., 0.), (0., 1., 0.)),
    ((1., 0., 0.), (0., 1., 0.)),
    ((1., 0., 0.), (0., 1., 0.)),
    ((1., 0., 0.), (0., 1., 0.)),
    ((0., -95./432., 185. + 103.*95./432.), (105./583., 0., 1. - 105.*20./583.))))


class Map:
    """
    Mappings of the playable area, consisting of image and derived array data
    and code references for the content of specific arrays.
    """

    LEVEL_FILES = {MapID.LEVEL_5V5: ('de_cache_radar.png', 'de_cache_code.png')}

    def __init__(self, map_id: int = MapID.LEVEL_5V5):
        self.id = map_id
        world_file, code_file = Map.LEVEL_FILES[map_id]

        self.world = ImageBank.load({}, 'world_5v5', 'maps', world_file, mode=cv2.IMREAD_COLOR)
        self.code_map = ImageBank.load({}, 'code_5v5', 'maps', code_file, mode=cv2.IMREAD_COLOR)
        self.bounds = self.world.shape[:2]

        self.fx_canvas = np.empty_like(self.world)
        self.code_map = np.concatenate((np.moveaxis(self.code_map, -1, 0), np.empty((4, *self.bounds), dtype=np.uint8)))
        self.id_map = np.empty((4, *self.bounds), dtype=np.int16)

        self.sound = self.code_map[MapID.CHANNEL_SOUND]
        self.height = self.code_map[MapID.CHANNEL_HEIGHT]
        self.landmark = self.code_map[MapID.CHANNEL_MARK]
        self.wall = self.code_map[MapID.CHANNEL_WALL]
        self.zone = self.code_map[MapID.CHANNEL_ZONE]
        self.zone_null = self.code_map[MapID.CHANNEL_ZONE_NULL]
        self.fx_ctr_map = self.code_map[MapID.CHANNEL_FX]

        self.zone_id = self.id_map[MapID.CHANNEL_ZONE_ID]
        self.object_id = self.id_map[MapID.CHANNEL_OBJECT_ID]
        self.player_id = self.id_map[MapID.CHANNEL_PLAYER_ID]
        self.player_id_null = self.id_map[MapID.CHANNEL_PLAYER_ID_NULL]

        self.wall[:] = np.uint8(self.height == MapID.HEIGHT_IMPASSABLE)
        self.reset()

        spawn_origin_t_y, spawn_origin_t_x = np.nonzero(self.landmark == MapID.LANDMARK_SPAWN_T)
        spawn_origin_ct_y, spawn_origin_ct_x = np.nonzero(self.landmark == MapID.LANDMARK_SPAWN_CT)

        self.spawn_origin_t = np.array((spawn_origin_t_x[0], spawn_origin_t_y[0]), dtype=np.float64)
        self.spawn_origin_ct = np.array((spawn_origin_ct_x[0], spawn_origin_ct_y[0]), dtype=np.float64)

    def reset(self):
        """Fill dynamic maps with null values."""

        self.fx_canvas.fill(0)
        self.fx_ctr_map.fill(0)
        self.zone.fill(MapID.ZONE_NULL)
        self.zone_null.fill(MapID.ZONE_NULL)
        self.zone_id.fill(MapID.OBJECT_ID_NULL)
        self.object_id.fill(MapID.OBJECT_ID_NULL)
        self.player_id.fill(MapID.PLAYER_ID_NULL)
        self.player_id_null.fill(MapID.PLAYER_ID_NULL)


class ImageBank(dict):
    COLOURS = {
        'black': np.array((0, 0, 0), dtype=np.uint8),
        'grey': np.array((63, 63, 63), dtype=np.uint8),
        'white': np.array((255, 255, 255), dtype=np.uint8),
        'red': np.array((0, 0, 255), dtype=np.uint8),
        'green': np.array((0, 255, 0), dtype=np.uint8),
        'blue': np.array((255, 0, 0), dtype=np.uint8),
        'yellow': np.array((0, 255, 255), dtype=np.uint8),
        'e_yellow': np.array((63, 191, 191), dtype=np.uint8),
        'e_red': np.array((0, 127, 255), dtype=np.uint8),
        't_cyan': np.array((235, 183, 0), dtype=np.uint8),
        't_red': np.array((0, 0, 224), dtype=np.uint8),
        'o_purple': np.array((140, 80, 140), dtype=np.uint8),
        'o_red': np.array((127, 127, 191), dtype=np.uint8),
        'p_green': np.array((0, 248, 160), dtype=np.uint8),
        'p_yellow': np.array((16, 192, 255), dtype=np.uint8),
        'p_red': np.array((96, 72, 255), dtype=np.uint8),
        'p_purple': np.array((224, 96, 160), dtype=np.uint8),
        'p_blue': np.array((224, 224, 0), dtype=np.uint8)}

    def __init__(self):
        super().__init__()

        # TODO: Read from fewer files and split at init

        # Characters and digits
        self.characters: dict[str, np.ndarray] = {}

        for charfile in os.listdir(os.path.join(ASSET_DIR, 'characters')):
            char = charfile.split('_')[1][:-4]
            self.characters[char] = self.load(char, 'characters', charfile)

        self.digits: tuple[np.ndarray, ...] = tuple(self.characters[str(num)] for num in range(10))

        # FoV endpoints
        self.load('endpoints', 'views', 'endpoints.png', mode=cv2.IMREAD_GRAYSCALE)

        # View-related
        self.load('window_base_lobby', 'views', 'lobby.png')
        self.load('window_base_world', 'views', 'main.png')
        self.load('overlay_mapstats', 'views', 'mapstats.png')
        self.load('overlay_terms', 'views', 'terms.png')
        self.load('overlay_items', 'views', 'items.png')
        self.load('overlay_store_t', 'views', 'store_t.png')
        self.load('overlay_store_ct', 'views', 'store_ct.png')

        self.load('code_view_terms', 'views', 'code_terms.png', mode=cv2.IMREAD_GRAYSCALE)
        self.load('code_view_items', 'views', 'code_items.png', mode=cv2.IMREAD_GRAYSCALE)
        self.load('code_view_store_t', 'views', 'code_store_t.png', mode=cv2.IMREAD_GRAYSCALE)
        self.load('code_view_store_ct', 'views', 'code_store_ct.png', mode=cv2.IMREAD_GRAYSCALE)

        # Specific icons
        self.load('icon_console_pointer', 'icons', 'pointer_console.png')
        self.load('icon_cursor', 'icons', 'pointer_cursor.png')
        self.load('icon_selected', 'icons', 'pointer_item.png')
        self.load('icon_reset', 'icons', 'phase_reset.png')
        self.load('icon_store', 'icons', 'phase_buy.png')

        self.load('group_spectators', 'icons', 'team_spectator.png')
        self.load('group_team_t', 'icons', 'team_t.png'),
        self.load('group_team_ct', 'icons', 'team_ct.png')

        for i in range(5):
            self.load(f'player_t{i+1}', 'icons', f'agent_t_{i}.png')
            self.load(f'player_ct{i+1}', 'icons', f'agent_ct_{i}.png')
            self.load(f'mark_t{i+1}', 'icons', f'ping_t_{i}.png')
            self.load(f'mark_ct{i+1}', 'icons', f'ping_ct_{i}.png')

        self.load('term_kill', 'icons', 'term_kill.png')
        self.load('term_move', 'icons', 'term_move.png')
        self.load('term_hold', 'icons', 'term_hold.png')
        self.load('term_see', 'icons', 'term_see.png')
        self.load('term_stop', 'icons', 'term_fullstop.png')
        self.load('term_exclame', 'icons', 'term_exclamation.png')
        self.load('term_ask', 'icons', 'term_question.png')

        self.load('item_armour', 'icons', 'i0_armour.png')
        self.load('item_rifle_t', 'icons', 'i1_rifle_t.png')
        self.load('item_rifle_ct', 'icons', 'i1_rifle_ct.png')
        self.load('item_smg_t', 'icons', 'i1_smg_t.png')
        self.load('item_smg_ct', 'icons', 'i1_smg_ct.png')
        self.load('item_shotgun_t', 'icons', 'i1_shotgun_t.png')
        self.load('item_shotgun_ct', 'icons', 'i1_shotgun_ct.png')
        self.load('item_sniper', 'icons', 'i1_sniper.png')
        self.load('item_pistol_t', 'icons', 'i2_pistol_t.png')
        self.load('item_pistol_ct', 'icons', 'i2_pistol_ct.png')
        self.load('item_knife', 'icons', 'i3_knife.png')
        self.load('item_dkit', 'icons', 'i4_dkit.png')
        self.load('item_c4', 'icons', 'i4_c4.png')
        self.load('item_flash', 'icons', 'i5_flash.png')
        self.load('item_explosive', 'icons', 'i5_explosive.png')
        self.load('item_incendiary_t', 'icons', 'i5_incendiary_t.png')
        self.load('item_incendiary_ct', 'icons', 'i5_incendiary_ct.png')
        self.load('item_smoke', 'icons', 'i5_smoke.png')

        # Remap for convenience
        self.id_icons: dict[str, np.ndarray] = {}

        for str_id, int_id in vars(GameID).items():
            arr = self.get(str_id.lower())

            if arr is not None:
                self.id_icons[int_id] = arr

        # Team and angle specific sprites
        self.sprites: dict[tuple[int, int], np.ndarray] = {}

        for i in range(16):
            self.sprites[(GameID.GROUP_TEAM_T, i)] = self.load(f'sprite_t_{i}', 'sprites', f't_{i}.png')
            self.sprites[(GameID.GROUP_TEAM_CT, i)] = self.load(f'sprite_ct_{i}', 'sprites', f'ct_{i}.png')

        self.sprites[(GameID.GROUP_TEAM_T, -1)] = self.load('sprite_t_-1', 'sprites', 't_dead.png')
        self.sprites[(GameID.GROUP_TEAM_CT, -1)] = self.load('sprite_ct_-1', 'sprites', 'ct_dead.png')

        # Associate colours with player position ids
        for key, val in self.COLOURS.items():
            self[f'clr_{key}'] = val

        self.other_colours = {
            'dead': self['clr_black'],
            'self': self['clr_white'],
            'obj_reg': self['clr_o_purple'],
            'obj_fuse': self['clr_o_red']}

        self.player_colours = {
            GameID.PLAYER_T1: self['clr_p_green'],
            GameID.PLAYER_T2: self['clr_p_yellow'],
            GameID.PLAYER_T3: self['clr_p_red'],
            GameID.PLAYER_T4: self['clr_p_purple'],
            GameID.PLAYER_T5: self['clr_p_blue'],
            GameID.PLAYER_CT1: self['clr_p_green'],
            GameID.PLAYER_CT2: self['clr_p_yellow'],
            GameID.PLAYER_CT3: self['clr_p_red'],
            GameID.PLAYER_CT4: self['clr_p_purple'],
            GameID.PLAYER_CT5: self['clr_p_blue']}

    def load(self, name: str, groupname: str, filename: str, mode: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
        """Wrapper around `cv2.imread` to minimise path specification."""

        if name in self:
            raise KeyError(f'Image associated with "{name}" already in image bank.')

        img = cv2.imread(os.path.join(ASSET_DIR, groupname, filename), flags=mode)

        self[name] = img

        return img


class SoundBank(SoundBankBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.load('ambient', 'general', 'bg_phoenixfacility_01_mod.wav', 0.5)
        self.load('clip_low', 'general', 'lowammo_01.wav')
        self.load('clip_empty', 'general', 'clipempty_rifle.wav')
        self.load('msg_sent', 'general', 'playerping.wav', 0.5)
        self.load('msg_deleted', 'general', 'menu_accept.wav', 0.5)
        self.load('word_added', 'general', 'counter_beep.wav', 0.5)
        self.load('msg_received', 'general', 'lobby_notification_chat.wav', 0.5)
        self.load('mark_received', 'general', 'ping_alert_01.wav', 0.5)
        self.load('reset_round', 'general', 'pl_respawn.wav', 0.5)
        self.load('reset_side', 'general', 'bonus_alert_start.wav', 0.5)
        self.load('planted', 'general', 'bombpl_mod.wav', 0.5)
        self.load('defused', 'general', 'bombdef_mod.wav', 0.5)
        self.load('ct_win', 'general', 'ctwin_mod.wav', 0.5)
        self.load('t_win', 'general', 'terwin_mod.wav', 0.5)
        self.load('get', 'general', 'pickup_weapon_01.wav')
        self.load('drop', 'grenades', 'grenade_throw.wav')
        self.load('buy', 'general', 'radial_menu_buy_02.wav')
        self.load('no_buy', 'general', 'weapon_cant_buy.wav')
        self.load('death', 'player', 'death1.wav', 0.5)
        self.load('hit', 'player', 'kevlar5.wav')
        self.load('sine_max', 'grenades', 'flashbang_sine1_new.wav')
        self.load('sine_mid', 'grenades', 'flashbang_sine2_new.wav')
        self.load('sine_min', 'grenades', 'flashbang_sine3_new.wav')

        # Movements
        self.movements = [self.load(f'movement_{i}', 'player', f'movement{i+1}.wav', 0.125) for i in range(3)]

        # Footsteps
        terrain_keys = (MapID.SOUND_CONCRETE, MapID.SOUND_DIRT, MapID.SOUND_WOOD, MapID.SOUND_METAL, MapID.SOUND_TILE)
        terrain_names = ('concrete', 'dirt', 'wood', 'metal', 'tile')

        self.footsteps: dict[int, list[list[np.ndarray]]] = {}

        for terrain_key, terrain_name in zip(terrain_keys, terrain_names):
            dirname = os.path.join(ASSET_DIR, 'sounds', 'player', terrain_name)
            filenames = os.listdir(dirname)

            self.footsteps[terrain_key] = [
                SoundBankBase.load(self, os.path.join(dirname, filename), f'{terrain_name}_{i}', 0.125)
                for i, filename in enumerate(filenames)]

        # Main weapons
        sound_keys = ('draw', 'attack', 'reload_start', 'reload_add', 'reload_end')
        item_file_map = {
            'rifle_t': ('ak47_draw.wav', 'ak47_01.wav', 'ak47_clipout.wav', 'ak47_clipin.wav', 'ak47_boltpull.wav'),
            'rifle_ct': ('m4a1_draw.wav', 'm4a1_01.wav', 'm4a1_clipout.wav', 'm4a1_clipin.wav', 'm4a1_cliphit.wav'),
            'smg_t': ('ump45_draw.wav', 'ump45_02.wav', 'ump45_clipout.wav', 'ump45_clipin.wav', 'ump45_bolt_mod.wav'),
            'smg_ct': ('mp9_draw.wav', 'mp9_01.wav', 'mp9_clipout.wav', 'mp9_clipin.wav', 'mp9_bolt_mod.wav'),
            'shotgun_t': ('nova_draw.wav', 'nova-1.wav', None, 'nova_insertshell.wav', 'nova_pump.wav'),
            'shotgun_ct': ('mag7_draw.wav', 'mag7_01.wav', 'mag7_clipout.wav', 'mag7_clipin.wav', 'mag7_pump_mod.wav'),
            'sniper': ('awp_draw.wav', 'awp_01.wav', 'awp_clipout.wav', 'awp_clip_mod.wav', 'awp_bolt_mod.wav'),
            'pistol_t': (
                'glock_draw.wav', 'hkp2000_01.wav', 'glock_clipout.wav', 'glock_clipin.wav', 'glock_slide_mod.wav'),
            'pistol_ct': (
                'usp_draw.wav', 'usp_01.wav', 'usp_sliderelease.wav', 'usp_clipin_mod.wav', 'usp_slideback.wav')}

        self.item_sounds: dict[str, dict[str, list[np.ndarray]]] = {item_name: {} for item_name in item_file_map}

        for item_name, item_files in item_file_map.items():
            for sound_key, filename in zip(sound_keys, item_files):
                if filename is None:
                    continue

                self.item_sounds[item_name][sound_key] = self.load(f'{item_name}_{sound_key}', item_name, filename)

        # Grenades
        sound_keys = ('draw', 'detonate', 'extinguish')
        shared_file_map = {'attack': 'grenade_throw.wav', 'bounce': 'he_bounce-1.wav', 'land': 'grenade_hit1.wav'}
        item_file_map = {
            'flash': ('flashbang_draw.wav', 'flashbang_explode1.wav', None),
            'explosive': ('he_draw.wav', 'hegrenade_detonate_03.wav', None),
            'incendiary_t': ('molotov_draw.wav', 'molotov_detonate_1_mod.wav', 'molotov_extinguish.wav'),
            'smoke': ('smokegrenade_draw.wav', 'smoke_emit.wav', None)}

        for item_name in item_file_map.keys():
            self.item_sounds[item_name] = {}

        for sound_key, filename in shared_file_map.items():
            sound = self.load(f'grenade_{sound_key}', 'grenades', filename)

            for item_name in item_file_map.keys():
                self.item_sounds[item_name][sound_key] = sound

        for item_name, item_files in item_file_map.items():
            for sound_key, filename in zip(sound_keys, item_files):
                if filename is None:
                    continue

                self.item_sounds[item_name][sound_key] = self.load(f'{item_name}_{sound_key}', 'grenades', filename)

        self.item_sounds['incendiary_ct'] = {
            sound_key: sound for sound_key, sound in self.item_sounds['incendiary_t'].items()}

        # Other
        self.item_sounds['armour'] = {}

        self.item_sounds['knife'] = {
            'draw': self.load('knife_draw', 'knife', 'knife_deploy.wav'),
            'attack': self.load('knife_attack', 'knife', 'knife_slash1.wav'),
            'front_hit': self.load('knife_front_hit', 'knife', 'knife_hit_01.wav'),
            'back_hit': self.load('knife_back_hit', 'knife', 'knife_stab.wav')}

        self.item_sounds['c4'] = {
            'draw': self.load('c4_draw', 'c4', 'c4_draw.wav'),
            'init': self.load('c4_init', 'c4', 'c4_initiate.wav'),
            'press1': self.load('c4_press1', 'c4', 'key_press1.wav'),
            'press2': self.load('c4_press2', 'c4', 'key_press2.wav'),
            'press3': self.load('c4_press3', 'c4', 'key_press3.wav'),
            'press4': self.load('c4_press4', 'c4', 'key_press4.wav'),
            'press5': self.load('c4_press5', 'c4', 'key_press5.wav'),
            'press6': self.load('c4_press6', 'c4', 'key_press6.wav'),
            'press7': self.load('c4_press7', 'c4', 'key_press7.wav'),
            'plant': self.load('c4_plant', 'c4', 'c4_plant.wav'),
            'disarming': self.load('c4_disarming', 'c4', 'c4_disarmstart.wav'),
            'disarmed': self.load('c4_disarmed', 'c4', 'c4_disarmfinish.wav'),
            'nvg': self.load('c4_nvg', 'c4', 'nvg_on_mod.wav'),
            'explode': self.load('c4_explode', 'c4', 'c4_explode1.wav'),
            'beep_a': self.load('c4_beep_a', 'c4', 'c4_beep2.wav'),
            'beep_b': self.load('c4_beep_b', 'c4', 'c4_beep3.wav')}

        self.item_sounds['dkit'] = {'draw': self.item_sounds['c4']['draw']}

    def load(self, name: str, groupname: str, filename: str, volume_mod: float = 1.) -> list[np.ndarray]:
        """Wrapper around `audio::SoundBank.load` to minimise path specification."""

        return SoundBankBase.load(self, os.path.join(ASSET_DIR, 'sounds', groupname, filename), name, volume_mod)
