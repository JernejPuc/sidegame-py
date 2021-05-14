"""Elements shared between different parts of SDG"""

import os
from typing import Dict, Iterable, Tuple, Union
import numpy as np
import cv2
from sidegame.networking.core import EventBase


ASSET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets'))


class GameID:
    """Unique IDs that can be used in messaging etc."""

    NULL = 0

    GROUP_OBJECTS = 1
    GROUP_SPECTATORS = 2
    GROUP_TEAM_T = 3
    GROUP_TEAM_CT = 4

    PLAYER_T1 = 31
    PLAYER_T2 = 32
    PLAYER_T3 = 33
    PLAYER_T4 = 34
    PLAYER_T5 = 35
    PLAYER_CT1 = 41
    PLAYER_CT2 = 42
    PLAYER_CT3 = 43
    PLAYER_CT4 = 44
    PLAYER_CT5 = 45

    ITEM_ARMOUR = 50
    ITEM_RIFLE_T = 61
    ITEM_RIFLE_CT = 62
    ITEM_SMG_T = 63
    ITEM_SMG_CT = 64
    ITEM_SHOTGUN_T = 65
    ITEM_SHOTGUN_CT = 66
    ITEM_SNIPER = 67
    ITEM_PISTOL_T = 71
    ITEM_PISTOL_CT = 72
    ITEM_KNIFE = 80
    ITEM_C4 = 91
    ITEM_DKIT = 92
    ITEM_FLASH = 100
    ITEM_EXPLOSIVE = 110
    ITEM_INCENDIARY_T = 121
    ITEM_INCENDIARY_CT = 122
    ITEM_SMOKE = 130

    TERM_MOVE = 141
    TERM_HOLD = 142
    TERM_SEE = 143
    TERM_STOP = 144
    TERM_EXCLAME = 145
    TERM_ASK = 146
    TERM_KILL = 147

    MARK_T1 = 151
    MARK_T2 = 152
    MARK_T3 = 153
    MARK_T4 = 154
    MARK_T5 = 155
    MARK_CT1 = 161
    MARK_CT2 = 162
    MARK_CT3 = 163
    MARK_CT4 = 164
    MARK_CT5 = 165

    PHASE_BUY = 171
    PHASE_PLANT = 172
    PHASE_DEFUSE = 173
    PHASE_RESET = 174

    VIEW_WORLD = 181
    VIEW_MAPSTATS = 182
    VIEW_TERMS = 183
    VIEW_ITEMS = 184
    VIEW_STORE = 185
    VIEW_LOBBY = 186

    TERRAIN_CONCRETE = 191
    TERRAIN_DIRT = 192
    TERRAIN_WOOD = 193
    TERRAIN_METAL = 194
    TERRAIN_TILE = 195

    ROLE_SPECTATOR = 201
    ROLE_PLAYER = 202
    ROLE_ADMIN = 203

    CMD_START_MATCH = 211
    CMD_END_MATCH = 212
    CMD_END_SESSION = 213
    CMD_SET_TEAM = 214
    CMD_SET_NAME = 215
    CMD_SET_ROLE = 216
    CMD_GET_LATENCY = 217

    LOG_BUY = 221
    LOG_MESSAGE = 222

    CHEAT_END_ROUND = 231
    CHEAT_DEV_MODE = 232
    CHEAT_MAX_MONEY = 233

    EVAL_MSG_MARK = 241
    EVAL_MSG_ITEM = 242
    EVAL_MSG_TERM = 243
    EVAL_MSG_SEND = 244
    EVAL_BUY = 245


class Event(EventBase):
    """Types of in-game events."""

    NULL = 0

    CTRL_MATCH_STARTED = 1
    CTRL_MATCH_ENDED = 2
    CTRL_MATCH_PHASE_CHANGED = 3
    CTRL_SESSION_ENDED = 4
    CTRL_PLAYER_CONNECTED = 5
    CTRL_PLAYER_DISCONNECTED = 6
    CTRL_PLAYER_MOVED = 7
    CTRL_PLAYER_CHANGED = 8
    CTRL_LATENCY_REQUESTED = 9

    OBJECT_SPAWN = 11
    OBJECT_EXPIRE = 12
    OBJECT_ASSIGN = 13
    OBJECT_TRIGGER = 14

    C4_PLANTED = 21
    C4_DEFUSED = 22
    C4_DETONATED = 23
    FX_C4_TOUCHED = 24
    FX_C4_INIT = 25
    FX_C4_KEY_PRESS = 26
    FX_C4_BEEP = 27
    FX_C4_BEEP_DEFUSING = 28
    FX_C4_NVG = 29

    FX_BOUNCE = 31
    FX_LAND = 32
    FX_ATTACK = 33
    FX_CLIP_LOW = 34
    FX_CLIP_EMPTY = 35
    FX_EXTINGUISH = 36
    FX_FOOTSTEP = 37
    FX_FLASH = 38
    FX_WALL_HIT = 39

    PLAYER_DAMAGE = 41
    PLAYER_DEATH = 42
    PLAYER_MESSAGE = 43
    PLAYER_RELOAD = 44

    ID_EVENTS: Tuple[int] = (
        CTRL_PLAYER_DISCONNECTED, OBJECT_EXPIRE, OBJECT_TRIGGER,
        C4_DETONATED, FX_C4_TOUCHED, FX_C4_INIT, FX_C4_KEY_PRESS, FX_C4_BEEP, FX_C4_BEEP_DEFUSING, FX_C4_NVG,
        FX_BOUNCE, FX_LAND, FX_CLIP_LOW, FX_CLIP_EMPTY, FX_EXTINGUISH, FX_FOOTSTEP)


class Message:
    """In-game chat message structure."""

    def __init__(
        self,
        position_id: int,
        round_: int,
        time_: float,
        words: Iterable[int],
        marks: Iterable[Tuple[Union[int, float]]] = None,
        sender_id: int = None
    ):
        self.position_id = position_id
        self.round = round_
        self.time = time_
        self.words = words
        self.marks = marks
        self.sender_id = sender_id


class Map:
    """
    Mappings of the playable area, consisting of image and derived array data
    and code references for the content of specific arrays.
    """

    _MAP_REFS: Dict[int, Dict[str, str]] = {
        0: {
            'NAME': 'Cache',
            'RADAR': os.path.join(ASSET_DIR, 'maps', 'de_cache_radar.png'),
            'CODE': os.path.join(ASSET_DIR, 'maps', 'de_cache_code.png')}}

    # Code map channels
    _CHANNEL_SOUND = 0
    _CHANNEL_HEIGHT = 1
    _CHANNEL_MARK = 2

    # Sound channel codes
    SOUND_CONCRETE = 255
    SOUND_DIRT = 205
    SOUND_WOOD = 155
    SOUND_METAL = 105
    SOUND_TILE = 55
    SOUND_NULL = 0

    # Height map codes
    HEIGHT_GROUND = 0
    HEIGHT_TRANSITION = 63
    HEIGHT_ELEVATED = 127
    HEIGHT_IMPASSABLE = 255

    # Landmark channel codes
    LANDMARK_NULL = 0
    LANDMARK_SPAWN_T = 31
    LANDMARK_SPAWN_CT = 63
    LANDMARK_PLANT_A = 127
    LANDMARK_PLANT_B = 255

    # Zone map codes
    ZONE_NULL = 0
    ZONE_FIRE = 127
    ZONE_SMOKE = 255

    # Entity map codes (max 32728 objects and 32727 entities)
    OBJECT_ID_NULL = 0
    PLAYER_ID_NULL = 32767

    def __init__(self, map_id: int = None, rng: np.random.Generator = None):
        self.rng = np.random.default_rng() if rng is None else rng

        if map_id is None or map_id not in self._MAP_REFS:
            map_id = self.get_random_map_id()

        self.id = map_id
        self.world = cv2.imread(self._MAP_REFS[map_id]['RADAR'], cv2.IMREAD_COLOR)

        code_map: np.ndarray = cv2.imread(self._MAP_REFS[map_id]['CODE'], cv2.IMREAD_COLOR)
        reference_shape: Tuple[int] = code_map.shape[:2]

        self.sound = code_map[..., self._CHANNEL_SOUND]
        self.height = code_map[..., self._CHANNEL_HEIGHT]
        self.landmark = code_map[..., self._CHANNEL_MARK]

        self.wall = np.uint8(self.height == self.HEIGHT_IMPASSABLE)

        self.zone = np.empty(reference_shape, dtype=np.uint8)
        self.zone_null = np.empty(reference_shape, dtype=np.uint8)
        self.zone_id = np.empty(reference_shape, dtype=np.int16)
        self.object_id = np.empty(reference_shape, dtype=np.int16)
        self.player_id = np.empty(reference_shape, dtype=np.int16)
        self.player_id_null = np.empty(reference_shape, dtype=np.int16)
        self.reset()

        spawn_origin_t_y, spawn_origin_t_x = np.where(self.landmark == self.LANDMARK_SPAWN_T)
        spawn_origin_ct_y, spawn_origin_ct_x = np.where(self.landmark == self.LANDMARK_SPAWN_CT)

        self.spawn_origin_t = np.array([spawn_origin_t_x[0], spawn_origin_t_y[0]])
        self.spawn_origin_ct = np.array([spawn_origin_ct_x[0], spawn_origin_ct_y[0]])

    def get_random_map_id(self) -> int:
        """Get random map id..."""

        return self.rng.choice(list(self._MAP_REFS.keys()))

    @classmethod
    def get_map_id_by_name(cls, name: str) -> Union[int, None]:
        """Get map id by name..."""

        ids = [k for k in cls._MAP_REFS if cls._MAP_REFS[k]['NAME'] == name]

        return ids[0] if ids else None

    def reset(self):
        """Fill dynamic maps with null values."""

        self.zone.fill(self.ZONE_NULL)
        self.zone_null.fill(self.ZONE_NULL)
        self.zone_id.fill(self.OBJECT_ID_NULL)
        self.object_id.fill(self.OBJECT_ID_NULL)
        self.player_id.fill(self.PLAYER_ID_NULL)
        self.player_id_null.fill(self.PLAYER_ID_NULL)
