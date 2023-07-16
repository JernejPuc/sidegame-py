"""Elements shared between different parts of SDG"""

from typing import Iterable, Tuple, Union

from sidegame.networking.core import EventBase


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
