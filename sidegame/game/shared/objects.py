"""Functional extensions of items in SDG"""

from collections import deque
from typing import Iterable

import numpy as np
from numpy import ndarray
from numba import jit

from sidegame.assets import Map
from sidegame.game import EventID, MapID
from sidegame.utils_jit import index2_by_tuple, get_disk_mask, vec2_norm2, fix_angle_range, F_PI, F_PI2
from sidegame.physics import ThrowableEntity, PlayerEntity, update_collider_map, trace_shot, trace_sight
from sidegame.networking.core import Entity, Event
from sidegame.game.shared import Item


MAX_SHOT_RANGE = 640.
DIST_MOD_REF_RANGE = 56.86
KNIFE_REACH = 4. * np.sqrt(2.)


class Object(Entity, ThrowableEntity):
    """
    An instance of an item, realised in the game world.

    Attributes:
        item: Item which the object is an instance of.
        owner: Player with control over the object.
        id: Object entity ID if detached from its owner, i.e. thrown.
        lifetime: Amount of time the object can persist in the game world.
        durability: Remaining object durability.
        magazine: Remaining ammo in the magazine.
        reserve: Remaining ammo in the reserve.
        carrying: Number of carried item instances.
    """

    def __init__(
        self,
        item: Item,
        owner: PlayerEntity,
        entity_id: int = 0,
        lifetime: float = np.inf,
        durability: float = None,
        magazine: int = None,
        reserve: int = None,
        carrying: int = None
    ):
        Entity.__init__(self, entity_id)
        ThrowableEntity.__init__(self, object_id=entity_id)

        self.item = item
        self.owner = owner
        self.lifetime = lifetime

        self.durability: float = item.durability_cap if durability is None else durability
        self.magazine: int = item.magazine_cap if magazine is None else magazine
        self.reserve: int = item.reserve_cap if reserve is None else reserve
        self.carrying: int = 1 if carrying is None else carrying

    def get_values(self):
        """Get ammo/state values."""

        return self.durability, self.magazine, self.reserve, self.carrying

    def set_values(self, durability: float, magazine: int, reserve: int, carrying: int):
        """Get ammo/state values."""

        self.durability = durability
        self.magazine = magazine
        self.reserve = reserve
        self.carrying = carrying

    def reset_ammunition(self):
        """Reset magazine and reserve to their maximum capacity."""

        self.magazine = self.item.magazine_cap
        self.reserve = self.item.reserve_cap

    def update(self, dt: float, map_: Map, _players: list[PlayerEntity], events: list[Event]):
        """
        Evaluate object lifetime and return any resulting events.

        Can be overridden to produce additional or different events.
        To this end, the object can interact with given `players` and `map`.
        """

        self.lifetime -= dt

        if self.lifetime > 0.:
            return self.update_move(dt, map_, events)

        events.append(Event(EventID.OBJECT_EXPIRE, self.id))

    def update_move(self, dt: float, map_: Map, events: list[Event]):
        """Move towards target position, returning any collision events."""

        old_pos = self.pos

        event_id = self.move(dt, map_.wall, map_.object_id)

        update_collider_map(self.covered_indices, map_.object_id, old_pos, self.pos, self.id, MapID.OBJECT_ID_NULL)

        if event_id == EventID.FX_BOUNCE:
            events.append(Event(event_id, self.id))

        elif event_id == EventID.FX_LAND:
            events.append(Event(event_id, self.id))


class Weapon(Object):
    """A primary weapon or pistol instance."""

    def __init__(self, item: Item, owner: PlayerEntity, rng: np.random.Generator = None):
        super().__init__(item, owner)

        self.rng = np.random.default_rng() if rng is None else rng

    def fire(
        self,
        pos: ndarray,
        vel: ndarray,
        angle: float,
        firing_inaccuracy: float,
        wall_map: ndarray,
        player_id_map: ndarray
    ) -> list[Event]:
        """Try to attack, returning damage, hit, and sound events."""

        # Convey empty clip sound
        if self.magazine == 0:
            return (Event(EventID.FX_CLIP_EMPTY, self.owner.id),)

        item = self.item
        owner_id = self.owner.id

        # Convey attack sound
        events = [Event(EventID.FX_ATTACK, (owner_id, item.id))]

        # Convey low-on-ammo sound
        if self.magazine <= self.item.magazine_cap//3:
            events.append(Event(EventID.FX_CLIP_LOW, owner_id))

        # Subtract ammo charge
        self.magazine -= 1

        # Interpolate base inacuraccies to randomise firing angle
        halved_inaccuracy = get_halved_inaccuracy(
            vel, item.velocity_cap, item.moving_inaccuracy, item.standing_inaccuracy, firing_inaccuracy)

        # Get hits
        for _ in range(item.pellets):
            firing_angle = angle + self.rng.triangular(-halved_inaccuracy, 0., halved_inaccuracy)

            event_id, hit_id, hit_pos, dmg = fire_weapon(
                owner_id, pos, firing_angle, item.base_damage, item.distance_modifier, wall_map, player_id_map)

            if event_id == EventID.PLAYER_DAMAGE:
                events.append(Event(event_id, (owner_id, hit_id, item.id, dmg)))

            elif event_id == EventID.FX_WALL_HIT:
                events.append(Event(event_id, (hit_pos, owner_id, item.id)))

        return events


class Knife(Object):
    """A knife item instance."""

    SWIPE_ANGLES = np.array([-F_PI/8., -F_PI/16., 0., F_PI/16., F_PI/8.])

    def slash(
        self,
        pos: ndarray,
        wall_map: ndarray,
        player_id_map: ndarray,
        players: dict[int, PlayerEntity]
    ) -> list[Event]:
        """Try to attack, returning damage, hit, and sound events."""

        owner_id = self.owner_id
        item_id = self.item.id
        events = []

        # Convey attack sound
        events.append(Event(EventID.FX_ATTACK, (owner_id, item_id)))

        # Get hits
        for swipe_angle in self.SWIPE_ANGLES + self.owner.angle:
            event_id, hit_id, hit_pos = slash_knife(owner_id, pos, swipe_angle, wall_map, player_id_map)

            if event_id == EventID.PLAYER_DAMAGE:
                dmg = self.get_damage(self.owner.angle, players[hit_id].angle)
                events.append(Event(event_id, (owner_id, hit_id, item_id, dmg)))
                break

            elif event_id == EventID.FX_WALL_HIT:
                events.append(Event(event_id, (hit_pos, owner_id, item_id)))

        return events

    def get_damage(self, angle: float, hit_angle) -> float:
        """Get damage based on angle of the hit (damage to the back is tripled)."""

        dmg = self.item.base_damage
        d_angle = fix_angle_range(angle - hit_angle)

        if abs(d_angle) < F_PI2:
            dmg *= 3.

        return dmg


class Flash(Object):
    """
    A flash grenade (flashbang) item instance.

    Reference:
    3kliksphilip: Where Flashbangs flash and bang | https://www.youtube.com/watch?v=aTR7Surb80w
    """

    def __init__(self, item: Item, owner: PlayerEntity):
        super().__init__(item, owner, lifetime=item.fuse_time)

    def update(self, dt: float, map_: Map, players: list[PlayerEntity], events: list[Event]):
        """
        Evaluate object lifetime and return any resulting events.

        Can include trigger and `FX_FLASH` events,
        the latter containing flash debuffs.
        """

        self.lifetime -= dt

        if self.lifetime > 0.:
            return self.update_move(dt, map_, events)

        self_pos = self.pos
        code_map = map_.code_map
        player_id_map = map_.player_id_null

        # Get debuff strength per player
        events.append(Event(EventID.OBJECT_TRIGGER, self.id))

        for player in players:
            debuff = get_flash_strength(self_pos, player.pos, player.angle, code_map, player_id_map)

            if debuff != 0.:
                events.append(
                    Event(EventID.FX_FLASH, (self.owner.id, player.id, debuff, self.get_debuff_duration(debuff))))

        # Expire upon being triggered
        events.append(Event(EventID.OBJECT_EXPIRE, self.id))

    @staticmethod
    def get_debuff_duration(debuff_strength: float) -> float:
        """
        Get debuff duration wrt. its strength.

        Associating 1.0 debuff strength with 4.87s lifetime and scaling linearly,
        0.2 debuff corresponds to 0.97s, 0.4 to 1.95s, 0.7 to 3.4s, etc.
        With the simplified formula above, these points are a bit off,
        but close enough.
        """

        return debuff_strength * 4.87


class Explosive(Object):
    """
    An explosive (frag/high-explosive/HE) grenade item instance.

    Reference:
    3kliksphilip: CS GO Advanced Grenade Facts | https://www.youtube.com/watch?v=wfejKsHX2zA
    """

    def __init__(self, item: Item, owner: PlayerEntity):
        super().__init__(item, owner, lifetime=item.fuse_time)

    def update(self, dt: float, map_: Map, players: list[PlayerEntity], events: list[Event]):
        """
        Evaluate object lifetime and return any resulting events.

        Can include trigger and damage events.
        """

        self.lifetime -= dt

        if self.lifetime > 0.:
            return self.update_move(dt, map_, events)

        self_pos = self.pos
        base_damage = self.item.base_damage
        blast_radius = self.item.radius
        wall_map = map_.wall
        player_id_map = map_.player_id_null

        # Get damage per player
        events.append(Event(EventID.OBJECT_TRIGGER, self.id))

        for player in players:
            damage = get_he_damage(self_pos, player.pos, base_damage, blast_radius, wall_map, player_id_map)

            if damage != 0.:
                events.append(Event(EventID.PLAYER_DAMAGE, (self.owner.id, player.id, self.item.id, damage)))

        # Expire upon being triggered
        events.append(Event(EventID.OBJECT_EXPIRE, self.id))


class Incendiary(Object):
    """
    An instance of an incendiary (molotov) grenade item variant.

    References:
    https://counterstrike.fandom.com/wiki/Incendiary_Grenade
    https://counterstrike.fandom.com/wiki/Molotov
    """

    # Evaluating limited times per second prevents damage logs from being created on each tick,
    # all of which would have to be communicated to clients
    EVAL_INTERVAL = 0.2

    def __init__(self, item: Item, owner: PlayerEntity):
        super().__init__(item, owner, lifetime=item.fuse_time + item.duration)

        self.triggered = False
        self.cover_indices: tuple[ndarray] = None
        self.next_eval_time = self.item.duration
        self.accumulated_dt = 0.

    def set_zone_cover(self, map_: Map) -> Iterable[Event]:
        """
        Claim area in zone and zone ID maps at cover indices,
        returning trigger and/or other events if cover indices were
        partially or fully reduced (flame extinguished).
        """

        event_id, self.cover_indices = set_flame_cover(
            self.id, self.pos, self.item.radius, map_.code_map, map_.zone_id)

        if event_id == EventID.OBJECT_EXPIRE:
            return (Event(EventID.FX_EXTINGUISH, self.id), Event(EventID.OBJECT_EXPIRE, self.id))

        elif event_id == EventID.FX_EXTINGUISH:
            return (Event(EventID.FX_EXTINGUISH, self.id), Event(EventID.OBJECT_TRIGGER, self.id))

        return (Event(EventID.OBJECT_TRIGGER, self.id), )

    def check_zone_cover(self, map_: Map) -> Iterable[Event]:
        """
        Check for and return events if previous cover indices were
        partially or fully reduced (flame extinguished).
        """

        event_id, self.cover_indices = check_flame_cover(self.id, self.cover_indices, map_.zone_id)

        if event_id == EventID.OBJECT_EXPIRE:
            return (Event(EventID.FX_EXTINGUISH, self.id), Event(EventID.OBJECT_EXPIRE, self.id))

        elif event_id == EventID.FX_EXTINGUISH:
            return (Event(EventID.FX_EXTINGUISH, self.id), )

        return Event.EMPTY_EVENT_LIST

    def clear_zone_cover(self, map_: Map) -> Iterable[Event]:
        """Clear zone and zone ID maps at still covered indices."""

        clear_zone_cover(self.id, self.cover_indices, map_.zone, map_.zone_id)

        return (Event(EventID.OBJECT_EXPIRE, self.id), )

    def update(self, dt: float, map_: Map, players: list[PlayerEntity], events: list[Event]):
        """
        Evaluate object lifetime and return any resulting events.

        Can include trigger, extinguish, and damage events.
        """

        self.lifetime -= dt

        if self.lifetime > self.item.duration:
            self.update_move(dt, map_, events)

        # Engulf area
        elif not self.triggered:
            self.triggered = True
            events.extend(self.set_zone_cover(map_))

        # Check for past extinguishments
        elif self.lifetime > 0.:
            events_ = self.check_zone_cover(map_)
            self.accumulated_dt += dt

            # 2 events means extinguished and expired, 1 only partially extinguished, 0 no change
            if len(events_) < 2 and self.lifetime <= self.next_eval_time:
                self.next_eval_time -= self.EVAL_INTERVAL
                events.extend(events_)

                self_id = self.id
                base_dmg = self.item.base_damage
                duration = self.item.duration

                # Get damage per player
                for player in players:
                    damage = get_flame_damage(
                        self_id, player.pos, player.covered_indices,
                        base_dmg, duration, self.accumulated_dt, map_.zone_id)

                    if damage != 0.:
                        events.append(
                            Event(EventID.PLAYER_DAMAGE, (self.owner.id, player.id, self.item.id, damage)))

                self.accumulated_dt = 0.

        else:
            events.extend(self.clear_zone_cover(map_))


class Smoke(Object):
    """A smoke grenade item instance."""

    def __init__(self, item: Item, owner: PlayerEntity):
        super().__init__(item, owner, lifetime=item.fuse_time + item.duration)

        self.triggered = False
        self.cover_indices: ndarray = None

    def set_zone_cover(self, map_: Map) -> Iterable[Event]:
        """
        Claim area in zone and zone id maps at cover indices,
        returning trigger and/or other events if cover indices of other
        objects' zones were partially or fully reduced (flame extinguished).
        """

        self.cover_indices = set_smoke_cover(self.id, self.pos, self.item.radius, map_.code_map, map_.zone_id)

        return (Event(EventID.OBJECT_TRIGGER, self.id), )

    def clear_zone_cover(self, map_: Map) -> Iterable[Event]:

        clear_zone_cover(self.id, self.cover_indices, map_.zone, map_.zone_id)

        return (Event(EventID.OBJECT_EXPIRE, self.id), )

    def update(self, dt: float, map_: Map, players: list[PlayerEntity], events: list[Event]):
        """
        Evaluate object lifetime and return any resulting events.

        Can include a trigger event, which can happen prematurely if the
        object has reached the end of its movement trajectory.
        """

        self.lifetime -= dt

        if self.lifetime > self.item.duration:
            if self.pos_target is None:
                self.lifetime = min(self.lifetime, self.item.duration + 0.5)

            else:
                self.update_move(dt, map_, events)

        elif not self.triggered:
            self.triggered = True
            events.extend(self.set_zone_cover(map_))

        elif self.lifetime <= 0.:
            events.extend(self.clear_zone_cover(map_))


class C4(Object):
    """
    An C4 item instance.

    References:
    https://counterstrike.fandom.com/wiki/C4_Explosive
    3kliksphilip: CS:GO's Bomb Explosion Radius | https://www.youtube.com/watch?v=RErEgU1ZeIE
    """

    # NOTE: Should be 40. under the competitive ruleset
    TIME_TO_EXPLODE = 45.
    TIME_TO_DEFUSE = 10.
    TIME_TO_PLANT = 3.2

    # pylint was being weird and could not recognise `TIME_TO_EXPLODE` in this scope...
    BEEP_TIMINGS = [45. - sum(0.980188**i for i in range(j)) for j in range(1, 103)]
    NVG_TIMING = 45. - 44.

    PRESS_PROGRESS_THRESHOLDS = [0.195, 0.285, 0.415, 0.495, 0.6, 0.75, 0.84]
    DEFUSE_DISTANCE_THRESHOLD = 8.

    PLANTING_LANDMARKS: tuple[int] = (MapID.LANDMARK_PLANT_A, MapID.LANDMARK_PLANT_B)

    def __init__(self, item: Item, owner: PlayerEntity):
        super().__init__(item, owner)

        self.beep_timings = deque(self.BEEP_TIMINGS)
        self.press_thresholds = deque(self.PRESS_PROGRESS_THRESHOLDS)

        self.plant_start_time: float = None
        self.plant_progress: float = None

        self.can_be_defused: bool = True
        self.defused_by: int = None
        self.defuse_start_time: float = None
        self.defuse_progress: float = None

    def try_plant(self, pos: ndarray, landmark_map: ndarray, timestamp: float) -> Event | None:
        """
        Start planting if on site or advance it, until planting is completed.
        Key press events can also be returned to indicate progress.
        """

        pos_y, pos_x = round(pos[1]), round(pos[0])

        # Check if the player is positioned at any planting site
        if landmark_map[pos_y, pos_x] in self.PLANTING_LANDMARKS:

            # Update plant
            if self.plant_start_time is not None:
                dt = timestamp - self.plant_start_time
                self.plant_progress = dt / self.TIME_TO_PLANT

            # Start plant
            else:
                self.plant_start_time = timestamp
                self.plant_progress = 0.
                return Event(EventID.FX_C4_INIT, self.owner.id)

        # Break plant
        else:
            self.plant_start_time = None
            self.plant_progress = 0.
            self.press_thresholds = deque(self.PRESS_PROGRESS_THRESHOLDS)
            return None

        # Confirm planted
        if self.plant_progress >= 1.:
            self.defuse_progress = 0.
            self.lifetime = self.TIME_TO_EXPLODE
            return Event(EventID.C4_PLANTED, self)

        # Play key press
        elif self.press_thresholds and self.plant_progress >= self.press_thresholds[0]:
            self.press_thresholds.popleft()
            return Event(EventID.FX_C4_KEY_PRESS, self.owner.id)

        return None

    def try_defuse(self, pos: ndarray, player_id: int, kit_available: bool, timestamp: float) -> Event | None:
        """Check if acted upon by a player, start, update, or end defusing."""

        # Check if the player is near enough to the object
        if vec2_norm2(pos - self.pos) <= self.DEFUSE_DISTANCE_THRESHOLD:

            # Update defuse
            if self.defused_by == player_id:
                dt = timestamp - self.defuse_start_time
                self.defuse_progress = dt / (self.TIME_TO_DEFUSE / 2. if kit_available else self.TIME_TO_DEFUSE)

            # Start defuse
            elif self.defused_by is None:
                self.defuse_start_time = timestamp
                self.defuse_progress = 0.
                self.defused_by = player_id
                return Event(EventID.FX_C4_TOUCHED, self.id)

        # Break defuse
        else:
            self.defuse_start_time = None
            self.defuse_progress = 0.
            self.defused_by = None
            return None

        # Confirm defused
        if self.defuse_progress >= 1.:
            return Event(EventID.C4_DEFUSED, self)

        return None

    def update(self, dt: float, map_: Map, players: list[PlayerEntity], events: list[Event]):
        """Generate associated events, including detonation and beep sounds."""

        self.lifetime -= dt

        # Detonate / expire
        if self.lifetime <= 0.:
            events.append(Event(EventID.C4_DETONATED, self.id))

            self_pos = self.pos
            base_dmg = self.item.base_damage
            radius = self.item.radius

            # Get damager per player
            for player in players:
                damage = get_c4_damage(self_pos, player.pos, base_dmg, radius)

                if damage != 0.:
                    events.append(Event(EventID.PLAYER_DAMAGE, (self.owner.id, player.id, self.item.id, damage)))

            return events.append(Event(EventID.OBJECT_EXPIRE, self.id))

        # If spawned as a dropped/pickupable object
        elif self.lifetime == np.inf:
            return self.update_move(dt, map_, events)

        # If beyond the point of no return
        if not self.can_be_defused:
            return

        elif self.NVG_TIMING >= self.lifetime:
            self.can_be_defused = False
            return events.append(Event(EventID.FX_C4_NVG, self.id))

        # Emmit beep (site and defusing can be inferred client-side)
        if self.beep_timings and self.beep_timings[0] >= self.lifetime:
            self.beep_timings.popleft()

            if self.defused_by is None:
                events.append(Event(EventID.FX_C4_BEEP, self.id))

            else:
                events.append(Event(EventID.FX_C4_BEEP_DEFUSING, self.id))


@jit('float64(float64[:], float64, float64, float64, float64)', nopython=True, nogil=True, cache=True)
def get_halved_inaccuracy(
    vel: ndarray,
    vel_cap: float,
    moving_inaccuracy: float,
    standing_inaccuracy: float,
    initial_inaccuracy: float
) -> float:
    """Interpolate between standing and moving inaccuracy based on current velocity."""

    # NOTE: Exp. 1. for walking, 0.25 for running, 0.5 as a compromise
    alpha = min(1., (max(0., vec2_norm2(vel)/vel_cap - 0.34) / (0.95 - 0.34))**0.5)
    base_inaccuracy = alpha * moving_inaccuracy + (1. - alpha) * standing_inaccuracy

    return (base_inaccuracy + initial_inaccuracy) / 2.


@jit(
    'Tuple((int64, int16, float64[:], float64))'
    '(int16, float64[:], float64, float64, float64, uint8[:, :], int16[:, :])',
    nopython=True, nogil=True, cache=True)
def fire_weapon(
    player_id: int,
    pos: ndarray,
    angle: float,
    base_damage: float,
    distance_modifier: float,
    wall_map: ndarray,
    player_id_map: ndarray
) -> tuple[int, int, ndarray, float]:

    event_id = EventID.NULL
    hit_id = np.int16(MapID.PLAYER_ID_NULL)
    dmg = 0.

    end_pos = pos + np.array((np.cos(angle), np.sin(angle))) * MAX_SHOT_RANGE
    hit_pos_checked = trace_shot(player_id, pos, end_pos, wall_map, player_id_map)

    # NOTE: Returns zeros if reaching target (max range)
    if np.any(hit_pos_checked):
        hit_id = player_id_map[round(hit_pos_checked[1]), round(hit_pos_checked[0])]

        # Check if any entity was hit
        if hit_id != MapID.PLAYER_ID_NULL:
            event_id = EventID.PLAYER_DAMAGE

            # Get damage based on distance to the target
            distance = vec2_norm2(pos - hit_pos_checked)
            dmg = base_damage * distance_modifier ** (distance / DIST_MOD_REF_RANGE)

        else:
            event_id = EventID.FX_WALL_HIT

    return event_id, hit_id, hit_pos_checked, dmg


@jit(
    'Tuple((int64, int16, float64[:]))'
    '(int16, float64[:], float64, uint8[:, :], int16[:, :])',
    nopython=True, nogil=True, cache=True)
def slash_knife(
    player_id: int,
    pos: ndarray,
    angle: float,
    wall_map: ndarray,
    player_id_map: ndarray
) -> tuple[int, int, ndarray]:

    event_id = EventID.NULL
    hit_id = np.int16(MapID.PLAYER_ID_NULL)

    end_pos = pos + np.array((np.cos(angle), np.sin(angle))) * KNIFE_REACH
    hit_pos_checked = trace_shot(player_id, pos, end_pos, wall_map, player_id_map)

    if np.any(hit_pos_checked):
        hit_id = player_id_map[round(hit_pos_checked[1]), round(hit_pos_checked[0])]

        # Check if any entity was hit
        if hit_id != MapID.PLAYER_ID_NULL:
            event_id = EventID.PLAYER_DAMAGE

        else:
            event_id = EventID.FX_WALL_HIT

    return event_id, hit_id, hit_pos_checked


@jit(
    'float64(float64[:], float64[:], float64, uint8[:, :, :], int16[:, :])',
    nopython=True, nogil=True, cache=True)
def get_flash_strength(
    source_pos: ndarray,
    pos: ndarray,
    angle: float,
    code_map: ndarray,
    player_id_map: ndarray
) -> float:
    """
    Get the strength of a flash event wrt. flashed player position, angle,
    and line of sight.
    """

    wall_map = code_map[np.int64(MapID.CHANNEL_WALL)]
    zone_map = code_map[np.int64(MapID.CHANNEL_ZONE)]

    if np.any(trace_sight(MapID.PLAYER_ID_NULL, source_pos, pos, wall_map, zone_map, player_id_map)):
        return 0.

    rel_position = source_pos - pos
    rel_x, rel_y = rel_position
    rel_angle = np.arctan2(rel_y, rel_x)

    viewing_angle = fix_angle_range(angle - rel_angle)
    distance = vec2_norm2(rel_position)

    return (1. - abs(viewing_angle) / F_PI) * max(0., 1. - distance / 216.)


@jit(
    'float64(float64[:], float64[:], float64, float64, uint8[:, :], int16[:, :])',
    nopython=True, nogil=True, cache=True)
def get_he_damage(
    source_pos: ndarray,
    pos: ndarray,
    base_damage: float,
    blast_radius: float,
    wall_map: ndarray,
    player_id_map: ndarray
) -> float:
    """
    Get the damage wrt. player position (distance) and line of sight.

    For simplicity, line of sight is only traced to a player's
    central position, so they can be shielded by a wall even if
    not fully behind it.
    """

    if np.any(trace_shot(MapID.PLAYER_ID_NULL, source_pos, pos, wall_map, player_id_map)):
        return 0.

    distance = vec2_norm2(source_pos - pos)

    if distance > blast_radius:
        return 0.

    return base_damage / (1. + (distance/15.)**3)


@jit(
    'Tuple((int64, UniTuple(int64[:], 2)))(int16, float64[:], float64, uint8[:, :, :], int16[:, :])',
    nopython=True, nogil=True, cache=True)
def set_flame_cover(
    obj_id: int,
    pos: ndarray,
    radius: float,
    code_map: ndarray,
    zone_id_map: ndarray
) -> tuple[int, tuple[ndarray, ndarray]]:

    wall_map = code_map[np.int64(MapID.CHANNEL_WALL)]
    zone_map = code_map[np.int64(MapID.CHANNEL_ZONE)]

    pos_y, pos_x, radius = round(pos[1]), round(pos[0]), round(radius)

    # Get square base (slice lengths should be odd)
    slice_y = slice(pos_y - radius, pos_y + radius + 1)
    slice_x = slice(pos_x - radius, pos_x + radius + 1)

    wall_chunk = wall_map[slice_y, slice_x]

    # Get disk mask
    cover = get_disk_mask(2*radius + 1)

    # Get flammable ground mask
    cover &= ~wall_chunk.astype(np.bool_)

    # Get smoke mask
    zone_chunk = zone_map[slice_y, slice_x]
    smoke = zone_chunk == np.uint8(MapID.ZONE_SMOKE)

    # Get overlap
    overlap = cover & smoke

    # Full overlap
    if np.all(overlap == cover):
        event_id = EventID.OBJECT_EXPIRE

    # Partial overlap
    elif np.any(overlap):
        cover &= ~smoke
        event_id = EventID.FX_EXTINGUISH

    # No overlap
    else:
        event_id = EventID.OBJECT_TRIGGER

    # Get global indices
    indices_y, indices_x = np.nonzero(cover)
    indices_y += pos_y - round(radius)
    indices_x += pos_x - round(radius)
    cover_indices = indices_y, indices_x

    # Set zone cover
    fire_id = np.uint8(MapID.ZONE_FIRE)

    for i in range(len(indices_y)):
        i_y = indices_y[i]
        i_x = indices_x[i]

        zone_map[i_y, i_x] = fire_id
        zone_id_map[i_y, i_x] = obj_id

    return event_id, cover_indices


@jit(
    'Tuple((int64, UniTuple(int64[:], 2)))(int16, UniTuple(int64[:], 2), int16[:, :])',
    nopython=True, nogil=True, cache=True)
def check_flame_cover(
    obj_id: int,
    cover_indices: tuple[ndarray, ndarray],
    zone_id_map: ndarray
) -> tuple[int, tuple[ndarray, ndarray]]:

    event_id = EventID.NULL
    covered_zone_mask = index2_by_tuple(zone_id_map, cover_indices) == obj_id

    # Extinguished
    if not np.any(covered_zone_mask):
        event_id = EventID.OBJECT_EXPIRE

    # Partially extinguished
    elif not np.all(covered_zone_mask):
        event_id = EventID.FX_EXTINGUISH
        cover_indices = cover_indices[0][covered_zone_mask], cover_indices[1][covered_zone_mask]

    return event_id, cover_indices


@jit(
    'void(int16, UniTuple(int64[:], 2), uint8[:, :], int16[:, :])',
    nopython=True, nogil=True, cache=True)
def clear_zone_cover(
    obj_id: int,
    cover_indices: tuple[ndarray, ndarray],
    zone_map: ndarray,
    zone_id_map: ndarray
):
    """Clear zone and zone ID maps at still covered indices."""

    covered_zone_mask = index2_by_tuple(zone_id_map, cover_indices) == obj_id

    if not np.any(covered_zone_mask):
        return

    if np.all(covered_zone_mask):
        indices_y, indices_x = cover_indices

    else:
        indices_y = cover_indices[0][covered_zone_mask]
        indices_x = cover_indices[1][covered_zone_mask]

    null_zone_id = np.uint8(MapID.ZONE_NULL)
    null_obj_id = np.int16(MapID.OBJECT_ID_NULL)

    for i in range(len(indices_y)):
        i_y = indices_y[i]
        i_x = indices_x[i]

        zone_map[i_y, i_x] = null_zone_id
        zone_id_map[i_y, i_x] = null_obj_id


@jit(
    'float64(int16, float64[:], UniTuple(int64[:], 2), float64, float64, float64, int16[:, :])',
    nopython=True, nogil=True, cache=True)
def get_flame_damage(
    obj_id: int,
    pos: ndarray,
    covered_indices: tuple[ndarray, ndarray],
    base_damage: float,
    duration: float,
    dt: float,
    zone_id_map: ndarray
) -> float:
    """
    Get the damage wrt. estimated time that the player has stood on
    the zone cover claimed by this object for.
    """

    pos_y = round(pos[1])
    pos_x = round(pos[0])

    indices_y, indices_x = covered_indices

    for i in range(len(indices_y)):
        i_y = indices_y[i] + pos_y
        i_x = indices_x[i] + pos_x

        if zone_id_map[i_y, i_x] == obj_id:
            return base_damage * dt / duration

    return 0.


@jit(
    'UniTuple(int64[:], 2)(int16, float64[:], float64, uint8[:, :, :], int16[:, :])',
    nopython=True, nogil=True, cache=True)
def set_smoke_cover(
    obj_id: int,
    pos: ndarray,
    radius: float,
    code_map: ndarray,
    zone_id_map: ndarray
) -> tuple[ndarray, ndarray]:
    wall_map = code_map[np.int64(MapID.CHANNEL_WALL)]
    zone_map = code_map[np.int64(MapID.CHANNEL_ZONE)]

    pos_y, pos_x, radius = round(pos[1]), round(pos[0]), round(radius)

    # Get square base (slice lengths should be odd)
    slice_y = slice(pos_y - radius, pos_y + radius + 1)
    slice_x = slice(pos_x - radius, pos_x + radius + 1)

    wall_chunk = wall_map[slice_y, slice_x]

    # Get disk mask
    cover = get_disk_mask(2*radius + 1)

    # Get ground mask
    cover &= ~wall_chunk.astype(np.bool_)

    # Get global indices
    indices_y, indices_x = np.nonzero(cover)
    indices_y += pos_y - round(radius)
    indices_x += pos_x - round(radius)
    cover_indices = indices_y, indices_x

    # Set zone cover
    smoke_id = np.uint8(MapID.ZONE_SMOKE)

    for i in range(len(indices_y)):
        i_y = indices_y[i]
        i_x = indices_x[i]

        zone_map[i_y, i_x] = smoke_id
        zone_id_map[i_y, i_x] = obj_id

    return cover_indices


@jit(
    'float64(float64[:], float64[:], float64, float64)',
    nopython=True, nogil=True, cache=True)
def get_c4_damage(
    source_pos: ndarray,
    pos: ndarray,
    base_damage: float,
    radius: float,
) -> float:
    """
    Get the damage wrt. player position (distance),
    without consideration for line of sight.
    """

    distance = vec2_norm2(source_pos - pos)

    if distance > radius:
        return 0.

    return (base_damage + 30.) / (1. + (distance / 81.5)**3) - 30.
