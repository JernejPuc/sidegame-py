"""Functional extensions of items in SDG"""

from collections import deque
from typing import Dict, Iterable, Optional, Tuple, Union
import numpy as np

from sidegame.ext import sdglib
from sidegame.physics import ThrowableEntity, PlayerEntity
from sidegame.networking.core import Entity
from sidegame.game.shared.core import GameID, Map, Event
from sidegame.game.shared.inventory import Item


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
        entity_id: Optional[int] = 0,
        lifetime: Optional[float] = np.Inf,
        durability: Optional[float] = None,
        magazine: Optional[int] = None,
        reserve: Optional[int] = None,
        carrying: Optional[int] = None
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

    def update(self, dt: float, _players: Iterable[PlayerEntity], _map: Map) -> Iterable[Event]:
        """
        Evaluate object lifetime and return any resulting events.

        Can be overridden to produce additional or different events.
        To this end, the object can interact with given `_players` and `_map`.
        """

        self.lifetime -= dt

        if self.lifetime > 0.:
            return self.update_move(dt, _map)

        return [Event(Event.OBJECT_EXPIRE, self.id)]

    def update_move(self, dt: float, map_: Map) -> Iterable[Event]:
        """Move towards target position, returning any collision events."""

        old_pos = self.pos

        status = self.move(dt, map_.wall, map_.object_id)

        self.update_collider_map(map_.object_id, old_pos, self.pos, claim_id=self.id, clear_id=Map.OBJECT_ID_NULL)

        if status == ThrowableEntity.COLLISION_BOUNCE:
            return [Event(Event.FX_BOUNCE, self.id)]

        elif status == ThrowableEntity.COLLISION_LANDING:
            return [Event(Event.FX_LAND, self.id)]

        else:
            return Event.EMPTY_EVENT_LIST


class Weapon(Object):
    """A primary weapon or pistol instance."""

    MAX_SHOT_RANGE = 640.
    DIST_MOD_REF_RANGE = 56.86

    def __init__(self, item: Item, owner: PlayerEntity, *args, rng: np.random.Generator = None, **kwargs):
        assert item.slot in (Item.SLOT_PRIMARY, Item.SLOT_PISTOL)

        super().__init__(item, owner, *args, **kwargs)

        self.rng = np.random.default_rng() if rng is None else rng

    def fire(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        angle: float,
        firing_inaccuracy: float,
        height_map: np.ndarray,
        player_map: np.ndarray
    ) -> Iterable[Event]:
        """Try to attack, returning damage, hit, and sound events."""

        # Convey empty clip sound
        if self.magazine == 0:
            return [Event(Event.FX_CLIP_EMPTY, self.owner.id)]

        # Convey attack sound
        events = deque()
        events.append(Event(Event.FX_ATTACK, (self.owner.id, self.item.id)))

        # Convey low on ammo sound
        if self.magazine <= self.item.magazine_cap//3:
            events.append(Event(Event.FX_CLIP_LOW, self.owner.id))

        # Subtract ammo charge
        self.magazine -= 1

        # Interpolate base inacuraccies
        # NOTE: Exp. 1. for walking, 0.25 for running, 0.5 as a compromise
        alpha = min(1., (max(0., np.linalg.norm(vel)/self.item.velocity_cap - 0.34) / (0.95 - 0.34))**0.5)
        base_inaccuracy = alpha * self.item.moving_inaccuracy + (1. - alpha) * self.item.standing_inaccuracy
        inaccuracy = base_inaccuracy + firing_inaccuracy

        # Get hits
        for _ in range(self.item.pellets):
            # Randomise angle wrt. inaccuracy
            firing_angle = angle + self.rng.triangular(-inaccuracy/2., 0., inaccuracy/2.)

            endpos = pos + np.array([np.cos(firing_angle), np.sin(firing_angle)]) * self.MAX_SHOT_RANGE
            endpos_check = sdglib.trace_shot(self.owner.id, pos, endpos, height_map, player_map)

            # NOTE: Returns zeros if reaching target (max range)
            if any(endpos_check):
                hit_id = player_map[round(endpos_check[1]), round(endpos_check[0])]

                # Check if any entity was hit
                if hit_id != Map.PLAYER_ID_NULL:
                    dmg = self.get_damage(np.linalg.norm(self.owner.pos - endpos_check))
                    events.append(Event(Event.PLAYER_DAMAGE, (self.owner.id, hit_id, self.item.id, dmg)))

                else:
                    events.append(Event(Event.FX_WALL_HIT, (endpos_check, self.owner.id, self.item.id)))

        return events

    def get_damage(self, distance: float) -> float:
        """Get damage based on distance to the target."""

        return self.item.base_damage * self.item.distance_modifier ** (distance / self.DIST_MOD_REF_RANGE)


class Knife(Object):
    """A knife item instance."""

    REACH = 4. * np.sqrt(2.)

    def __init__(self, item: Item, owner: PlayerEntity):
        assert item.id == GameID.ITEM_KNIFE
        super().__init__(item, owner)

    def slash(
        self,
        pos: np.ndarray,
        height_map: np.ndarray,
        player_map: np.ndarray,
        players: Dict[int, PlayerEntity]
    ) -> Iterable[Event]:
        """Try to attack, returning damage, hit, and sound events."""

        events = deque()

        # Convey attack sound
        events.append(Event(Event.FX_ATTACK, (self.owner.id, self.item.id)))

        hit_id = Map.PLAYER_ID_NULL

        # Get hits
        for swipe_angle in np.array([-np.pi/8., -np.pi/16., 0., np.pi/16., np.pi/8.]) + self.owner.angle:
            endpos = pos + np.array([np.cos(swipe_angle), np.sin(swipe_angle)]) * self.REACH
            endpos_check = sdglib.trace_shot(self.owner.id, pos, endpos, height_map, player_map)

            if any(endpos_check):
                hit_id = player_map[round(endpos_check[1]), round(endpos_check[0])]

                if hit_id == Map.PLAYER_ID_NULL:
                    events.append(Event(Event.FX_WALL_HIT, (endpos_check, self.owner.id, self.item.id)))

        # Check if any entity was hit
        if hit_id != Map.PLAYER_ID_NULL:
            hit_angle = players[hit_id].angle
            dmg = self.get_damage(self.owner.angle, hit_angle)
            events.append(Event(Event.PLAYER_DAMAGE, (self.owner.id, hit_id, self.item.id, dmg)))

        return events

    def get_damage(self, angle: float, hit_angle) -> float:
        """Get damage based on angle of the hit (damage to the back is tripled)."""

        dmg = self.item.base_damage
        d_angle = angle - hit_angle

        if d_angle > self.F_PI:
            d_angle = -self.F_2PI + d_angle
        elif d_angle < -self.F_PI:
            d_angle = self.F_2PI + d_angle

        if abs(d_angle) < self.F_PI2:
            dmg *= 3.

        return dmg


class Flash(Object):
    """
    A flash grenade (flashbang) item instance.

    Reference:
    3kliksphilip: Where Flashbangs flash and bang | https://www.youtube.com/watch?v=aTR7Surb80w
    """

    def __init__(self, item: Item, owner: PlayerEntity):
        assert item.id == GameID.ITEM_FLASH
        super().__init__(item, owner, lifetime=item.fuse_time)

    def update(self, dt: float, _players: Iterable[PlayerEntity], _map: Map) -> Iterable[Event]:
        """
        Evaluate object lifetime and return any resulting events.

        Can include trigger and `FX_FLASH` events,
        the latter containing flash debuffs.
        """

        self.lifetime -= dt

        if self.lifetime > 0.:
            return self.update_move(dt, _map)

        # Get debuff strength per player
        events = deque()

        events.append(Event(Event.OBJECT_TRIGGER, self.id))

        for player in _players:
            if player.health:
                debuff = self.get_debuff_strength(player.pos, player.angle, _map)

                if debuff != 0.:
                    events.append(
                        Event(Event.FX_FLASH, (self.owner.id, player.id, debuff, self.get_debuff_duration(debuff))))

        # Expire upon being triggered
        events.append(Event(Event.OBJECT_EXPIRE, self.id))

        return events

    def get_debuff_strength(self, pos: np.ndarray, angle: float, map_: Map) -> float:
        """
        Get the strength of a flash event wrt. flashed player position, angle,
        and line of sight.
        """

        if any(sdglib.trace_sight(self.owner.id, pos, self.pos, map_.height, map_.player_id, map_.zone)):
            return 0.

        rel_position = self.pos - pos
        rel_x, rel_y = rel_position
        rel_angle = np.arctan2(rel_y, rel_x)

        viewing_angle = self.fix_angle_range(angle - rel_angle)
        distance = np.linalg.norm(rel_position)

        return (1. - np.abs(viewing_angle) / np.pi) * max(0., 1. - distance / 216.)

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
        assert item.id == GameID.ITEM_EXPLOSIVE
        super().__init__(item, owner, lifetime=item.fuse_time)

    def update(self, dt: float, _players: Iterable[PlayerEntity], _map: Map) -> Iterable[Event]:
        """
        Evaluate object lifetime and return any resulting events.

        Can include trigger and damage events.
        """

        self.lifetime -= dt

        if self.lifetime > 0.:
            return self.update_move(dt, _map)

        # Get damage per player
        events = deque()

        events.append(Event(Event.OBJECT_TRIGGER, self.id))

        for player in _players:
            if player.health:
                damage = self.get_damage(player.pos, _map)

                if damage != 0.:
                    events.append(Event(Event.PLAYER_DAMAGE, (self.owner.id, player.id, self.item.id, damage)))

        # Expire upon being triggered
        events.append(Event(Event.OBJECT_EXPIRE, self.id))

        return events

    def get_damage(self, pos: np.ndarray, _map: Map) -> float:
        """
        Get the damage wrt. player position (distance) and line of sight.

        For simplicity, line of sight is only traced to a player's
        central position, so it can be shielded by a wall even if
        not fully behind it.
        """

        if any(sdglib.trace_shot(self.owner.id, pos, self.pos, _map.height, _map.player_id)):
            return 0.

        distance = np.linalg.norm(self.pos - pos)

        if distance > self.item.radius:
            return 0.

        return self.item.base_damage / (1. + (distance/15.)**3)


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
        assert item.id in (GameID.ITEM_INCENDIARY_T, GameID.ITEM_INCENDIARY_CT)
        super().__init__(item, owner, lifetime=(item.fuse_time + item.duration))

        self.triggered = False
        self.cover_indices: Tuple[np.ndarray] = None
        self.next_eval_time = self.item.duration
        self.accumulated_dt = 0.

    def set_zone_cover(self, wall_map: np.ndarray, zone_map: np.ndarray, zone_id_map: np.ndarray) -> Iterable[Event]:
        """
        Claim area in zone and zone ID maps at cover indices,
        returning trigger and/or other events if cover indices were
        partially or fully reduced (flame extinguished).
        """

        pos_y, pos_x, radius = round(self.pos[1]), round(self.pos[0]), round(self.item.radius)

        # Get square base (slice lengths should be odd)
        slice_y = slice(pos_y - radius, pos_y + radius + 1)
        slice_x = slice(pos_x - radius, pos_x + radius + 1)

        wall_chunk = wall_map[slice_y, slice_x]

        # Get disk mask
        indices = np.indices(wall_chunk.shape) - np.array([radius, radius])[..., None, None]
        cover = np.linalg.norm(indices, axis=0) < self.item.radius

        # Get flammable ground mask
        cover &= ~wall_chunk.astype(np.bool)

        # Get smoke mask
        zone_chunk = zone_map[slice_y, slice_x]
        smoke = zone_chunk == Map.ZONE_SMOKE

        # Get overlap
        overlap = cover & smoke

        # Full overlap
        if np.all(overlap == cover):
            return [Event(Event.FX_EXTINGUISH, self.id), Event(Event.OBJECT_EXPIRE, self.id)]

        # Partial overlap
        elif np.any(overlap):
            cover &= ~smoke
            events = [Event(Event.FX_EXTINGUISH, self.id), Event(Event.OBJECT_TRIGGER, self.id)]

        # No overlap
        else:
            events = [Event(Event.OBJECT_TRIGGER, self.id)]

        # Get global indices
        indices_y, indices_x = np.where(cover)
        indices_y -= int(radius)
        indices_x -= int(radius)
        self.cover_indices = indices_y + pos_y, indices_x + pos_x

        # Set zone cover
        zone_map[self.cover_indices] = Map.ZONE_FIRE
        zone_id_map[self.cover_indices] = self.id

        return events

    def check_zone_cover(self, zone_map: np.ndarray, zone_id_map: np.ndarray) -> Iterable[Event]:
        """
        Check for and return events if previous cover indices were
        partially or fully reduced (flame extinguished).
        """

        covered_zone = (zone_id_map[self.cover_indices] == self.id) | (zone_map[self.cover_indices] != Map.ZONE_SMOKE)

        # Extinguished
        if not np.any(covered_zone):
            return [Event(Event.FX_EXTINGUISH, self.id), Event(Event.OBJECT_EXPIRE, self.id)]

        # Partially extinguished
        elif not np.all(covered_zone):
            self.cover_indices = self.cover_indices[0][covered_zone], self.cover_indices[1][covered_zone]
            return [Event(Event.FX_EXTINGUISH, self.id)]

        # Nothing new
        return Event.EMPTY_EVENT_LIST

    def clear_zone_cover(self, zone_map: np.ndarray, zone_id_map: np.ndarray) -> Iterable[Event]:
        """Clear zone and zone ID maps at still covered indices."""

        covered_zone = zone_id_map[self.cover_indices] == self.id
        cover_indices = self.cover_indices[0][covered_zone], self.cover_indices[1][covered_zone]

        zone_map[cover_indices] = Map.ZONE_NULL
        zone_id_map[cover_indices] = Map.OBJECT_ID_NULL

        return [Event(Event.OBJECT_EXPIRE, self.id)]

    def update(self, dt: float, _players: Iterable[PlayerEntity], _map: Map) -> Iterable[Event]:
        """
        Evaluate object lifetime and return any resulting events.

        Can include trigger, extinguish, and damage events.
        """

        self.lifetime -= dt

        if self.lifetime > self.item.duration:
            return self.update_move(dt, _map)

        # Engulf area
        elif not self.triggered:
            self.triggered = True
            return self.set_zone_cover(_map.wall, _map.zone, _map.zone_id)

        # Check for past extinguishments
        elif self.lifetime > 0.:
            events = self.check_zone_cover(_map.zone, _map.zone_id)
            self.accumulated_dt += dt

            # 2 events means extinguished and expired, 1 only partially extinguished, 0 no change
            if len(events) < 2 and self.lifetime <= self.next_eval_time:
                self.next_eval_time -= self.EVAL_INTERVAL
                events = deque(events)

                # Get damage per player
                for player in _players:
                    if player.health:
                        damage = self.get_damage(player, self.accumulated_dt, _map.zone_id)

                        if damage != 0.:
                            events.append(Event(Event.PLAYER_DAMAGE, (self.owner.id, player.id, self.item.id, damage)))

                self.accumulated_dt = 0.

            return events

        else:
            return self.clear_zone_cover(_map.zone, _map.zone_id)

    def get_damage(self, player: PlayerEntity, dt: float, zone_id_map: np.ndarray) -> float:
        """
        Get the damage wrt. estimated time that the player has stood on
        the zone cover claimed by this object for.
        """

        if np.any(zone_id_map[player.get_covered_indices()] == self.id):
            return self.item.base_damage * dt / self.item.duration

        return 0.


class Smoke(Object):
    """A smoke grenade item instance."""

    def __init__(self, item: Item, owner: PlayerEntity):
        assert item.id == GameID.ITEM_SMOKE
        super().__init__(item, owner, lifetime=(item.fuse_time + item.duration))

        self.triggered = False
        self.cover_indices: np.ndarray = None

    def set_zone_cover(self, wall_map: np.ndarray, zone_map: np.ndarray, zone_id_map: np.ndarray) -> Iterable[Event]:
        """
        Claim area in zone and zone id maps at cover indices,
        returning trigger and/or other events if cover indices of other
        objects' zones were partially or fully reduced (flame extinguished).
        """

        pos_y, pos_x, radius = round(self.pos[1]), round(self.pos[0]), round(self.item.radius)

        # Get square base (slice lengths should be odd)
        slice_y = slice(pos_y - radius, pos_y + radius + 1)
        slice_x = slice(pos_x - radius, pos_x + radius + 1)

        wall_chunk = wall_map[slice_y, slice_x]

        # Get disk mask
        indices = np.indices(wall_chunk.shape) - np.array([radius, radius])[..., None, None]
        cover = np.linalg.norm(indices, axis=0) < self.item.radius

        # Get ground mask
        cover &= ~wall_chunk.astype(np.bool)

        # Get global indices
        indices_y, indices_x = np.where(cover)
        indices_y -= int(radius)
        indices_x -= int(radius)
        self.cover_indices = indices_y + pos_y, indices_x + pos_x

        # Set zone cover
        zone_map[self.cover_indices] = Map.ZONE_SMOKE
        zone_id_map[self.cover_indices] = self.id

        return [Event(Event.OBJECT_TRIGGER, self.id)]

    def clear_zone_cover(self, zone_map: np.ndarray, zone_id_map: np.ndarray) -> Iterable[Event]:
        """Clear zone and zone ID maps at still covered indices."""

        covered_zone = zone_id_map[self.cover_indices] == self.id
        cover_indices = self.cover_indices[0][covered_zone], self.cover_indices[1][covered_zone]

        zone_map[cover_indices] = Map.ZONE_NULL
        zone_id_map[cover_indices] = Map.OBJECT_ID_NULL

        return [Event(Event.OBJECT_EXPIRE, self.id)]

    def update(self, dt: float, _players: Iterable[PlayerEntity], _map: Map) -> Iterable[Event]:
        """
        Evaluate object lifetime and return any resulting events.

        Can include a trigger event.
        """

        self.lifetime -= dt

        if self.lifetime > self.item.duration:
            return self.update_move(dt, _map)

        elif not self.triggered:
            self.triggered = True
            return self.set_zone_cover(_map.wall, _map.zone, _map.zone_id)

        elif self.lifetime > 0.:
            return Event.EMPTY_EVENT_LIST

        else:
            return self.clear_zone_cover(_map.zone, _map.zone_id)


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

    PLANTING_LANDMARKS: Tuple[int] = (Map.LANDMARK_PLANT_A, Map.LANDMARK_PLANT_B)

    def __init__(self, item: Item, owner: PlayerEntity):
        assert item.id == GameID.ITEM_C4
        super().__init__(item, owner)

        self.beep_timings = deque(self.BEEP_TIMINGS)
        self.press_thresholds = deque(self.PRESS_PROGRESS_THRESHOLDS)

        self.plant_start_time: float = None
        self.plant_progress: float = None

        self.can_be_defused: bool = True
        self.defused_by: int = None
        self.defuse_start_time: float = None
        self.defuse_progress: float = None

    def try_plant(self, pos: np.ndarray, landmark_map: np.ndarray, focus: bool, timestamp: float) -> Union[Event, None]:
        """
        Start planting if on site or advance it, until planting is completed.
        Key press events can also be returned to indicate progress.
        """

        pos_y, pos_x = round(pos[1]), round(pos[0])

        if landmark_map[pos_y, pos_x] in self.PLANTING_LANDMARKS and focus:
            if self.plant_start_time is not None:
                # Update_plant
                dt = timestamp - self.plant_start_time
                self.plant_progress = dt / self.TIME_TO_PLANT

            else:
                # Start plant
                self.plant_start_time = timestamp
                self.plant_progress = 0.
                return Event(Event.FX_C4_INIT, self.owner.id)

        else:
            # End plant
            self.plant_start_time = None
            self.plant_progress = 0.
            self.press_thresholds = deque(self.PRESS_PROGRESS_THRESHOLDS)
            return None

        # Confirm planted
        if self.plant_progress >= 1.:
            self.defuse_progress = 0.
            self.lifetime = self.TIME_TO_EXPLODE
            return Event(Event.C4_PLANTED, self)

        # Play key press
        elif self.press_thresholds and self.plant_progress >= self.press_thresholds[0]:
            del self.press_thresholds[0]
            return Event(Event.FX_C4_KEY_PRESS, self.owner.id)

        return None

    def update(self, dt: float, _players: Iterable[PlayerEntity], _map: Map) -> Iterable[Event]:
        """
        Check if acted upon by a player, start, update, or end defuse,
        and generate associated events, including detonation and beep sounds.
        """

        self.lifetime -= dt

        # Detonate / expire
        if self.lifetime <= 0.:
            events = deque()
            events.append(Event(Event.C4_DETONATED, self.id))

            # Get damager per player
            for player in _players:
                if player.health:
                    damage = self.get_damage(player.pos, _map)

                    if damage != 0.:
                        events.append(Event(Event.PLAYER_DAMAGE, (self.owner.id, player.id, self.item.id, damage)))

            events.append(Event(Event.OBJECT_EXPIRE, self.id))
            return events

        # If spawned as a dropped/pickupable object
        elif self.lifetime == np.Inf:
            return self.update_move(dt, _map)

        # If beyond the point of no return
        if not self.can_be_defused:
            return Event.EMPTY_EVENT_LIST

        elif self.NVG_TIMING >= self.lifetime:
            self.can_be_defused = False
            return [Event(Event.FX_C4_NVG, self.id)]

        events = deque()

        # Emmit beep (site can be inferred client-side)
        if self.beep_timings and self.beep_timings[0] >= self.lifetime:
            del self.beep_timings[0]

            if self.defused_by is None:
                events.append(Event(Event.FX_C4_BEEP, self.id))
            else:
                events.append(Event(Event.FX_C4_BEEP_DEFUSING, self.id))

        for player in _players:
            if player.team == GameID.GROUP_TEAM_CT and player.actions:
                last_action = player.actions[-1]
                timestamp = last_action.timestamp
                pos, focus, kit = last_action.data

                focus = focus and np.linalg.norm(pos - self.pos) <= self.DEFUSE_DISTANCE_THRESHOLD

                if self.defused_by == player.id:
                    # Update defuse
                    if focus:
                        dt = timestamp - self.defuse_start_time
                        self.defuse_progress = dt / (self.TIME_TO_DEFUSE / 2. if kit else self.TIME_TO_DEFUSE)

                    # Break defuse
                    else:
                        self.defused_by = None
                        self.defuse_start_time = None
                        self.defuse_progress = 0.

                # Start defuse
                elif self.defused_by is None and focus:
                    self.defused_by = player.id
                    self.defuse_start_time = timestamp
                    self.defuse_progress = 0.
                    events.append(Event(Event.FX_C4_TOUCHED, self.id))

        # Complete defuse
        if self.defuse_progress >= 1.:
            events.append(Event(Event.C4_DEFUSED, self))
            events.append(Event(Event.OBJECT_EXPIRE, self.id))

        return events

    def get_damage(self, pos: np.ndarray, _map: Map) -> float:
        """
        Get the damage wrt. player position (distance),
        without consideration for line of sight.
        """

        distance = np.linalg.norm(self.pos - pos)

        if distance > self.item.radius:
            return 0.

        return (self.item.base_damage + 30.) / (1. + (distance/81.5)**3) - 30.
