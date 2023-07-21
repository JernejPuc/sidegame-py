"""
Definitions of entities that are realised in the game world.

Originally used external binaries, mainly to perform pixel iteration on a lower level.
Now temporarily includes some other jitted utilities.
"""

import math

import numpy as np
from numpy import ndarray
from numba import jit

from sidegame.game import EventID, MapID
from sidegame.utils_jit import (
    get_centred_indices, vec2_norm2, fix_angle_range, init_bresenham, step_bresenham,
    F_PI, F_PI2, F_NPI4, F_N3PI4, F_SQRT2)


_NULL_INIT_VEL = np.array((np.nan, np.nan))
_BOUNCE_VEL_PRESERVATION = math.sqrt(2.) / 2.
_VERTICAL_KERNEL = np.array(((-1, -1, -1), (0, 0, 0), (1, 1, 1)), dtype=np.int16)
_HORIZONTAL_KERNEL = np.array(((-1, 0, 1), (-1, 0, 1), (-1, 0, 1)), dtype=np.int16)

MAX_VIEW_RANGE = 106.5
MOUSE_MVMT_TO_ANGLE = np.arctan(1. / MAX_VIEW_RANGE)

_ACCELERATING_MASS = 1. / 5.5
_MAX_DT = 0.75 * _ACCELERATING_MASS


class OrientedEntity:
    """
    An entity with attributes and methods for determining its position and angle.
    As such, it is sufficient for audio sources and listeners, and serves
    as a parent class for other types of entities.

    Rounded position coordinates map to pixel centres, so that, for example,
    (1., 1.) corresponds to the pixel at indices (1, 1), while lines
    (x, 0.5), (x, 1.5), (0.5, y), and (1.5, y) represent its borders
    and middle points to neighbouring pixels.

    Coordinates are expected to increase rightwards (x) and upwards (y).
    Using numpy 2D-array indices as discrete coordinate points,
    the vertical (y) axis becomes inverted, basically flipping the world
    over the x axis.

    To retain consistency, this inversion is reflected in the angle definition
    as well. That is, considering the angles of 0 or 2Pi on the horizontal right
    and Pi or -Pi on the horizontal left, the angle must decrease when rotated
    counter-clockwise, and increase when rotated clockwise.
    """

    def __init__(self):
        self.pos = np.array((0., 0.))
        self.angle = 0.


class ColliderEntity(OrientedEntity):
    """
    An entity that takes up space in the world and can thus be involved in
    collisions.

    NOTE: Entity width should be odd, because even-width colliders can be
    positioned in multiple ways without a clear central pixel, which can
    cause issues in collision handling or visual representation.
    """

    def __init__(self, width: int):
        super().__init__()

        self.halfwidth = width//2
        self.covered_indices = get_centred_indices(width)

    def get_position_indices(self) -> tuple[int, int]:
        """
        Get the rounded positions corresponding to indices on the vertical
        and horizontal axes (numpy order).

        Entity position can be explicitly specified to cover cases where
        multiple, e.g. older, positions are stored and being dealt with.
        """

        return get_position_indices(self.pos)

    def get_covered_indices(self) -> tuple[ndarray, ndarray]:
        """
        Get the indices on the vertical and horizontal axes that are
        approximately covered by the entity.

        Entity position can be explicitly specified to cover cases where
        multiple, e.g. older, positions are stored and being dealt with.
        """

        return get_covered_indices(self.covered_indices, self.pos)

    def update_collider_map(
        self,
        collider_map: ndarray,
        old_pos: ndarray,
        new_pos: ndarray,
        claim_id: int,
        clear_id: int
    ):
        """
        Update collider map when moving or rewinding (basically, move the hitbox).
        Interaction is restricted to indices that belong to claimed (or clear) ID.
        """

        return update_collider_map(self.covered_indices, collider_map, old_pos, new_pos, claim_id, clear_id)


class ThrowableEntity(ColliderEntity):
    """A pixel-wide object entity that can be thrown and bounce off of walls."""

    def __init__(self, object_id: int = MapID.OBJECT_ID_NULL):
        super().__init__(1)

        self.id = object_id
        self.pos_target = np.array((np.nan,)*2)
        self.vel = np.array((0., 0.))

    def throw(self, pos_thrower: ndarray, pos_target: ndarray, init_throw_vel: float, init_vel: ndarray = None):
        """
        Set initial velocity and target position of the thrown object.

        If initial velocity is provided, e.g. to simulate momentum,
        it is added to throw-induced velocity and the target position
        is moved by 1-second travel distance.
        """

        if init_vel is None:
            init_vel = _NULL_INIT_VEL

        self.pos = pos_thrower
        self.pos_target, self.vel = throw_object(pos_thrower, pos_target, init_throw_vel, init_vel)

    def move(self, dt: float, wall_map: ndarray, object_map: ndarray) -> int:
        """
        Move the object forward in time by time step `dt`.

        The update can return different status flags, which can be relayed
        and used to trigger external events and effects.
        """

        self.pos, self.pos_target, self.vel, event_id = move_object(
            self.id, self.pos, self.pos_target, self.vel, dt, wall_map, object_map)

        return event_id


class PlayerEntity(ColliderEntity):
    """
    An entity that can move itself.

    It does not bounce, but rather comes to a halt on collision,
    and has a larger collider (width).

    Instead of relaying collision checks, it keeps track of the current
    position within a connected graph of movement states and handles
    transitions between them.

    NOTE: There is a discrepancy between which pixels are checked to
    determine collisions and which are covered by the sprite.
    This is because player sprites have even (4px) width, which can be
    positioned in multiple ways without a clear central pixel.
    Instead of requiring collision handling to check the cleared and claimed
    areas on each movement update, visual representation is compromised instead,
    by using a consistent collider with smaller, but odd, width (3px).
    """

    STATE_STILL = 0
    STATE_WALKING = 1
    STATE_RUNNING = 2

    _WALK_RESET_DISTANCE = 12.25
    _STILLNESS_THRESHOLD = 1.

    def __init__(self, player_id: int = MapID.PLAYER_ID_NULL, rng: np.random.Generator = None):
        super().__init__(3)

        self.id = player_id
        self.rng = np.random.default_rng() if rng is None else rng

        self.vel = np.array((0., 0.))
        self.acc = np.array((0., 0.))

        self.state = self.STATE_STILL
        self.run_distance = 0.

    def move(
        self,
        dt: float,
        force_w: int,
        force_d: int,
        mouse_hor: int,
        walking: bool,
        max_vel: float,
        height_map: ndarray,
        player_id_map: ndarray
    ) -> int:
        """
        Move the player forward in time by time step `dt`.

        The update can return a boolean flag for a footstep event, which can be
        relayed and used to trigger external events and effects.

        NOTE: After moving, the entity map should be updated externally.
        """

        self.pos, self.vel, self.acc, self.angle, step_dist, vel_norm = move_player(
            self.id, self.pos, self.vel, self.acc, self.angle,
            dt, max_vel, force_w, force_d, mouse_hor, height_map, player_id_map)

        # Progress the movement state
        return self.advance_step(step_dist, vel_norm, walking)

    def advance_step(self, distance: float, velocity: float, walking: bool) -> bool:
        """
        Movement 'animation' formulated as a finite-state machine.

        Distance travelled in the running state is being tracked,
        resetting and raising a flag corresponding to a footstep event
        after passing a certain threshold.

        To prevent the ability of purposefully resetting the distance just
        before triggering it, transitions into the running state should
        begin with a footstep, but this causes issues with many sounds being
        played when repeatedly transitioning between states, e.g. by walking
        into a wall. Hence, some randomness and past distance travelled are used
        instead.
        """

        if self.state == self.STATE_STILL:
            if velocity > self._STILLNESS_THRESHOLD:
                if walking:
                    self.state = self.STATE_WALKING

                else:
                    self.state = self.STATE_RUNNING
                    self.run_distance = self.rng.random() * self._WALK_RESET_DISTANCE

        elif self.state == self.STATE_WALKING:
            if velocity <= self._STILLNESS_THRESHOLD:
                self.state = self.STATE_STILL

            elif not walking:
                self.state = self.STATE_RUNNING
                self.run_distance += self.rng.random() * self._WALK_RESET_DISTANCE / 2.

        else:
            self.run_distance += distance

            if velocity <= self._STILLNESS_THRESHOLD:
                self.state = self.STATE_STILL
                self.run_distance = 0.

            elif walking:
                self.state = self.STATE_WALKING

            elif self.run_distance >= self._WALK_RESET_DISTANCE:
                self.run_distance = 0.
                return True

        return False

    def get_focal_point(
        self, wall_map: ndarray, zone_map: ndarray, entity_map: ndarray, max_range: float = 1e+3
    ) -> ndarray:
        """
        Get the farthest unobstructed point directly in front of the player.

        NOTE: Returns zeros if max range was reached. Any handling of this
        should be done externally.
        """

        endpoint = self.pos + np.array((math.cos(self.angle), math.sin(self.angle))) * max_range

        return trace_sight(self.id, self.pos, endpoint, wall_map, zone_map, entity_map)


@jit('UniTuple(int64, 2)(float64[:])', nopython=True, nogil=True, cache=True)
def get_position_indices(pos: ndarray) -> tuple[int, int]:
    return round(pos[1]), round(pos[0])


@jit('UniTuple(int64[:], 2)(UniTuple(int64[:], 2), float64[:])', nopython=True, nogil=True, cache=True)
def get_covered_indices(covered_indices: tuple[ndarray, ndarray], pos: ndarray) -> tuple[ndarray, ndarray]:
    pos_y, pos_x = get_position_indices(pos)

    return covered_indices[0] + pos_y, covered_indices[1] + pos_x


@jit(
    'void(UniTuple(int64[:], 2), int16[:, :], float64[:], float64[:], int16, int16)',
    nopython=True, nogil=True, cache=True)
def update_collider_map(
    covered_indices: ndarray,
    collider_map: ndarray,
    old_pos: ndarray,
    new_pos: ndarray,
    claim_id: int,
    clear_id: int
):
    old_pos_y, old_pos_x = get_position_indices(old_pos)
    new_pos_y, new_pos_x = get_position_indices(new_pos)

    # Update entity map if covered area has changed
    if old_pos_y == new_pos_y and old_pos_x == new_pos_x and collider_map[old_pos_y, old_pos_x] != clear_id:
        return

    old_covered_indices_y = covered_indices[0] + old_pos_y
    old_covered_indices_x = covered_indices[1] + old_pos_x

    new_covered_indices_y = covered_indices[0] + new_pos_y
    new_covered_indices_x = covered_indices[1] + new_pos_x

    # Clear currently covered area
    for i in range(len(old_covered_indices_y)):
        i_y = old_covered_indices_y[i]
        i_x = old_covered_indices_x[i]

        if collider_map[i_y, i_x] == claim_id:
            collider_map[i_y, i_x] = clear_id

    # Claim newly covered area
    for i in range(len(new_covered_indices_y)):
        i_y = new_covered_indices_y[i]
        i_x = new_covered_indices_x[i]

        if collider_map[i_y, i_x] == clear_id:
            collider_map[i_y, i_x] = claim_id


@jit('UniTuple(float64[:], 2)(float64[:], float64[:], float64, float64[:])', nopython=True, nogil=True, cache=True)
def throw_object(
    pos_thrower: ndarray,
    pos_target: ndarray,
    init_throw_vel: float,
    init_vel: ndarray
):
    pos_x, pos_y = pos_thrower
    pos_x_target, pos_y_target = pos_target

    throw_angle = np.arctan2(pos_y_target - pos_y, pos_x_target - pos_x)
    throw_dir = np.array((np.cos(throw_angle), np.sin(throw_angle)))

    vel = throw_dir * init_throw_vel

    if not np.isnan(init_vel[0]):
        pos_target += init_vel
        vel += init_vel

    return pos_target, vel


@jit('float64[:](int16, float64[:], float64[:], uint8[:, :], int16[:, :])', nopython=True, nogil=True, cache=True)
def check_object_path(
    self_id: int,
    pos_0: ndarray,
    pos_1: ndarray,
    wall_map: ndarray,
    object_map: ndarray
) -> ndarray:
    """
    Move object with `self_id` from `pos_0` to `pos_1`.
    Returns zeros on success and last valid point on collision.
    """

    x1, y1, dx, dy, tx, ty, sx, sy, e = init_bresenham(pos_0, pos_1)

    tx_prev = tx
    ty_prev = ty
    wall = 0
    object_ = 0

    pos_1_checked = np.zeros(2)

    # Trace up to target or collision
    while True:
        wall = wall_map[ty, tx]
        object_ = object_map[ty, tx]

        if wall == MapID.MASK_TERRAIN_WALL or (object_ != MapID.OBJECT_ID_NULL and object_ != self_id):
            pos_1_checked[0] = tx_prev
            pos_1_checked[1] = ty_prev
            break

        else:
            tx_prev = tx
            ty_prev = ty

        if tx == x1 and ty == y1:
            break

        tx, ty, e = step_bresenham(dx, dy, tx, ty, sx, sy, e)

    return pos_1_checked


@jit('UniTuple(float64[:], 2)(float64[:], float64[:], float64[:], uint8[:, :])', nopython=True, nogil=True, cache=True)
def bounce(
    pos_hit: ndarray,
    pos_target: ndarray,
    vel: ndarray,
    wall_map: ndarray
) -> tuple[ndarray, ndarray]:
    """
    Use two kernels, arctan2, and some conditions to 'classify' the area
    around the hit pixel and roughly determine the angle of reflection,
    which is then used to set the new endpoint (target of movement).

    Velocity magnitude is also decreased to simulate the effects of
    (non-)elasticity and friction.

    Kernels are defined so that a wall on the right, with incoming normal
    pointing rightward, has normal angle 0, and that the wall above, with
    incoming normal pointing upward (flipped world downward), has normal
    angle -Pi/2.
    """

    pos_x_hit, pos_y_hit = pos_hit
    pos_x_target, pos_y_target = pos_target

    # Get correlations
    hit_wall_nbhood = wall_map[round(pos_y_hit)-1:round(pos_y_hit)+2, round(pos_x_hit)-1:round(pos_x_hit)+2]

    corr_y = np.sum(_VERTICAL_KERNEL * hit_wall_nbhood)
    corr_x = np.sum(_HORIZONTAL_KERNEL * hit_wall_nbhood)

    # Infer angle of wall's incoming normal
    wall_angle = np.arctan2(corr_y, corr_x)

    # Modify wall angle to conform to legacy conditions
    # (edge case handling seemed more intuitive under a different definition)
    wall_angle -= F_PI2

    # Get difference relative to wall angle
    incoming_angle = np.arctan2(pos_y_target - pos_y_hit, pos_x_target - pos_x_hit)
    angle_diff = fix_angle_range(incoming_angle - wall_angle)

    # Handle edge cases
    if 0. < angle_diff < F_PI:
        bounce_angle = 2.*wall_angle - incoming_angle

    elif F_NPI4 < angle_diff <= 0.:
        bounce_angle = 2.*wall_angle - incoming_angle - F_PI2

    elif (angle_diff == F_PI) or (-F_PI <= angle_diff < F_N3PI4):
        bounce_angle = 2.*wall_angle - incoming_angle + F_PI2

    else:
        bounce_angle = 2.*wall_angle - incoming_angle + F_PI

    # Get new direction
    bounce_dir = np.array((np.cos(bounce_angle), np.sin(bounce_angle)))

    # Set new target
    path_remaining = vec2_norm2(pos_target - pos_hit)
    new_pos_target = pos_hit + bounce_dir * path_remaining

    # Set new velocity
    new_vel = bounce_dir * vec2_norm2(vel) * _BOUNCE_VEL_PRESERVATION

    return new_pos_target, new_vel


@jit(
    'Tuple((float64[:], float64[:], float64[:], uint8))'
    '(int16, float64[:], float64[:],  float64[:], float64, uint8[:, :], int16[:, :])',
    nopython=True, nogil=True, cache=True)
def move_object(
    obj_id: int,
    pos: ndarray,
    pos_target: ndarray,
    vel: ndarray,
    dt: float,
    wall_map: ndarray,
    object_map: ndarray
) -> tuple[ndarray, ndarray, ndarray, int]:

    # Return if no movement issued
    if np.isnan(pos_target[0]):
        return pos, pos_target, vel, EventID.NULL

    # Predict new position
    new_pos = pos + vel * dt

    # End movement on overshoot or if velocity indicates too many bounces
    if (
        vec2_norm2(new_pos - pos_target) >= vec2_norm2(pos - pos_target) or
        vec2_norm2(vel) < 1.
    ):
        pos_target[0] = np.nan
        vel[:] = 0.

        return pos, pos_target, vel, EventID.FX_LAND

    # Check for and handle collision
    new_pos_checked = check_object_path(obj_id, pos, new_pos, wall_map, object_map)

    if np.any(new_pos_checked):
        pos_target, vel = bounce(new_pos_checked, pos_target, vel, wall_map)

        return new_pos_checked, pos_target, vel, EventID.FX_BOUNCE

    else:
        return new_pos, pos_target, vel, EventID.NULL


@jit('float64[:](int16, float64[:], float64[:], uint8[:, :], int16[:, :])', nopython=True, nogil=True, cache=True)
def check_player_path(
    self_id: int,
    pos_0: ndarray,
    pos_1: ndarray,
    height_map: ndarray,
    player_id_map: ndarray
) -> ndarray:
    """
    Move player with `self_id` from `pos_0` to `pos_1`.
    Returns zeros on success and last valid point on collision.
    """

    x1, y1, dx, dy, tx, ty, sx, sy, e = init_bresenham(pos_0, pos_1)

    tx_prev = tx
    ty_prev = ty
    terrain = 0
    player_id = 0

    in_transition = False
    collision_detected = False

    pos_1_check = np.zeros(2)

    # Check 9 core pixels (3x3) for all possibilities
    for i in range(-1, 2):
        for j in range(-1, 2):
            terrain = height_map[ty + i, tx + j]

            if terrain >= MapID.HEIGHT_TRANSITION:
                in_transition = True
                break

        else:
            continue

        break

    # Trace up to target or collision
    while True:
        for i in range(-1, 2):
            for j in range(-1, 2):
                terrain = height_map[ty + i, tx + j]
                player_id = player_id_map[ty + i, tx + j]

                if (
                    (player_id != MapID.PLAYER_ID_NULL and player_id != self_id)
                    or terrain > MapID.HEIGHT_ELEVATED
                    or (terrain == MapID.HEIGHT_ELEVATED and not in_transition)
                ):
                    collision_detected = True
                    break

            else:
                continue

            break

        if collision_detected:
            pos_1_check[0] = tx_prev
            pos_1_check[1] = ty_prev
            break

        else:
            tx_prev = tx
            ty_prev = ty

        if tx == x1 and ty == y1:
            break

        tx, ty, e = step_bresenham(dx, dy, tx, ty, sx, sy, e)

    return pos_1_check


@jit(
    'Tuple((float64[:], float64[:], float64[:], float64, float64, float64))'
    '(int16, float64[:], float64[:], float64[:], '
    'float64, float64, float64, int64, int64, int64, uint8[:, :], int16[:, :])',
    nopython=True, nogil=True, cache=True)
def move_player(
    player_id: int,
    pos: ndarray,
    vel: ndarray,
    acc: ndarray,
    angle: float,
    dt: float,
    max_vel: float,
    force_w: int,
    force_d: int,
    mouse_hor: int,
    height_map: ndarray,
    player_id_map: ndarray
) -> tuple[ndarray, ndarray, ndarray, float, float]:

    # Convert mouse movement to angle difference and add it to current angle
    angle = fix_angle_range(angle + (mouse_hor * MOUSE_MVMT_TO_ANGLE))

    # Enforce consistent magnitude of accelerating force
    if force_w and force_d:
        force_w /= F_SQRT2
        force_d /= F_SQRT2

    # Get accelerating force in the global system
    force_y = np.sin(angle)*force_w + np.cos(angle)*force_d
    force_x = np.cos(angle)*force_w - np.sin(angle)*force_d

    # Self-regulate accelerating force and get acceleration
    # Accelerating force is formulated as a function of maximum velocity,
    # while force of friction is formulated as a function of current velocity
    acc[1] = (force_y*max_vel - vel[1]) / _ACCELERATING_MASS
    acc[0] = (force_x*max_vel - vel[0]) / _ACCELERATING_MASS

    # Limit dt to prevent chaotic jumps of players with infrequent updates
    dt = min(dt, _MAX_DT)

    # Predict new position
    new_pos = pos + vel*dt + 0.5*acc*dt**2

    # Trace path to predicted pixel position until it or an obstruction is reached
    # To preserve floating prediction in case of successful traces, they are distinguished by returning zeros
    new_pos_checked = check_player_path(player_id, pos, new_pos, height_map, player_id_map)

    # On collision, stop prematurely, otherwise, confirm velocity update
    if np.any(new_pos_checked):
        new_pos = new_pos_checked
        vel[:] = 0.

    else:
        vel += acc * dt

    step_dist = vec2_norm2(new_pos - pos)
    vel_norm = vec2_norm2(vel)

    return new_pos, vel, acc, angle, step_dist, vel_norm


@jit('float64[:](int16, float64[:], float64[:], uint8[:, :], int16[:, :])', nopython=True, nogil=True, cache=True)
def trace_shot(
    self_id: int,
    pos_0: ndarray,
    pos_1: ndarray,
    wall_map: ndarray,
    player_id_map: ndarray
) -> ndarray:
    """
    Trace shot cast by player with `self_id` from `pos_0` in direction of `pos_1`.
    Returns the point of a hit or zeros on reaching end of range.
    """

    x1, y1, dx, dy, tx, ty, sx, sy, e = init_bresenham(pos_0, pos_1)

    wall = 0
    player_id = 0

    pos_1_check = np.zeros(2)

    # Trace up to hit or end of range
    while True:
        wall = wall_map[ty, tx]
        player_id = player_id_map[ty, tx]

        if (player_id != MapID.PLAYER_ID_NULL and player_id != self_id) or wall == MapID.MASK_TERRAIN_WALL:
            pos_1_check[0] = tx
            pos_1_check[1] = ty
            break

        if tx == x1 and ty == y1:
            break

        tx, ty, e = step_bresenham(dx, dy, tx, ty, sx, sy, e)

    return pos_1_check


@jit(
    'float64[:](int16, float64[:], float64[:], uint8[:, :], uint8[:, :], int16[:, :])',
    nopython=True, nogil=True, cache=True)
def trace_sight(
    self_id: int,
    pos_0: ndarray,
    pos_1: ndarray,
    wall_map: ndarray,
    zone_map: ndarray,
    player_id_map: ndarray
) -> ndarray:
    """
    Trace line of sight of player with `self_id` from `pos_0` to `pos_1`.
    Returns zeros on success and last valid point on occlusion.
    """

    x1, y1, dx, dy, tx, ty, sx, sy, e = init_bresenham(pos_0, pos_1)

    wall = 0
    player_id = 0
    zone = 0

    pos_1_check = np.zeros(2)

    # Check for target or occlusion
    while True:
        wall = wall_map[ty, tx]
        player_id = player_id_map[ty, tx]
        zone = zone_map[ty, tx]

        if (
            (player_id != MapID.PLAYER_ID_NULL and player_id != self_id)
            or zone == MapID.ZONE_SMOKE
            or wall == MapID.MASK_TERRAIN_WALL
        ):
            pos_1_check[0] = tx
            pos_1_check[1] = ty
            break

        if tx == x1 and ty == y1:
            break

        tx, ty, e = step_bresenham(dx, dy, tx, ty, sx, sy, e)

    return pos_1_check
