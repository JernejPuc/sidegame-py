"""Definitions of entities that are realised in the game world."""

from typing import Tuple, Union
import numpy as np
from sidegame.ext import sdglib


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

    F_PI = np.pi
    F_2PI = 2. * np.pi
    F_PI2 = np.pi / 2.
    F_NPI4 = -np.pi / 4.
    F_N3PI4 = -np.pi * 3. / 4.
    F_SQRT2 = np.sqrt(2.)

    def __init__(self):
        self.pos = np.array([0., 0.])
        self.angle = 0.

    def fix_angle_range(self, angle: float) -> float:
        """Ensure that the angle lies between -Pi and Pi."""

        if angle > self.F_2PI:
            angle %= self.F_2PI
        elif angle < -self.F_2PI:
            angle %= -self.F_2PI

        if angle > self.F_PI:
            angle -= self.F_2PI
        elif angle < -self.F_PI:
            angle += self.F_2PI

        return angle


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
        self.covered_indices = np.indices((width, width)) - self.halfwidth

    def get_position_indices(self, pos: np.ndarray = None) -> Tuple[int]:
        """
        Get the rounded positions corresponding to indices on the vertical
        and horizontal axes (numpy order).

        Entity position can be explicitly specified to cover cases where
        multiple, e.g. older, positions are stored and being dealt with.
        """

        if pos is None:
            pos = self.pos

        return round(pos[1]), round(pos[0])

    def get_covered_indices(self, pos: np.ndarray = None) -> Union[Tuple[int], Tuple[np.ndarray]]:
        """
        Get the indices on the vertical and horizontal axes that are
        approximately covered by the entity.

        Entity position can be explicitly specified to cover cases where
        multiple, e.g. older, positions are stored and being dealt with.
        """

        if pos is None:
            pos = self.pos

        pos_y, pos_x = self.get_position_indices(pos)

        if not self.halfwidth:
            return pos_y, pos_x

        return self.covered_indices[0] + pos_y, self.covered_indices[1] + pos_x

    def update_collider_map(
        self,
        collider_map: np.ndarray,
        old_pos: np.ndarray,
        new_pos: np.ndarray,
        claim_id: int = 1,
        clear_id: int = 0,
        check_claimed_area: bool = False,
        check_cleared_area: bool = False
    ):
        """
        Update collider map when moving or rewinding
        (basically, move the hitbox).

        Optionally, restrict interaction to indices that belong to claimed
        (or clear) ID.
        """

        # Update entity map if covered area has changed
        old_pos_y, old_pos_x = self.get_position_indices(old_pos)
        new_pos_y, new_pos_x = self.get_position_indices(new_pos)

        if old_pos_y != new_pos_y or old_pos_x != new_pos_x or collider_map[old_pos_y, old_pos_x] == clear_id:
            old_covered_indices = self.get_covered_indices(old_pos)
            new_covered_indices = self.get_covered_indices(new_pos)

            if check_cleared_area:
                valid_mask = np.where(collider_map[old_covered_indices] == claim_id)
                old_covered_indices = old_covered_indices[0][valid_mask], old_covered_indices[1][valid_mask]

            if check_claimed_area:
                valid_mask = np.where(collider_map[new_covered_indices] == clear_id)
                new_covered_indices = new_covered_indices[0][valid_mask], new_covered_indices[1][valid_mask]

            # Clear currently covered area
            collider_map[old_covered_indices] = clear_id

            # Claim newly covered area
            collider_map[new_covered_indices] = claim_id


class ThrowableEntity(ColliderEntity):
    """A pixel-wide object entity that can be thrown and bounce off of walls."""

    COLLISION_NONE = 0
    COLLISION_BOUNCE = 1
    COLLISION_LANDING = 2

    _BOUNCE_VEL_PRESERVATION = np.sqrt(2.)/2.
    _VERTICAL_KERNEL = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.int16)
    _HORIZONTAL_KERNEL = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.int16)

    def __init__(self, object_id: int = 0):
        super().__init__(1)

        self.id = object_id
        self.pos_target: np.ndarray = None
        self.vel = np.array([0., 0.])

    def throw(
        self,
        pos_thrower: np.ndarray,
        pos_target: np.ndarray,
        init_throw_vel: float,
        init_vel: np.ndarray = None
    ):
        """
        Set initial velocity and target position of the thrown object.

        If initial velocity is provided, e.g. to simulate momentum,
        it is added to throw-induced velocity and the target position
        is moved by 1-second travel distance.
        """

        pos_x, pos_y = pos_thrower
        pos_x_target, pos_y_target = pos_target

        throw_angle = np.arctan2(pos_y_target - pos_y, pos_x_target - pos_x)
        throw_dir = np.array([np.cos(throw_angle), np.sin(throw_angle)])

        self.pos = pos_thrower
        self.pos_target = pos_target
        self.vel = throw_dir * init_throw_vel

        if init_vel is not None:
            self.pos_target += init_vel * 1.
            self.vel += init_vel

    def move(self, dt: float, wall_map: np.ndarray, object_map: np.ndarray) -> int:
        """
        Move the object forward in time by time step `dt`.

        The update can return different status flags, which can be relayed
        and used to trigger external events and effects.
        """

        # Return if no movement issued
        if self.pos_target is None:
            return self.COLLISION_NONE

        # Predict new position
        new_pos = self.pos + self.vel * dt

        # End movement on overshoot or if velocity indicates too many bounces
        if (
            np.linalg.norm(new_pos - self.pos_target) >= np.linalg.norm(self.pos - self.pos_target) or
            np.linalg.norm(self.vel) < 1.
        ):
            self.pos_target = None
            self.vel.fill(0.)
            return self.COLLISION_LANDING

        # Check for and handle collision
        new_pos_check = sdglib.move_object(self.id, self.pos, new_pos, wall_map, object_map)

        if any(new_pos_check):
            self.bounce(new_pos_check, wall_map)
            self.pos = new_pos_check
            return self.COLLISION_BOUNCE
        else:
            self.pos = new_pos
            return self.COLLISION_NONE

    def bounce(self, pos_hit: np.ndarray, wall_map: np.ndarray):
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
        pos_x_target, pos_y_target = self.pos_target

        # Get correlations
        hit_wall_nbhood = wall_map[round(pos_y_hit)-1:round(pos_y_hit)+2, round(pos_x_hit)-1:round(pos_x_hit)+2]

        corr_y = np.sum(self._VERTICAL_KERNEL * hit_wall_nbhood)
        corr_x = np.sum(self._HORIZONTAL_KERNEL * hit_wall_nbhood)

        # Infer angle of wall's incoming normal
        wall_angle = np.arctan2(corr_y, corr_x)

        # Modify wall angle to conform to legacy conditions
        # (edge case handling seemed more intuitive under a different definition)
        wall_angle -= self.F_PI2

        # Get difference relative to wall angle
        incoming_angle = np.arctan2(pos_y_target - pos_y_hit, pos_x_target - pos_x_hit)
        angle_diff = self.fix_angle_range(incoming_angle - wall_angle)

        # Handle edge cases
        if 0. < angle_diff < self.F_PI:
            bounce_angle = 2.*wall_angle - incoming_angle

        elif self.F_NPI4 < angle_diff <= 0.:
            bounce_angle = 2.*wall_angle - incoming_angle - self.F_PI2

        elif (angle_diff == self.F_PI) or (-self.F_PI <= angle_diff < self.F_N3PI4):
            bounce_angle = 2.*wall_angle - incoming_angle + self.F_PI2

        else:
            bounce_angle = 2.*wall_angle - incoming_angle + self.F_PI

        bounce_angle = self.fix_angle_range(bounce_angle)

        # Get new direction
        bounce_dir = np.array([np.cos(bounce_angle), np.sin(bounce_angle)])

        # Set new target
        path_remaining = np.linalg.norm(self.pos_target - pos_hit)
        self.pos_target = pos_hit + bounce_dir * path_remaining

        # Set new velocity
        self.vel = bounce_dir * np.linalg.norm(self.vel) * self._BOUNCE_VEL_PRESERVATION


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

    MAX_VIEW_RANGE = 106.5
    MOUSE_MVMT_TO_ANGLE = np.arctan(1./MAX_VIEW_RANGE)

    _WALK_RESET_DISTANCE = 12.25
    _STILLNESS_THRESHOLD = 1.

    _ACCELERATING_MASS = 1./5.5
    _MAX_DT = 0.75*_ACCELERATING_MASS

    def __init__(self, player_id: int = 32727, rng: np.random.Generator = None):
        super().__init__(3)

        self.id = player_id
        self.rng = np.random.default_rng() if rng is None else rng

        self.vel = np.array([0., 0.])
        self.acc = np.array([0., 0.])

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
        height_map: np.ndarray,
        player_map: np.ndarray
    ) -> int:
        """
        Move the player forward in time by time step `dt`.

        The update can return a boolean flag for a footstep event, which can be
        relayed and used to trigger external events and effects.

        NOTE: After moving, the entity map should be updated externally.
        """

        # Convert mouse movement to angle difference and add it to current angle
        self.angle = self.fix_angle_range(self.angle + (mouse_hor * self.MOUSE_MVMT_TO_ANGLE))

        # Enforce consistent magnitude of accelerating force
        if force_w and force_d:
            force_w /= self.F_SQRT2
            force_d /= self.F_SQRT2

        # Get accelerating force in the global system
        force_y = np.sin(self.angle)*force_w + np.cos(self.angle)*force_d
        force_x = np.cos(self.angle)*force_w - np.sin(self.angle)*force_d

        # Self-regulate accelerating force and get acceleration
        # Accelerating force is formulated as a function of maximum velocity,
        # while force of friction is formulated as a function of current velocity
        self.acc[1] = (force_y*max_vel - self.vel[1]) / self._ACCELERATING_MASS
        self.acc[0] = (force_x*max_vel - self.vel[0]) / self._ACCELERATING_MASS

        # Limit dt to prevent chaotic jumps of players with infrequent updates
        dt = min(dt, self._MAX_DT)

        # Predict new position
        new_pos = self.pos + self.vel*dt + 0.5*self.acc*dt**2

        # Trace path to predicted pixel position until it or an obstruction is reached
        # To preserve floating prediction in case of successful traces, they are distinguished by returning zeros
        new_pos_check = sdglib.move_player(self.id, self.pos, new_pos, height_map, player_map)

        # On collision, stop prematurely, otherwise, confirm velocity update
        if any(new_pos_check):
            self.vel.fill(0.)

            # NOTE: Need to copy the array, because the data is in Rust heap,
            # though it should still be managed (and released) by Python's GC
            # See: https://docs.rs/numpy/0.13.1/numpy/array/struct.PyArray.html#memory-location
            new_pos = new_pos_check.copy()

        else:
            self.vel += self.acc*dt

        # Progress the movement state
        emit_footstep = self.advance_step(np.linalg.norm(new_pos - self.pos), np.linalg.norm(self.vel), walking)

        # Update position
        self.pos = new_pos

        return emit_footstep

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

        emit_footstep = False

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
                emit_footstep = True

        return emit_footstep

    def get_focal_point(
        self, height_map: np.ndarray, entity_map: np.ndarray, zone_map: np.ndarray, max_range: float = 1e+3
    ) -> np.ndarray:
        """
        Get the farthest unobstructed point directly in front of the player.

        NOTE: Returns zeros if max range was reached. Any handling of this
        should be done externally.
        """

        endpoint = self.pos + np.array([np.cos(self.angle), np.sin(self.angle)]) * max_range

        return sdglib.trace_sight(self.id, self.pos, endpoint, height_map, entity_map, zone_map)
