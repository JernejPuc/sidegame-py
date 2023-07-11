import numpy as np
from numpy import ndarray
from numba import jit

from sidegame.game.shared.core import Map


MASK_TERRAIN_WALL = 1
VIS_LEVEL_FULL = 4
VIS_LEVEL_SHADOW = 2
VIS_LEVEL_SMOKE = 1


@jit('float64[:](int16, float64[:], float64[:], uint8[:, :], int16[:, :])', nopython=True, nogil=True)
def move_player(
    self_id: int,
    pos_1: ndarray,
    pos_2: ndarray,
    height_map: ndarray,
    player_id_map: ndarray
) -> ndarray:

    terrain = 0
    player_id = 0

    in_transition = False
    collision_detected = False

    # Round to pixel positions
    x0 = round(pos_1[0])
    y0 = round(pos_1[1])
    x1 = round(pos_2[0])
    y1 = round(pos_2[1])

    # Init out
    pos_2_check = np.zeros(2)

    # Init for Bresenham
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)

    e = dx + dy
    e2 = 0

    ty = ty_prev = y0
    tx = tx_prev = x0

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    # Check 9 core pixels (3x3) for all possibilities
    for i in range(-1, 2):
        for j in range(-1, 2):
            terrain = height_map[ty + i, tx + j]

            if terrain >= Map.HEIGHT_TRANSITION:
                in_transition = True

    # Trace up to target or collision
    while True:
        for i in range(-1, 2):
            for j in range(-1, 2):
                terrain = height_map[ty + i, tx + j]
                player_id = player_id_map[ty + i, tx + j]

                if (
                    ((player_id != Map.PLAYER_ID_NULL) and (player_id != self_id))
                    or (terrain > Map.HEIGHT_ELEVATED)
                    or ((terrain == Map.HEIGHT_ELEVATED) and (not in_transition))
                ):
                    collision_detected = True

        if collision_detected:
            pos_2_check[0] = tx_prev
            pos_2_check[1] = ty_prev
            break

        else:
            tx_prev = tx
            ty_prev = ty

        # Bresenham
        if (tx == x1) and (ty == y1):
            break

        e2 = 2 * e

        if e2 >= dy:
            e += dy
            tx += sx

        if e2 <= dx:
            e += dx
            ty += sy

    return pos_2_check


@jit('float64[:](int16, float64[:], float64[:], uint8[:, :], int16[:, :])', nopython=True, nogil=True)
def move_object(
    self_id: int,
    pos_1: ndarray,
    pos_2: ndarray,
    wall_map: ndarray,
    object_map: ndarray
) -> ndarray:

    terrain = 0
    object_ = 0

    collision_detected = False

    # Round to pixel positions
    x0 = round(pos_1[0])
    y0 = round(pos_1[1])
    x1 = round(pos_2[0])
    y1 = round(pos_2[1])

    # Init out
    pos_2_check = np.zeros(2)

    # Init for Bresenham
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)

    e = dx + dy
    e2 = 0

    ty = ty_prev = y0
    tx = tx_prev = x0

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    # Trace up to target or collision
    while True:
        terrain = wall_map[ty, tx]
        object_ = object_map[ty, tx]

        if (terrain == MASK_TERRAIN_WALL) or ((object_ != Map.OBJECT_ID_NULL) and (object_ != self_id)):
            collision_detected = True

        if collision_detected:
            pos_2_check[0] = tx_prev
            pos_2_check[1] = ty_prev
            break

        else:
            tx_prev = tx
            ty_prev = ty

        # Bresenham
        if (tx == x1) and (ty == y1):
            break

        e2 = 2 * e

        if e2 >= dy:
            e += dy
            tx += sx

        if e2 <= dx:
            e += dx
            ty += sy

    return pos_2_check


@jit('float64[:](int16, float64[:], float64[:], uint8[:, :], int16[:, :])', nopython=True, nogil=True)
def trace_shot(
    self_id: int,
    pos_1: ndarray,
    pos_2: ndarray,
    height_map: ndarray,
    player_id_map: ndarray
) -> ndarray:

    terrain = 0
    player_id = 0

    # Round to pixel positions
    x0 = round(pos_1[0])
    y0 = round(pos_1[1])
    x1 = round(pos_2[0])
    y1 = round(pos_2[1])

    # Init out
    pos_2_check = np.zeros(2)

    # Init for bresenham
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)

    e = dx + dy
    e2 = 0

    ty = y0
    tx = x0

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    # Trace up to hit or end of range
    while True:
        terrain = height_map[ty, tx]
        player_id = player_id_map[ty, tx]

        if ((player_id != Map.PLAYER_ID_NULL) and (player_id != self_id)) or (terrain > Map.HEIGHT_ELEVATED):
            pos_2_check[0] = tx
            pos_2_check[1] = ty
            break

        # Bresenham
        if (tx == x1) and (ty == y1):
            break

        e2 = 2 * e

        if e2 >= dy:
            e += dy
            tx += sx

        if e2 <= dx:
            e += dx
            ty += sy

    return pos_2_check


@jit('float64[:](int16, float64[:], float64[:], uint8[:, :], int16[:, :], uint8[:, :])', nopython=True, nogil=True)
def trace_sight(
    self_id: int,
    pos_1: ndarray,
    pos_2: ndarray,
    height_map: ndarray,
    player_id_map: ndarray,
    zone_map: ndarray
) -> ndarray:

    terrain = 0
    player_id = 0
    zone = 0

    # Round to pixel positions
    x0 = round(pos_1[0])
    y0 = round(pos_1[1])
    x1 = round(pos_2[0])
    y1 = round(pos_2[1])

    # Init out
    pos_2_check = np.zeros(2)

    # Init for bresenham
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)

    e = dx + dy
    e2 = 0

    ty = y0
    tx = x0

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    # Trace up to target or occlusion
    while True:
        terrain = height_map[ty, tx]
        player_id = player_id_map[ty, tx]
        zone = zone_map[ty, tx]

        if (
            ((player_id != Map.PLAYER_ID_NULL) and (player_id != self_id))
            or (zone == Map.ZONE_SMOKE)
            or (terrain > Map.HEIGHT_ELEVATED)
        ):
            pos_2_check[0] = tx
            pos_2_check[1] = ty
            break

        # Bresenham
        if (tx == x1) and (ty == y1):
            break

        e2 = 2 * e

        if e2 >= dy:
            e += dy
            tx += sx

        if e2 <= dx:
            e += dx
            ty += sy

    return pos_2_check


@jit(
    'uint8[:, :](int16, int64, int64, int64, int64, uint8[:, :], int16[:, :], uint8[:, :], uint8[:, :])',
    nopython=True,
    nogil=True)
def mask_visible_line(
    self_id: int,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    height_map: ndarray,
    player_map: ndarray,
    zone_map: ndarray,
    mask: ndarray
) -> ndarray:

    terrain = 0
    player_id = 0
    zone = 0

    visibility_level = VIS_LEVEL_FULL

    # Init for Bresenham
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)

    e = dx + dy
    e2 = 0

    ty = y0
    tx = x0

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    # Mask up to endpoint or occlusion
    while True:
        terrain = height_map[ty, tx]
        player_id = player_map[ty, tx]
        zone = zone_map[ty, tx]

        if terrain > Map.HEIGHT_ELEVATED:
            break

        elif (zone == Map.ZONE_SMOKE) and (visibility_level > VIS_LEVEL_SMOKE):
            visibility_level = VIS_LEVEL_SMOKE

        elif (player_id != Map.PLAYER_ID_NULL) and (player_id != self_id) and (visibility_level > VIS_LEVEL_SHADOW):
            visibility_level = VIS_LEVEL_SHADOW

        mask[ty, tx] = visibility_level

        # Bresenham
        if (tx == x1) and (ty == y1):
            break

        e2 = 2 * e

        if e2 >= dy:
            e += dy
            tx += sx

        if e2 <= dx:
            e += dx
            ty += sy

    return mask


@jit(
    'uint8[:, :](int16, uint8[:, :], int16[:, :], uint8[:, :], int64[:], int64[:], int64[:], int64[:])',
    nopython=True,
    nogil=True)
def mask_view(
    self_id: int,
    height_map: ndarray,
    player_map: ndarray,
    zone_map: ndarray,
    left_ends_y: ndarray,
    left_ends_x: ndarray,
    right_ends_y: ndarray,
    right_ends_x: ndarray
) -> ndarray:

    y0 = 107
    x0_left = 95
    x0_right = 96
    x1 = 0
    y1 = 0

    mask = np.zeros((108, 192), dtype=np.uint8)

    for idx in range(left_ends_y.shape[0]):
        x1 = left_ends_x[idx]
        y1 = left_ends_y[idx]

        mask = mask_visible_line(self_id, y0, x0_left, y1, x1, height_map, player_map, zone_map, mask)

    for idx in range(right_ends_y.shape[0]):
        x1 = right_ends_x[idx]
        y1 = right_ends_y[idx]

        mask = mask_visible_line(self_id, y0, x0_right, y1, x1, height_map, player_map, zone_map, mask)

    return mask


@jit('uint8[:, :](int16, float64[:], float64[:])', nopython=True, nogil=True)
def mask_ray(
    length: int,
    pos_1: ndarray,
    pos_2: ndarray
) -> ndarray:

    # Round to pixel positions
    x0 = round(pos_1[0])
    y0 = round(pos_1[1])
    x1 = round(pos_2[0])
    y1 = round(pos_2[1])

    # Init out
    mask = np.zeros((length, length), dtype=np.uint8)

    # Init for bresenham
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)

    e = dx + dy
    e2 = 0

    ty = y0
    tx = x0

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    # Mask up to endpoint
    while True:
        mask[ty, tx] = 1

        # Bresenham
        if (tx == x1) and (ty == y1):
            break

        e2 = 2 * e

        if e2 >= dy:
            e += dy
            tx += sx

        if e2 <= dx:
            e += dx
            ty += sy

    return mask
