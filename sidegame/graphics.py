"""Preparation and drawing of elements in image arrays."""

import numpy as np
import cv2
from numpy import ndarray
from numba import jit, types

from sidegame.game import MapID
from sidegame.utils_jit import step_bresenham, DEG_DIV_RAD


_Y0 = 107
_X0_LEFT = 95
_X0_RIGHT = 96
_FRAME_HEIGHT = 108
_FRAME_WIDTH = 192


@jit('uint8[:, :](uint8[:, :], uint8[:, :], float32)', nopython=True, nogil=True, cache=True)
def lerp2(a: ndarray, b: ndarray, x: float) -> ndarray:
    """Linear interpolation between two colour arrays by the given factor."""

    return (a.astype(np.float32) * x + b.astype(np.float32) * (1. - x)).astype(np.uint8)


@jit('uint8[:, :, :](uint8[:, :, :], uint8[:, :, :], float32)', nopython=True, nogil=True, cache=True)
def lerp3(a: ndarray, b: ndarray, x: float) -> ndarray:
    """Linear interpolation between two colour images by the given factor."""

    return (a.astype(np.float32) * x + b.astype(np.float32) * (1. - x)).astype(np.uint8)


@jit('boolean[:](UniTuple(int64[:], 2), UniTuple(int64, 2))', nopython=True, nogil=True, cache=True)
def get_bound_mask(indices: tuple[ndarray], bounds: tuple[int]) -> ndarray:
    """Check if (2D) indices lie within specified bounds (e.g. image dimensions)."""

    indices_y, indices_x = indices
    bound_y, bound_x = bounds

    valid_mask = (
        (indices_y >= 0) &
        (indices_y < bound_y) &
        (indices_x >= 0) &
        (indices_x < bound_x))

    return valid_mask


@jit('UniTuple(int64[:], 2)(UniTuple(int64[:], 2), UniTuple(int64, 2))', nopython=True, nogil=True, cache=True)
def enforce_bounds(indices: tuple[ndarray], bounds: tuple[int]) -> tuple[ndarray]:
    """Ensure that (2D) indices lie within specified bounds (e.g. image dimensions)."""

    valid_mask = get_bound_mask(indices, bounds)

    return indices[0][valid_mask], indices[1][valid_mask]


@jit(
    (types.Array(types.uint8, 3, 'A'), types.Array(types.uint8, 3, 'A', readonly=True), types.int64, types.int64),
    nopython=True, nogil=True, cache=True)
def draw_image(
    canvas: ndarray,
    image: ndarray,
    pos_y: int,
    pos_x: int
):
    """
    Draw an element corresponding to 4-channel image data.

    The alpha channel (3) is used to determine which parts to draw,
    while positional offsets determine the drawing location.

    Note that relative indices are expected to be relative to the top-left
    corner of the image. This should be taken into account when providing
    positional offsets to properly centre the drawing on the canvas.
    """

    bound_y = canvas.shape[0]
    bound_x = canvas.shape[1]

    for i_y in range(image.shape[0]):
        for i_x in range(image.shape[1]):
            if image[i_y, i_x, 3] > 0:
                p_y = i_y + pos_y
                p_x = i_x + pos_x

                if 0 <= p_y < bound_y and 0 <= p_x < bound_x:
                    canvas[p_y, p_x] = image[i_y, i_x, :3]


# NOTE: Passing typed dict would be expensive and not beneficial here
def draw_text(
    canvas: ndarray,
    characters: dict[str, ndarray],
    null_char: ndarray,
    text: str,
    pos_y: int,
    pos_x: int,
    spacing: int = 1
) -> int:
    """
    Draw text as a sequence of characters in rightward order.

    Returns the position of the succeeding character,
    so that text drawings can be chained.
    """

    # Correct spacing for character width
    spacing += 5

    for char in text:
        draw_image(canvas, characters.get(char, null_char), pos_y, pos_x)
        pos_x += spacing

    return pos_x


@jit(nopython=True, nogil=True, cache=True)
def draw_number(
    canvas: ndarray,
    digits: tuple[ndarray, ...],
    num: int,
    pos_y: int,
    pos_x: int,
    spacing: int = 1,
    min_n_digits: int = 1,
):
    """
    Draw a non-negative number as a sequence of digits in leftward order.

    For displaying negative numbers or chaining with text and other numbers,
    `draw_text` should be used instead, with the number provided as a string.

    If `min_n_digits` is specified, zeros are used for padding.
    """

    # Correct spacing for character width
    spacing += 3

    n_drawn_digits = 0

    while True:
        dig = num % 10
        num = num // 10

        draw_image(canvas, digits[dig], pos_y, pos_x)
        n_drawn_digits += 1

        if num == 0 and n_drawn_digits >= min_n_digits:
            break

        else:
            pos_x -= spacing


def draw_colour(
    canvas: ndarray,
    cover_indices: tuple[ndarray],
    colour: ndarray,
    opacity: float = 1.,
    pos_y: int = 0,
    pos_x: int = 0,
    bounds: tuple[int] = None,
    background: ndarray = None
):
    """Draw a cover of colour over a part of the canvas."""

    if pos_y or pos_x:
        cover_indices = cover_indices[0] + pos_y, cover_indices[1] + pos_x

    if bounds is not None:
        cover_indices = enforce_bounds(cover_indices, bounds)

    if background is None:
        background = canvas

    if opacity != 1.:
        canvas[cover_indices] = lerp2(colour[None], background[cover_indices], opacity)

    else:
        canvas[cover_indices] = colour


def draw_overlay(canvas: ndarray, overlay: ndarray, opacity: float = 1.) -> ndarray:
    """Draw a full-sized overlay over the canvas."""

    if opacity == 1.:
        if overlay.shape[-1] == canvas.shape[-1]:
            return overlay

        else:
            return np.tile(overlay, (1, 1, canvas.shape[-1]))

    return lerp3(overlay, canvas, opacity)


def draw_muted(canvas: ndarray, opacity: float = 0.5) -> ndarray:
    """Convert the canvas into full or partial grayscale for a muted image effect."""

    overlay = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)[..., None]

    return draw_overlay(canvas, overlay, opacity=opacity)


def get_camera_warp(pos: tuple[float], angle: float, viewpoint: tuple[float], scale: float = 1.) -> ndarray:
    """
    Get the matrix corresponding to the transformation of world coordinates
    into a local system.

    Coordinates are rotated by the camera angle around its position
    and translated by it, as well as by an offset to move the centre of rotation
    into its expected position as a viewpoint on the canvas.
    """

    warp = cv2.getRotationMatrix2D(pos, angle*DEG_DIV_RAD, scale)

    pos_x, pos_y = pos
    viewpoint_x, viewpoint_y = viewpoint

    warp[0, 2] += viewpoint_x - pos_x
    warp[1, 2] += viewpoint_y - pos_y

    return warp


def get_inverse_warp(pos: tuple[float], angle: float, viewpoint: tuple[float], scale: float = 1.) -> ndarray:
    """
    Get the matrix corresponding to the transformation of a local system
    into world coordinates.
    """

    warp = cv2.getRotationMatrix2D(viewpoint, -angle*DEG_DIV_RAD, scale)

    pos_x, pos_y = pos
    viewpoint_x, viewpoint_y = viewpoint

    warp[0, 2] -= viewpoint_x - pos_x
    warp[1, 2] -= viewpoint_y - pos_y

    return warp


def project_into_view(
    world_image: ndarray,
    camera_pos: tuple[float],
    camera_angle: float,
    camera_viewpoint: tuple[float],
    frame_size: tuple[int],
    scale: float = 1.,
    preserve_values: bool = False
) -> ndarray:
    """
    Get the matrix corresponding to the transformation of world coordinates
    into a local system and use it to get the view within a frame.

    If the world image is a code map, i.e. comprised of discrete values with
    specific meaning, nearest-neighbour interpolation can be used to preserve
    them, although this might cause some issues on borders of the segmentation.

    Optionally, the projection is also scalable in resolution, which can be used
    for e.g. anti-aliasing.
    """

    camera_warp = get_camera_warp(camera_pos, camera_angle, camera_viewpoint, scale=scale)
    interp_mode = cv2.INTER_NEAREST if preserve_values else cv2.INTER_LINEAR

    return cv2.warpAffine(world_image, camera_warp, frame_size, flags=interp_mode)


@jit('float64[:](float64[:, :], float64[:])', nopython=True, nogil=True, cache=True)
def warp_position(warp: ndarray, pos: ndarray) -> ndarray:
    x, y = pos

    pos_x = warp[0, 0] * x + warp[0, 1] * y + warp[0, 2]
    pos_y = warp[1, 0] * x + warp[1, 1] * y + warp[1, 2]

    return np.array((pos_x, pos_y))


@jit('UniTuple(int64, 2)(float64[:, :], int64, int64, int64, int64)', nopython=True, nogil=True, cache=True)
def warp_indices(warp: ndarray, x: int, y: int, dimx: int, dimy: int) -> tuple[int, int]:
    x = float(x)
    y = float(y)

    return (
        max(0, min(dimx, round(warp[0, 0] * x + warp[0, 1] * y + warp[0, 2]))),
        max(0, min(dimy, round(warp[1, 0] * x + warp[1, 1] * y + warp[1, 2]))))


@jit(
    'uint8[:, :](int16, int64, int64, int64, int64, uint8[:, :], int16[:, :], uint8[:, :], '
    'uint8[:, :], uint8[:, :, :], uint8[:, :, :], float64[:, :], uint8[:, :])',
    nopython=True, nogil=True, cache=True)
def mask_visible_line(
    self_id: int,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    wall_map: ndarray,
    player_id_map: ndarray,
    zone_map: ndarray,
    fx_ctr_map: ndarray,
    fx_ref: ndarray,
    world: ndarray,
    warp: ndarray,
    mask: ndarray
) -> ndarray:

    wall = 0
    player_id = 0
    vis_level = MapID.VIS_LEVEL_FULL
    dimy, dimx = wall_map.shape

    # Init for Bresenham
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    e = dx + dy
    ty = y0
    tx = x0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    # Mask up to endpoint or occlusion
    while True:
        wx, wy = warp_indices(warp, tx, ty, dimx, dimy)
        wall = wall_map[wy, wx]

        if wall == MapID.MASK_TERRAIN_WALL:
            if fx_ctr_map[wy, wx] > 0:
                world[ty, tx] = fx_ref[wy, wx]

            break

        player_id = player_id_map[wy, wx]

        if (zone_map[wy, wx] == MapID.ZONE_SMOKE) and (vis_level > MapID.VIS_LEVEL_SMOKE):
            vis_level = MapID.VIS_LEVEL_SMOKE

        elif player_id != MapID.PLAYER_ID_NULL and player_id != self_id and vis_level > MapID.VIS_LEVEL_SHADOW:
            vis_level = MapID.VIS_LEVEL_SHADOW

        mask[ty, tx] = vis_level

        if vis_level > 0 and fx_ctr_map[wy, wx] > 0:
            world[ty, tx] = fx_ref[wy, wx]

        if tx == x1 and ty == y1:
            break

        tx, ty, e = step_bresenham(dx, dy, tx, ty, sx, sy, e)

    return mask


@jit(
    'float32[:, :](int16, uint8[:, :, :], int16[:, :, :], '
    'uint8[:, :, :], uint8[:, :, :], float64[:, :], '
    'UniTuple(int64[:], 4))',
    nopython=True, nogil=True, cache=True)
def mask_view(
    self_id: int,
    code_map: ndarray,
    id_map: ndarray,
    fx_ref: ndarray,
    world: ndarray,
    warp: ndarray,
    endpoints: tuple[ndarray]
) -> ndarray:
    """
    Mask pixels that are (un)obstructed by terrain, other entities, or fog,
    within limited viewable area specified by given endpoints.

    Masking is performed by casting rays from a preset starting point
    towards all given endpoints, masking unoccluded points.
    This is done separately for left and right parts of the view.

    `observer_id` is explicitly provided to prevent cases where the
    observing entity would itself block the rays from progressing outwards.
    """

    wall_map = code_map[np.int64(MapID.CHANNEL_WALL)]

    if self_id == MapID.PLAYER_ID_NULL:
        zone_map = code_map[np.int64(MapID.CHANNEL_ZONE_NULL)]
        fx_ctr_map = code_map[np.int64(MapID.CHANNEL_ZONE_NULL)]
        player_id_map = id_map[np.int64(MapID.CHANNEL_PLAYER_ID_NULL)]

    else:
        zone_map = code_map[np.int64(MapID.CHANNEL_ZONE)]
        fx_ctr_map = code_map[np.int64(MapID.CHANNEL_FX)]
        player_id_map = id_map[np.int64(MapID.CHANNEL_PLAYER_ID)]

    left_ends_y, left_ends_x, right_ends_y, right_ends_x = endpoints

    mask = np.zeros((_FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.uint8)

    for idx in range(left_ends_y.shape[0]):
        x1 = left_ends_x[idx]
        y1 = left_ends_y[idx]

        mask = mask_visible_line(
            self_id, _Y0, _X0_LEFT, y1, x1, wall_map, player_id_map, zone_map, fx_ctr_map, fx_ref, world, warp, mask)

    for idx in range(right_ends_y.shape[0]):
        x1 = right_ends_x[idx]
        y1 = right_ends_y[idx]

        mask = mask_visible_line(
            self_id, _Y0, _X0_RIGHT, y1, x1, wall_map, player_id_map, zone_map, fx_ctr_map, fx_ref, world, warp, mask)

    vis_level_full = np.float32(np.int64(MapID.VIS_LEVEL_FULL))
    mask = (mask.astype(np.float32) + vis_level_full) / np.float32(2. * vis_level_full)

    return mask
