"""Preparation and drawing of elements in image arrays."""

import os
from typing import Tuple

import numpy as np
import cv2
from numba import jit, types
from numba.typed import Dict

from sidegame.ext import sdglib


_CHAR_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)),  'assets', 'characters')
_ENDPOINT_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'views', 'endpoints.png')

CHARACTERS: dict[str, np.ndarray] = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.Array(types.uint8, 3, 'A', readonly=True))

for char in os.listdir(_CHAR_DIR):
    CHARACTERS[char.split('_')[1][:-4]] = cv2.imread(os.path.join(_CHAR_DIR, char), cv2.IMREAD_UNCHANGED)

NULL_CHARACTER: np.ndarray = CHARACTERS['null']
DIGITS: tuple[np.ndarray] = tuple(CHARACTERS[str(num)] for num in range(10))


@jit('uint8[:, :](uint8[:, :], uint8[:, :], float32)', nopython=True, nogil=True)
def lerp2(a: np.ndarray, b: np.ndarray, x: float) -> np.ndarray:
    """Linear interpolation between two colour arrays by the given factor."""

    return (a.astype(np.float32) * x + b.astype(np.float32) * (1. - x)).astype(np.uint8)


@jit('uint8[:, :, :](uint8[:, :, :], uint8[:, :, :], float32)', nopython=True, nogil=True)
def lerp3(a: np.ndarray, b: np.ndarray, x: float) -> np.ndarray:
    """Linear interpolation between two colour images by the given factor."""

    return (a.astype(np.float32) * x + b.astype(np.float32) * (1. - x)).astype(np.uint8)


@jit('boolean[:](UniTuple(int64[:], 2), UniTuple(int64, 2))', nopython=True, nogil=True)
def get_bound_mask(indices: Tuple[np.ndarray], bounds: Tuple[int]) -> np.ndarray:
    """Check if (2D) indices lie within specified bounds (e.g. image dimensions)."""

    indices_y, indices_x = indices
    bound_y, bound_x = bounds

    valid_mask = (
        (indices_y >= 0) &
        (indices_y < bound_y) &
        (indices_x >= 0) &
        (indices_x < bound_x))

    return valid_mask


@jit('UniTuple(int64[:], 2)(UniTuple(int64[:], 2), UniTuple(int64, 2))', nopython=True, nogil=True)
def enforce_bounds(indices: Tuple[np.ndarray], bounds: Tuple[int]) -> Tuple[np.ndarray]:
    """Ensure that (2D) indices lie within specified bounds (e.g. image dimensions)."""

    valid_mask = get_bound_mask(indices, bounds)

    return indices[0][valid_mask], indices[1][valid_mask]


@jit(
    (types.Array(types.uint8, 3, 'A'), types.Array(types.uint8, 3, 'A', readonly=True), types.int64, types.int64),
    nopython=True, nogil=True)
def draw_image(
    canvas: np.ndarray,
    image: np.ndarray,
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


def draw_text(
    canvas: np.ndarray,
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
        draw_image(canvas, CHARACTERS.get(char, NULL_CHARACTER), pos_y, pos_x)
        pos_x += spacing

    return pos_x


@jit(nopython=True, nogil=True)
def draw_number(
    canvas: np.ndarray,
    num: int,
    pos_y: int,
    pos_x: int,
    spacing: int = 1,
    min_n_digits: int = 1
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

        draw_image(canvas, DIGITS[dig], pos_y, pos_x)
        n_drawn_digits += 1

        if num == 0 and n_drawn_digits >= min_n_digits:
            break

        else:
            pos_x -= spacing


def draw_colour(
    canvas: np.ndarray,
    cover_indices: Tuple[np.ndarray],
    colour: np.ndarray,
    opacity: float = 1.,
    pos_y: int = 0,
    pos_x: int = 0,
    bounds: Tuple[int] = None,
    background: np.ndarray = None
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


def draw_overlay(canvas: np.ndarray, overlay: np.ndarray, opacity: float = 1.) -> np.ndarray:
    """Draw a full-sized overlay over the canvas."""

    if opacity == 1.:
        if overlay.shape[-1] == canvas.shape[-1]:
            return overlay
        else:
            return np.tile(overlay, (1, 1, canvas.shape[-1]))

    return lerp3(overlay, canvas, opacity)


def draw_muted(canvas: np.ndarray, opacity: float = 0.5) -> np.ndarray:
    """Convert the canvas into full or partial grayscale for a muted image effect."""

    overlay = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)[..., None]

    return draw_overlay(canvas, overlay, opacity=opacity)


def get_camera_warp(pos: Tuple[float], angle: float, viewpoint: Tuple[float], scale: float = 1.) -> np.ndarray:
    """
    Get the matrix corresponding to the transformation of world coordinates
    into a local system.

    Coordinates are rotated by the camera angle around its position
    and translated by it, as well as by an offset to move the centre of rotation
    into its expected position as a viewpoint on the canvas.
    """

    warp = cv2.getRotationMatrix2D(pos, angle*180./np.pi, scale)

    pos_x, pos_y = pos
    viewpoint_x, viewpoint_y = viewpoint

    warp[0, 2] += viewpoint_x - pos_x
    warp[1, 2] += viewpoint_y - pos_y

    return warp


def get_inverse_warp(pos: Tuple[float], angle: float, viewpoint: Tuple[float], scale: float = 1.) -> np.ndarray:
    """
    Get the matrix corresponding to the transformation of a local system
    into world coordinates.
    """

    warp = cv2.getRotationMatrix2D(viewpoint, -angle*180./np.pi, scale)

    pos_x, pos_y = pos
    viewpoint_x, viewpoint_y = viewpoint

    warp[0, 2] -= viewpoint_x - pos_x
    warp[1, 2] -= viewpoint_y - pos_y

    return warp


def project_into_view(
    world_image: np.ndarray,
    camera_pos: Tuple[float],
    camera_angle: float,
    camera_viewpoint: Tuple[float],
    frame_size: Tuple[int],
    scale: float = 1.,
    preserve_values: bool = False
) -> np.ndarray:
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


def get_view_endpoints(fov_deg: float, radius: float) -> Tuple[np.ndarray]:
    """
    Given a field of view and limited radius, get the endpoints for rays that
    define the viewable area.
    """

    assert fov_deg <= 180., f'Max. field of view range exceeded: {fov_deg:.2f}'

    # Get endpoint image source
    endpoint_image = cv2.imread(_ENDPOINT_PATH, cv2.IMREAD_GRAYSCALE)

    # Convert FOV to radians
    fov_rad = fov_deg * np.pi / 180.

    # Mask away the endpoints below the bottom threshold determined by FOV
    threshold_y = int(np.ceil(radius * (1. - np.cos(fov_rad/2.))))
    endpoint_image[threshold_y:] = 0

    # Split the image into left and right halves
    left_half_image, right_half_image = np.hsplit(endpoint_image, 2)

    # Get endpoints (indices)
    left_half_idy, left_half_idx = np.nonzero(left_half_image)
    right_half_idy, right_half_idx = np.nonzero(right_half_image)

    # Correct right half indices after splitting
    right_half_idx += endpoint_image.shape[1] // 2

    return left_half_idy, left_half_idx, right_half_idy, right_half_idx


def render_view(
    world_view: np.ndarray,
    height_map: np.ndarray,
    entity_map: np.ndarray,
    zone_map: np.ndarray,
    fx_map: np.ndarray,
    fx_ref: np.ndarray,
    warp: np.ndarray,
    endpoints: Tuple[np.ndarray],
    observer_id: int = 0
) -> np.ndarray:
    """
    Mask pixels that are (un)obstructed by terrain, other entities, or fog,
    within limited viewable area specified by given endpoints.

    Masking is performed by ray tracing from a hard-coded position of origin
    to each endpoint and checking terminal conditions along the way.

    `observer_id` is explicitly provided to prevent cases where the
    observing entity would itself block the rays from progressing outwards.
    """

    return sdglib.mask_view(observer_id, height_map, entity_map, zone_map, fx_map, fx_ref, world_view, warp, *endpoints)
