import numpy as np
from numpy import ndarray
from numba import jit


F_PI = np.pi
F_2PI = 2. * F_PI
F_PI2 = F_PI / 2.
F_NPI4 = -F_PI / 4.
F_N3PI4 = -F_PI * 3. / 4.
F_SQRT2 = 2 ** 0.5
RAD_DIV_DEG = F_PI / 180.
DEG_DIV_RAD = 180. / F_PI


@jit('int16[:](int16[:, :], UniTuple(int64[:], 2))', nopython=True, nogil=True, cache=True)
def index2_by_tuple(arr2d: ndarray, indices: tuple[ndarray, ndarray]) -> ndarray:
    """Index a 2D array by given y and x-indices."""

    indices_y, indices_x = indices
    values = np.empty(len(indices_y), dtype=np.int16)

    for i in range(len(indices_y)):
        values[i] = arr2d[indices_y[i], indices_x[i]]

    return values


@jit('float64(float64[:])', nopython=True, nogil=True, cache=True)
def vec2_norm2(vec: ndarray) -> float:
    """Get the 2nd norm of a 2-element vector."""

    return (vec[0]**2 + vec[1]**2) ** 0.5


@jit('float64[:](float64[:], float64)', nopython=True, nogil=True, cache=True)
def vec2_rot_(vec: ndarray, angle: float) -> ndarray:
    """Rotate a 2-element vector by the given angle."""

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    x = cos_a*vec[0] - sin_a*vec[1]
    y = sin_a*vec[0] + cos_a*vec[1]

    vec[0] = x
    vec[1] = y

    return vec


@jit(
    'float64[:](float64[:], UniTuple(float64, 2), UniTuple(float64, 2), float64)',
    nopython=True, nogil=True, cache=True)
def vec2_lerp_(target: ndarray, a: tuple[float, float], b: tuple[float, float], x: float) -> ndarray:
    """Linear interpolation between 2-element vectors in tuple form by the given factor."""

    inv_x = 1. - x

    target[0] = a[0] * x + b[0] * inv_x
    target[1] = a[1] * x + b[1] * inv_x

    return target


@jit('float64(float64)', nopython=True, nogil=True, cache=True)
def fix_angle_range(angle: float) -> float:
    """Ensure that the angle lies between -Pi and Pi."""

    if angle > F_2PI:
        angle %= F_2PI
    elif angle < -F_2PI:
        angle %= -F_2PI

    if angle > F_PI:
        angle -= F_2PI
    elif angle < -F_PI:
        angle += F_2PI

    return angle


@jit('float64(float64, float64, float64)', nopython=True, nogil=True, cache=True)
def angle_lerp(a: float, b: float, x: float) -> float:
    """
    When crossing -pi/pi, the negative angle needs to be brought into the
    positive range (or vice versa), so that convex combination can be performed.
    """

    if a*b < 0. and abs(a - b) > F_PI:
        if a < 0.:
            a += F_2PI

        else:
            b += F_2PI

    return fix_angle_range(a * x + b * (1. - x))


@jit('float64(float64[:], float64[:])', nopython=True, nogil=True, cache=True)
def angle_from_diff(p0: ndarray, p1: ndarray) -> float:
    y, x = p0 - p1

    return np.arctan2(y, x)


@jit('float64(float64[:], float64[:], float64)', nopython=True, nogil=True, cache=True)
def rel_angle_from_diff(p0: ndarray, p1: ndarray, ref_angle: float) -> float:
    angle = angle_from_diff(p0, p1)

    return fix_angle_range(ref_angle - angle)


@jit('UniTuple(int64, 9)(float64[:], float64[:])', nopython=True, nogil=True, cache=True)
def init_bresenham(p0: ndarray, p1: ndarray) -> tuple[int, ...]:
    """Get initial parameters for a line tracer."""

    x0 = round(p0[0])
    y0 = round(p0[1])
    x1 = round(p1[0])
    y1 = round(p1[1])

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)

    e = dx + dy

    tx = x0
    ty = y0

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    return x1, y1, dx, dy, tx, ty, sx, sy, e


@jit('UniTuple(int64, 3)(int64, int64, int64, int64, int64, int64, int64)', nopython=True, nogil=True, cache=True)
def step_bresenham(dx: int, dy: int, tx: int, ty: int, sx: int, sy: int, e: int) -> tuple[int, ...]:
    """Update parameters for a line tracer."""

    e2 = 2 * e

    if e2 >= dy:
        e += dy
        tx += sx

    if e2 <= dx:
        e += dx
        ty += sy

    return tx, ty, e


@jit('uint8[:, :](int16, float64[:], float64[:])', nopython=True, nogil=True, cache=True)
def mask_line(
    length: int,
    pos_0: ndarray,
    pos_1: ndarray
) -> ndarray:
    """
    Cast ray from `pos_0`, the centre of a square with given side `length`,
    towards an endpoint `pos_1`, masking points that are traversed along the way.
    """

    x1, y1, dx, dy, tx, ty, sx, sy, e = init_bresenham(pos_0, pos_1)

    mask = np.zeros((length, length), dtype=np.uint8)

    # Mask up to endpoint
    while True:
        mask[ty, tx] = 1

        if tx == x1 and ty == y1:
            break

        tx, ty, e = step_bresenham(dx, dy, tx, ty, sx, sy, e)

    return mask


@jit('UniTuple(int64[:], 2)(float64, int64)', nopython=True, nogil=True, cache=True)
def get_line_indices(angle: float, length: int):
    """Get indices of an oriented line with given length relative to its origin."""

    indexed_area_diameter = length*2 + 1
    starting_point = np.array((length, length), dtype=np.float64)
    endpoint = starting_point + np.array((np.cos(angle), np.sin(angle))) * length

    line_mask = mask_line(indexed_area_diameter, starting_point, endpoint)
    indices_y, indices_x = np.nonzero(line_mask)

    return indices_y - length, indices_x - length


@jit('UniTuple(int64[:, :], 2)(int64)', nopython=True, nogil=True, cache=True)
def get_centred_index_grid(width: int) -> ndarray:
    """Stand-in for a use case of numpy.indices."""

    indices = np.arange(width)
    ones = np.ones(width, dtype=np.int64)
    halfwidth = width // 2

    indices_y = np.expand_dims(indices, -1) * np.expand_dims(ones, 0) - halfwidth
    indices_x = np.expand_dims(indices, 0) * np.expand_dims(ones, -1) - halfwidth

    return indices_y, indices_x


@jit('UniTuple(int64[:], 2)(int64)', nopython=True, nogil=True, cache=True)
def get_centred_indices(width: int) -> ndarray:
    indices_y, indices_x = get_centred_index_grid(width)

    return indices_y.flatten(), indices_x.flatten()


@jit('boolean[:, :](int64)', nopython=True, nogil=True, cache=True)
def get_disk_mask(width: int) -> ndarray:
    """Get a 2D boolean mask for a disk of the given pixel width."""

    centred_indices_y, centred_indices_x = get_centred_index_grid(width)
    halfwidth = width / 2.

    return (centred_indices_y**2 + centred_indices_x**2) ** 0.5 <= halfwidth


@jit('UniTuple(int64[:], 2)(int64)', nopython=True, nogil=True, cache=True)
def get_disk_indices(radius: int) -> ndarray:
    """Get indices of a disk with given width relative to its centre."""

    width = 2*radius + 1
    indices_y, indices_x = np.nonzero(get_disk_mask(width))

    return indices_y - radius, indices_x - radius
