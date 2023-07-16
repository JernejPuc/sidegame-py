"""Visual elements that change and expire over time."""

from typing import Union
import numpy as np

from sidegame.assets import ImageBank
from sidegame.ext import sdglib


class Effect:
    """
    A transient that can be updated during its lifetime and expires
    after it runs out.
    """

    TYPE_NULL = 0
    TYPE_MARK = 1
    TYPE_COLOUR = 2
    TYPE_OVERLAY = 3

    def __init__(self, type_: int, lifetime: float):
        self.type = type_
        self.lifetime = lifetime

    def update(self, dt: float) -> bool:
        """
        Advance effect state and decrement its lifetime, returning a flag
        that signals whether it is still active.

        The advancing step is called before checking lifetime to ensure that
        the effect is processed at least once.
        """

        self.step()

        self.lifetime -= dt

        return self.lifetime > 0.

    def step(self):
        """Advance effect state."""


class Mark(Effect):
    """
    A marked position with single pixel-wide cover and an associated id.
    Includes fading to convey staleness of the mark.
    """

    def __init__(
        self,
        pos_y: int,
        pos_x: int,
        associated_id: int = None,
        opacity: float = 1.,
        lifetime: float = 12.
    ):
        super().__init__(Effect.TYPE_MARK, lifetime)

        self.cover_indices = np.array(0, ndmin=1, dtype=np.int32), np.array(0, ndmin=1, dtype=np.int32)
        self.pos_y = pos_y
        self.pos_x = pos_x
        self.associated_id = associated_id
        self.opacity = opacity
        self.starting_opacity = opacity
        self.starting_lifetime = lifetime

    def step(self):
        self.opacity = self.starting_opacity * self.lifetime / self.starting_lifetime


class Colour(Effect):
    """A cover of uniform colour and opacity."""

    def __init__(
        self,
        cover_indices: np.ndarray,
        colour: np.ndarray,
        lifetime: float,
        pos_y: int = 0,
        pos_x: int = 0,
        opacity: float = 1.
    ):
        super().__init__(Effect.TYPE_COLOUR, lifetime)

        self.cover_indices = cover_indices
        self.pos_y = pos_y
        self.pos_x = pos_x
        self.world_indices = cover_indices[0] + pos_y, cover_indices[1] + pos_x

        self.colour = colour
        self.opacity = opacity

        self.starting_colour = colour
        self.starting_opacity = opacity
        self.starting_lifetime = lifetime

    def step(self):
        self.update_colour()

    def update_colour(self):
        """
        Modify colour and opacity considering the current point in the
        overall lifetime.
        """

    @staticmethod
    def get_disk_indices(radius: int):
        """Get indices of a disk with given radius relative to its centre."""

        rel_indices = np.indices((radius*2 + 1, radius*2 + 1)) - np.array([radius, radius])[..., None, None]

        indices_y, indices_x = np.nonzero(np.linalg.norm(rel_indices, axis=0) <= radius)

        return indices_y - radius, indices_x - radius

    @staticmethod
    def get_line_indices(angle: float, length: int):
        """Get indices of an oriented line with given length relative to its origin."""

        indexed_area_diameter = length*2 + 1
        starting_point = np.array([length, length], dtype=np.float64)
        endpoint = starting_point + np.array([np.cos(angle), np.sin(angle)]) * length

        line_mask = sdglib.mask_ray(indexed_area_diameter, starting_point, endpoint)
        indices_y, indices_x = np.nonzero(line_mask)

        return indices_y - length, indices_x - length


class Explosion(Colour):
    """A bright yellow-coloured disk-shaped effect."""

    COLOUR = ImageBank.COLOURS['yellow']

    def __init__(self, pos: np.ndarray, radius: Union[int, float], lifetime: float):
        cover_indices = self.get_disk_indices(round(radius))
        pos_y = round(pos[1])
        pos_x = round(pos[0])

        super().__init__(cover_indices, self.COLOUR.copy(), lifetime, pos_y=pos_y, pos_x=pos_x, opacity=0.5)

    def update_colour(self):
        lifetime_ratio = self.lifetime / self.starting_lifetime

        if lifetime_ratio > 0.1:
            intensity = (lifetime_ratio - 0.1) * 0.5 / 0.9 + 0.5
        else:
            intensity = lifetime_ratio * 0.5 / 0.1
            self.opacity = intensity

        self.colour = np.uint8(self.starting_colour * intensity)


class Flame(Colour):
    """A warm-coloured disk-shaped effect with waning red component."""

    COLOUR = ImageBank.COLOURS['e_red']

    def __init__(self, pos: np.ndarray, radius: Union[int, float], lifetime: float):
        cover_indices = self.get_disk_indices(round(radius))
        pos_y = round(pos[1])
        pos_x = round(pos[0])

        super().__init__(cover_indices, self.COLOUR.copy(), lifetime, pos_y=pos_y, pos_x=pos_x, opacity=0.5)

    def update_colour(self):
        lifetime_ratio = self.lifetime / self.starting_lifetime

        if lifetime_ratio > 0.1:
            intensity = (lifetime_ratio - 0.1) * 0.5 / 0.9 + 0.5
        else:
            intensity = lifetime_ratio * 0.5 / 0.1
            self.opacity = intensity

        self.colour[2] = np.uint8(self.COLOUR[2] * (1. + intensity)/2.)


class Fog(Colour):
    """A grey disk-shaped effect with fading opacity."""

    COLOUR = ImageBank.COLOURS['grey']

    def __init__(self, pos: np.ndarray, radius: Union[int, float], lifetime: float):
        cover_indices = self.get_disk_indices(round(radius))
        pos_y = round(pos[1])
        pos_x = round(pos[0])

        super().__init__(cover_indices, self.COLOUR.copy(), lifetime, pos_y=pos_y, pos_x=pos_x, opacity=0.8)

    def update_colour(self):
        lifetime_ratio = self.lifetime / self.starting_lifetime

        if lifetime_ratio > 0.1:
            intensity = (lifetime_ratio - 0.1) * 0.4 / 0.9 + 0.4
        else:
            intensity = lifetime_ratio * 0.4 / 0.1

        self.opacity = intensity


class Gunfire(Colour):
    """
    A muzzle flash spewing in the direction faced by the firing entity
    and tied to its initial firing position.
    """

    COLOUR = ImageBank.COLOURS['e_yellow']

    def __init__(self, pos: np.ndarray, angle: float, length: int, lifetime: float):
        cover_indices = self.get_line_indices(angle, length)
        pos_y = round(pos[1])
        pos_x = round(pos[0])

        super().__init__(cover_indices, self.COLOUR.copy(), lifetime, pos_y=pos_y, pos_x=pos_x, opacity=1.)

    def update_colour(self):
        self.opacity = self.starting_opacity * self.lifetime / self.starting_lifetime


class Decal(Colour):
    """
    A marked position with single pixel-wide cover and an associated colour.
    Includes fading to convey staleness of the mark.
    """

    def __init__(
        self,
        pos_y: Union[int, float],
        pos_x: Union[int, float],
        colour: np.ndarray = ImageBank.COLOURS['black'],
        opacity: float = 0.8,
        lifetime: float = np.Inf
    ):
        cover_indices = np.array(0, ndmin=1), np.array(0, ndmin=1)
        pos_y = round(pos_y)
        pos_x = round(pos_x)

        super().__init__(cover_indices, colour.copy(), lifetime, pos_y=pos_y, pos_x=pos_x, opacity=opacity)

    def update_colour(self):
        if self.lifetime != np.Inf:
            self.opacity = self.starting_opacity * self.lifetime / self.starting_lifetime


class Residual(Effect):
    """A flashed and fading residual image overlay."""

    def __init__(self, image: np.ndarray, lifetime: float, flash: float = 0., opacity: float = 1.):
        super().__init__(Effect.TYPE_OVERLAY, lifetime)

        self.overlay = np.clip(image + 192.*flash, 0., 255.).astype(np.uint8) if flash != 1. else image
        self.opacity = opacity
        self.starting_opacity = opacity
        self.starting_lifetime = lifetime

    def step(self):
        self.opacity = self.starting_opacity * self.lifetime / self.starting_lifetime
