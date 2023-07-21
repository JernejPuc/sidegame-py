import logging
import sys
from collections import deque
from time import sleep
from typing import Any, Callable

from sdl2 import SDL_Delay


def get_logger(name: str = None, path: str = None, level: int = logging.DEBUG) -> logging.Logger:
    """Get a file or stdout logger with preset format."""

    log_handler = logging.FileHandler(path) if path is not None else logging.StreamHandler(sys.stdout)
    log_handler.setLevel(level)
    log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    logger = logging.getLogger(name=name)
    logger.setLevel(level)
    logger.addHandler(log_handler)

    return logger


class MovingAverageTracker:
    """
    Tracks the values within the last time window of specified length.

    NOTE: Displayed values are rounded to 1 decimal, so smaller values
    need to be appropriately scaled before updating the tracker.

    NOTE: Could be optimised to update the tracked value directly,
    instead of summing the entire window with every iteration,
    but improvement should be negligible wrt. everything else.
    See: https://docs.python.org/3/library/collections.html#deque-recipes
    """

    LEVEL_OFF = 0
    LEVEL_TRACK = 1
    LEVEL_DISPLAY = 2

    def __init__(
        self,
        window_length: int,
        display_prefix: str = '',
        display_suffix: str = '',
        level: int = LEVEL_TRACK
    ):
        self.window_length = window_length
        self.display_prefix = display_prefix
        self.display_suffix = display_suffix

        self.level = level
        self.track: bool = level >= self.LEVEL_TRACK
        self.display: bool = level == self.LEVEL_DISPLAY

        self.offset = 0.
        self.value = 0.
        self.buffer: deque[float] = deque()
        self.buffer_full = False

    def update(self, value: float):
        """Update moving average by discarding the oldest and adding the newest value."""

        if not self.track:
            return

        if self.buffer_full:
            self.buffer.popleft()
            self.buffer.append(value)
            self.set_value()

        else:
            self.buffer.append(value)

            if len(self.buffer) >= self.window_length:
                self.buffer_full = True

        if self.display:
            print(f'{self.display_prefix}{self.value:.1f}{self.display_suffix}', end='')

    def set_value(self):
        """Get the offset average of currently buffered values."""

        self.value = sum(self.buffer) / len(self.buffer) + self.offset

    def set_level(self, level: int):
        """Set tracker level to enable or disable it on the fly."""

        self.level = level
        self.track = level >= self.LEVEL_TRACK
        self.display = level == self.LEVEL_DISPLAY

    def set_window_length(self, window_length: int):
        """Set new window length and accordingly reduce or unlock the value buffer."""

        self.window_length = window_length

        if len(self.buffer) >= self.window_length:
            while len(self.buffer) > self.window_length:
                self.buffer.popleft()

        else:
            self.buffer_full = False

    def reset(self):
        """Reset the tracker by clearing its value buffer."""

        self.buffer.clear()
        self.buffer_full = False


class TickLimiter(MovingAverageTracker):
    """
    Tracks the moving average of ticks in a 1-second window (FPS)
    and limits them with delays to target the specified upper bound.

    NOTE: On Windows, the built-in `sleep` method is not precise enough
    to be useful for accurate delays, with 1ms targets often resulting
    in delays in the range of 2-10ms or worse.

    In comparison, `SDL_Delay` is most often within 10% of the 1ms target,
    but it can only work with discrete ms steps instead of fractions.
    Because `sleep` seems to work fine on Linux, the delay function is
    system-specific.

    See: https://stackoverflow.com/questions/1133857/how-accurate-is-pythons-time-sleep/15967564#15967564
    """

    _MIN_DELAY = 1e-3

    def __init__(
        self,
        tick_rate: float,
        display_prefix: str = '',
        display_suffix: str = '',
        level: int = MovingAverageTracker.LEVEL_DISPLAY
    ):
        super().__init__(round(tick_rate), display_prefix=display_prefix, display_suffix=display_suffix, level=level)

        self.tick_rate = tick_rate
        self.update_interval = 1. / tick_rate
        self.delay: Callable = (lambda t: SDL_Delay(round(t * 1e+3))) if sys.platform == 'win32' else sleep

    def update_and_delay(self, dt_loop: float, t_clock: float):
        """
        Update moving average of FPS and determine the time to delay
        until the next update.

        If a frame skip is detected, where the main loop duration exceeded
        the update interval, there is no delay. This can happen randomly
        and with minimal consequences, e.g. if the targeted delay was only
        slightly overstepped, or consistently, i.e. if FPS is bound by
        the resources of the CPU instead.
        """

        self.update(t_clock)

        if dt_loop < self.update_interval:
            self.delay(self.update_interval - t_clock % self.update_interval)

    def set_value(self):
        """Override `set_value` to reflect the temporal nature of the tracked value."""

        self.value = self.tick_rate / (self.buffer[-1] - self.buffer[0] + self.update_interval)

    def set_tick_rate(self, new_tick_rate: float):
        """
        Set new tick rate and accordingly update the attributes associated with
        tracking and delays.
        """

        self.tick_rate = new_tick_rate
        self.update_interval = 1. / new_tick_rate
        self.set_window_length(round(new_tick_rate))


class StridedFunction:
    """
    A wrapper around a callable that conditionally executes with given stride
    to detach specific periodic operations, e.g. polling events or sending data,
    from higher framerate.

    NOTE: Because ticks are discrete and might not nicely match the stride,
    the effective stride may vary and the actual number of calls per time window
    could end up lower than expected.
    """

    def __init__(self, fn: Callable, stride: int | float):
        self._fn = fn
        self._stride = max(1, stride)
        self._counter: int | float = 1
        self._regular = self._stride == 1

    def __call__(self, *args, **kwargs) -> Any | None:
        """
        Call and relay the result of the wrapped function if its stride
        is reached, resetting its counter. Otherwise, increment it and return.
        """

        if self._regular:
            return self._fn(*args, **kwargs)

        elif self._counter >= self._stride:
            self._counter = self._counter - self._stride + 1
            return self._fn(*args, **kwargs)

        else:
            self._counter += 1
            return None
