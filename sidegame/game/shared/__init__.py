from sidegame.game.shared.inventory import Item, Inventory
from sidegame.game.shared.objects import Object, Weapon, Knife, Flash, Explosive, Incendiary, Smoke, C4
from sidegame.game.shared.player import Player
from sidegame.game.shared.session import Session


class Message:
    """In-game chat message structure."""

    def __init__(
        self,
        position_id: int,
        round_: int,
        time_: float,
        words: list[int],
        marks: list[tuple[int | float, ...]] = None,
        sender_id: int = None
    ):
        self.position_id = position_id
        self.round = round_
        self.time = time_
        self.words = words
        self.marks = marks
        self.sender_id = sender_id
