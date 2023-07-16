"""Parameter values for items in SDG"""

from copy import copy
from typing import Any, Dict, List, Tuple, Union

from sidegame.game import GameID


class Item:
    """
    An abstract in-game item, i.e. describing its general characteristics,
    but not holding any instance data, e.g. the current state of ammunition.

    NOTE: Some attributes may not be relevant for every item.

    Attributes:
        id: Unique ID, by which the item can be identified in e.g. messaging.
        slot: Indexed space on the HUD belonging to a primary category.
        subslot: Indexed space on the HUD belonging to a secondary category.
        price: Monetary cost of purchasing the item in any round.
        reward: Monetary gain on a kill (of the opponent) with the item.
        velocity_cap: Maximum achievable velocity while holding the item.
        durability_cap: Damage points the item can endure before being lost.
        magazine_cap: Number of item uses before it needs to be reloaded.
        reserve_cap: Number of additional item uses, replenishing the magazine.
        carrying_cap: Number of item instances that can be carried at a time.
        use_pos_offset: Translational recoil on a single item use.
        use_angle_std: Std. deviation of rotational recoil sampled on a single item use.
        use_interval: Time before the item can be reused.
        draw_time: Time before the item becomes usable when drawing it.
        base_damage: Damage to an unarmoured opponent upon a point blank hit.
        armour_pen: Percentage of damage passed to a hit armoured opponent.
        scoped: Flag of whether holding the item limits field-of-view (FOV).
        ammo_per_restore: Amount of item uses made available on a `RLD_ADD` event.
        reload_events: Timestamped sequence of reloading events.
        reload_time: Time until the item becomes usable when reloaded.
        recovery_time: Time until firing inaccuracy drops by an order of magnitude.
        distance_modifier: Factor regulating distance-based damage drop-off.
        standing_inaccuracy: Base inaccuracy (angle deviation in radians) while standing still.
        moving_inaccuracy: Base innacuracy (angle deviation in radians) while moving at maximum velocity.
        firing_inaccuracy: Additional innacuracy (angle deviation in radians) accumulated per shot.
        flash_level: Length of the weapon's muzzle flash (ray) indicating firing direction.
        pellets: Number of shots to trace per single firing.
        fuse_time: Time until the item's effect is triggered.
        duration: Duration of the item's effect.
        radius: Radius of the item's visual and/or functional effect.
    """

    # Main slot index
    SLOT_ARMOUR = 0
    SLOT_PRIMARY = 1
    SLOT_PISTOL = 2
    SLOT_KNIFE = 3
    SLOT_OTHER = 4
    SLOT_UTILITY = 5
    SLOT_NULL = 255

    WEAPON_SLOTS = (SLOT_PRIMARY, SLOT_PISTOL)
    UNDROPPABLE_SLOTS = (SLOT_ARMOUR, SLOT_KNIFE)

    # Utility slot index
    SUBSLOT_NULL = 0
    SUBSLOT_FLASH = 0
    SUBSLOT_EXPLOSIVE = 1
    SUBSLOT_INCENDIARY = 2
    SUBSLOT_SMOKE = 3

    # Interaction flags
    RLD_DRAW = 0
    RLD_START = 1
    RLD_ADD = 2
    RLD_END = 3

    EMPTY_RLD_EVENT_LIST: List[Tuple[float, int]] = []

    def __init__(
        self,
        id_: int = GameID.NULL,
        slot: int = SLOT_NULL,
        subslot: int = SUBSLOT_NULL,
        price: int = 0,
        reward: int = 0,
        velocity_cap: float = 28.43,
        durability_cap: float = 100.,
        magazine_cap: int = 0,
        reserve_cap: int = 0,
        carrying_cap: int = 1,
        use_pos_offset: float = 0.,
        use_angle_std: float = 0.,
        use_interval: float = 1.,
        draw_time: float = 1.,
        base_damage: float = 0.,
        armour_pen: float = 0.,
        scoped: int = False,
        ammo_per_restore: int = 0,
        reload_events: List[Tuple[float, int]] = None,
        reload_time: float = 0.,
        recovery_time: float = 0.5,
        distance_modifier: float = 1.,
        standing_inaccuracy: float = 0.,
        moving_inaccuracy: float = 0.,
        firing_inaccuracy: float = 0.,
        flash_level: int = 0,
        pellets: int = 1,
        fuse_time: float = 0.,
        duration: float = 0.,
        radius: float = 0.
    ):
        self.id = id_
        self.slot = slot
        self.subslot = subslot
        self.price = price
        self.reward = reward
        self.velocity_cap = velocity_cap
        self.durability_cap = durability_cap
        self.magazine_cap = magazine_cap
        self.reserve_cap = reserve_cap
        self.carrying_cap = carrying_cap
        self.use_pos_offset = use_pos_offset
        self.use_angle_std = use_angle_std
        self.use_interval = use_interval
        self.draw_time = draw_time

        self.base_damage = base_damage
        self.armour_pen = armour_pen

        self.scoped = scoped
        self.ammo_per_restore = ammo_per_restore
        self.reload_events = reload_events if reload_events is not None else self.EMPTY_RLD_EVENT_LIST
        self.reload_time = reload_time
        self.recovery_time = recovery_time
        self.distance_modifier = distance_modifier
        self.standing_inaccuracy = standing_inaccuracy
        self.moving_inaccuracy = moving_inaccuracy
        self.firing_inaccuracy = firing_inaccuracy
        self.flash_level = flash_level
        self.pellets = pellets

        self.fuse_time = fuse_time
        self.duration = duration
        self.radius = radius

        self.icon: Any = None
        self.sounds: Dict[str, List[Any]] = None

    def init_assets(self, name: str, images: dict[str, Any], sounds: dict[str, Any]):
        """
        Load the assets pointed to by preset paths.

        Expected assignment types:
            icon: 3D (4-channel) array
            sounds: Annotated lists of 2D (2-channel) arrays
        """

        self.icon = images[f'item_{name}']
        self.sounds = sounds[name]

    def get_sound(self, sound_key: str) -> Union[List[Any], None]:
        """Try to retrieve a specific sound, returning `None` if not found or inaccessible."""

        return self.sounds.get(sound_key, None) if self.sounds is not None else None


class Inventory:
    """
    A container of select items based on their CSGO counterparts.

    The selection mostly intended to group specific classes into one
    representative item, e.g. the AK represents the T-side rifle class.
    This gives fewer options for situational buys, but should cover
    the standard/recommended buys well enough.

    NOTE: There are some functional differences, e.g. every weapon being able
    to fire automatically when holding down the trigger.

    An uninitialised `Inventory` allows addressing contained items
    and their parameters, but their `icon` and `sounds` attributes
    are only set upon initialisation, where the items are copied
    (owned) and their assets loaded.

    Sources:
    https://counterstrike.fandom.com/wiki/Weapons
    https://counterstrike.fandom.com/wiki/Money
    https://counterstrike.fandom.com/wiki/Movement
    https://counterstrike.fandom.com/wiki/Damage_dropoff
    Dinoswarleaf: CS:GO's Grenades Timing & Damage Analysis | https://www.youtube.com/watch?v=Cd80AYP59qE
    BlackRetina & SlothSquadron: CSGO Weapon Spreadsheet |
        https://docs.google.com/spreadsheets/d/11tDzUNBq9zIX6_9Rel__fdAUezAQzSnh5AVYzCP060c/edit#gid=8
    """

    armour = Item(
        id_=GameID.ITEM_ARMOUR,
        slot=Item.SLOT_ARMOUR,
        price=650)

    rifle_t = Item(
        id_=GameID.ITEM_RIFLE_T,
        slot=Item.SLOT_PRIMARY,
        price=2700,
        reward=300,
        velocity_cap=24.49,
        magazine_cap=30,
        reserve_cap=90,
        use_pos_offset=-0.3,
        use_angle_std=8.367e-3,
        use_interval=0.1,
        draw_time=1.,
        base_damage=36.,
        armour_pen=0.775,
        ammo_per_restore=30,
        reload_events=[(1.17, Item.RLD_ADD), (2.43-1./3., Item.RLD_END)],
        reload_time=2.43,
        recovery_time=0.37,
        distance_modifier=0.98,
        standing_inaccuracy=0.014,
        moving_inaccuracy=0.3602,
        firing_inaccuracy=0.0156,
        flash_level=4)

    rifle_ct = Item(
        id_=GameID.ITEM_RIFLE_CT,
        slot=Item.SLOT_PRIMARY,
        price=3100,
        reward=300,
        velocity_cap=25.59,
        magazine_cap=30,
        reserve_cap=90,
        use_pos_offset=-0.23,
        use_angle_std=8.367e-3,
        use_interval=0.09,
        draw_time=1.13,
        base_damage=33.,
        armour_pen=0.7,
        ammo_per_restore=30,
        reload_events=[(1.37, Item.RLD_ADD), (3.07-1.13/3., Item.RLD_END)],
        reload_time=3.07,
        recovery_time=0.43,
        distance_modifier=0.97,
        standing_inaccuracy=0.011,
        moving_inaccuracy=0.2848,
        firing_inaccuracy=0.014,
        flash_level=4)

    smg_t = Item(
        id_=GameID.ITEM_SMG_T,
        slot=Item.SLOT_PRIMARY,
        price=1200,
        reward=600,
        velocity_cap=26.15,
        magazine_cap=25,
        reserve_cap=100,
        use_pos_offset=-0.23,
        use_angle_std=6.324e-3,
        use_interval=0.09,
        draw_time=1.,
        base_damage=35.,
        armour_pen=0.65,
        ammo_per_restore=25,
        reload_events=[(1.50, Item.RLD_ADD), (3.43-1./3., Item.RLD_END)],
        reload_time=3.43,
        recovery_time=0.35,
        distance_modifier=0.75,
        standing_inaccuracy=0.0289,
        moving_inaccuracy=0.0863,
        firing_inaccuracy=0.0068,
        flash_level=3)

    smg_ct = Item(
        id_=GameID.ITEM_SMG_CT,
        slot=Item.SLOT_PRIMARY,
        price=1250,
        reward=600,
        velocity_cap=27.29,
        magazine_cap=30,
        reserve_cap=120,
        use_pos_offset=-0.19,
        use_angle_std=8.367e-3,
        use_interval=0.07,
        draw_time=1.2,
        base_damage=26.,
        armour_pen=0.6,
        ammo_per_restore=30,
        reload_events=[(0.87, Item.RLD_ADD), (2.13-1.2/3., Item.RLD_END)],
        reload_time=2.13,
        recovery_time=0.26,
        distance_modifier=0.87,
        standing_inaccuracy=0.0192,
        moving_inaccuracy=0.0772,
        firing_inaccuracy=0.0074,
        flash_level=3)

    shotgun_t = Item(
        id_=GameID.ITEM_SHOTGUN_T,
        slot=Item.SLOT_PRIMARY,
        price=1050,
        reward=900,
        velocity_cap=25.02,
        magazine_cap=8,
        reserve_cap=32,
        use_pos_offset=-1.43,
        use_angle_std=4.472e-3,
        use_interval=0.88,
        draw_time=1.,
        base_damage=26.,
        armour_pen=0.5,
        ammo_per_restore=1,
        reload_events=[(0.54 * shell_idx, Item.RLD_ADD) for shell_idx in range(1, 9)] + [(4.74-1./3., Item.RLD_END)],
        reload_time=4.74,
        recovery_time=0.46,
        distance_modifier=0.7,
        standing_inaccuracy=0.0939,
        moving_inaccuracy=0.1671,
        firing_inaccuracy=0.0194,
        flash_level=3,
        pellets=9)

    shotgun_ct = Item(
        id_=GameID.ITEM_SHOTGUN_CT,
        slot=Item.SLOT_PRIMARY,
        price=1300,
        reward=900,
        velocity_cap=25.59,
        magazine_cap=5,
        reserve_cap=32,
        use_pos_offset=-1.65,
        use_angle_std=4.472e-3,
        use_interval=0.85,
        draw_time=1.,
        base_damage=30.,
        armour_pen=0.75,
        ammo_per_restore=5,
        reload_events=[(1.07, Item.RLD_ADD), (2.47-1./3., Item.RLD_END)],
        reload_time=2.47,
        recovery_time=0.4,
        distance_modifier=0.45,
        standing_inaccuracy=0.0939,
        moving_inaccuracy=0.1258,
        firing_inaccuracy=0.0224,
        flash_level=3,
        pellets=8)

    sniper = Item(
        id_=GameID.ITEM_SNIPER,
        slot=Item.SLOT_PRIMARY,
        price=4750,
        reward=100,
        velocity_cap=11.37,
        magazine_cap=10,
        reserve_cap=30,
        use_pos_offset=-0.78,
        use_angle_std=4.472e-3,
        use_interval=1.46,
        draw_time=1.25,
        base_damage=115.,
        armour_pen=0.975,
        scoped=True,
        ammo_per_restore=10,
        reload_events=[(2., Item.RLD_ADD), (3.67-1.25/3., Item.RLD_END)],
        reload_time=3.67,
        recovery_time=0.35,
        distance_modifier=0.99,
        standing_inaccuracy=0.0044,
        moving_inaccuracy=0.5040,
        firing_inaccuracy=0.1076,
        flash_level=5)

    pistol_t = Item(
        id_=GameID.ITEM_PISTOL_T,
        slot=Item.SLOT_PISTOL,
        price=None,
        reward=300,
        velocity_cap=27.29,
        magazine_cap=20,
        reserve_cap=120,
        use_pos_offset=-0.18,
        use_angle_std=4.472e-3,
        use_interval=0.15,
        draw_time=1.1,
        base_damage=30.,
        armour_pen=0.47,
        ammo_per_restore=20,
        reload_events=[(0.93, Item.RLD_ADD), (2.27-1.1/3., Item.RLD_END)],
        reload_time=2.27,
        recovery_time=0.2,
        distance_modifier=0.85,
        standing_inaccuracy=0.0152,
        moving_inaccuracy=0.0352,
        firing_inaccuracy=0.1119,
        flash_level=2)

    pistol_ct = Item(
        id_=GameID.ITEM_PISTOL_CT,
        slot=Item.SLOT_PISTOL,
        price=None,
        reward=300,
        velocity_cap=27.29,
        magazine_cap=12,
        reserve_cap=24,
        use_pos_offset=-0.23,
        use_angle_std=0.,
        use_interval=0.17,
        draw_time=1.,
        base_damage=35.,
        armour_pen=0.505,
        ammo_per_restore=12,
        reload_events=[(0.97, Item.RLD_ADD), (2.17-1./3., Item.RLD_END)],
        reload_time=2.17,
        recovery_time=0.35,
        distance_modifier=0.91,
        standing_inaccuracy=0.0128,
        moving_inaccuracy=0.0405,
        firing_inaccuracy=0.1418,
        flash_level=2)

    knife = Item(
        id_=GameID.ITEM_KNIFE,
        slot=Item.SLOT_KNIFE,
        reward=1500,
        use_pos_offset=-0.5,
        use_angle_std=0.003,
        use_interval=1.,
        draw_time=1.,
        base_damage=65.,
        armour_pen=0.85)

    c4 = Item(
        id_=GameID.ITEM_C4,
        slot=Item.SLOT_OTHER,
        subslot=Item.SUBSLOT_NULL,
        base_damage=500.,
        armour_pen=0.6,
        radius=204.)

    dkit = Item(
        id_=GameID.ITEM_DKIT,
        slot=Item.SLOT_OTHER,
        subslot=Item.SUBSLOT_NULL,
        price=400)

    flash = Item(
        id_=GameID.ITEM_FLASH,
        slot=Item.SLOT_UTILITY,
        subslot=Item.SUBSLOT_FLASH,
        price=200,
        reward=300,
        velocity_cap=27.86,
        carrying_cap=2,
        use_pos_offset=-0.5,
        use_angle_std=0.003,
        fuse_time=1.7,
        radius=5.)

    explosive = Item(
        id_=GameID.ITEM_EXPLOSIVE,
        slot=Item.SLOT_UTILITY,
        subslot=Item.SUBSLOT_EXPLOSIVE,
        price=300,
        reward=300,
        velocity_cap=27.86,
        use_pos_offset=-0.5,
        use_angle_std=0.003,
        base_damage=98.,
        armour_pen=0.575,
        fuse_time=1.7,
        radius=44.7)

    incendiary_t = Item(
        id_=GameID.ITEM_INCENDIARY_T,
        slot=Item.SLOT_UTILITY,
        subslot=Item.SUBSLOT_INCENDIARY,
        price=400,
        reward=300,
        velocity_cap=27.86,
        use_pos_offset=-0.5,
        use_angle_std=0.003,
        base_damage=280.,
        armour_pen=1.,
        fuse_time=2.1,
        duration=7.,
        radius=12.1)

    incendiary_ct = Item(
        id_=GameID.ITEM_INCENDIARY_CT,
        slot=Item.SLOT_UTILITY,
        subslot=Item.SUBSLOT_INCENDIARY,
        price=600,
        reward=300,
        velocity_cap=27.86,
        base_damage=280.,
        use_pos_offset=-0.5,
        use_angle_std=0.003,
        armour_pen=1.,
        fuse_time=2.1,
        duration=7.,
        radius=12.1)

    smoke = Item(
        id_=GameID.ITEM_SMOKE,
        slot=Item.SLOT_UTILITY,
        subslot=Item.SUBSLOT_SMOKE,
        price=300,
        reward=300,
        velocity_cap=27.86,
        use_pos_offset=-0.5,
        use_angle_std=0.003,
        fuse_time=2.1,
        duration=18.,
        radius=16.8)

    _ITEM_KEYS = (
        'armour',
        'rifle_t',
        'rifle_ct',
        'smg_t',
        'smg_ct',
        'shotgun_t',
        'shotgun_ct',
        'sniper',
        'pistol_t',
        'pistol_ct',
        'knife',
        'c4',
        'dkit',
        'flash',
        'explosive',
        'incendiary_t',
        'incendiary_ct',
        'smoke')

    def __init__(self, images: dict[str, Any] = None, sounds: dict[str, Any] = None):
        self._id_dict: Dict[int, Item] = {}

        for item_key in self._ITEM_KEYS:
            item = getattr(self, item_key)

            if images is not None and sounds is not None:
                item = copy(item)
                item.init_assets(item_key, images, sounds)
                setattr(self, item_key, item)

            self._id_dict[item.id] = item

    def get_item_by_id(self, item_id: int) -> Union[Item, None]:
        """Access an item in the internal dict of items."""

        return self._id_dict.get(item_id, None)
