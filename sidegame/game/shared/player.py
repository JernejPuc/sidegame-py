"""
Contained player-related variables and rules within SDG

References:
https://developer.valvesoftware.com/wiki/Source_Multiplayer_Networking#Lag_compensation
https://developer.valvesoftware.com/wiki/Latency_Compensating_Methods_in_Client/Server_In-game_Protocol_Design_and_Optimization
https://www.gabrielgambetta.com/lag-compensation.html
"""

from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from sidegame.physics import PlayerEntity
from sidegame.networking.core import Entry, Action, Entity
from sidegame.game.shared.core import GameID, Map, Event
from sidegame.game.shared.inventory import Item, Inventory
from sidegame.game.shared.objects import Object, Weapon, Knife, Flash, Explosive, Incendiary, Smoke


class Player(Entity, PlayerEntity):
    """
    An agent, acting upon and interacting with the game world.

    NOTE: Some aspects are only applicable server-side, e.g. lag compensation.
    """

    MAX_BUY_RANGE = 64.
    MAX_DROP_RANGE = 16.
    MAX_PICKUP_RANGE = 16.
    MAX_ACCEPTABLE_LAG = 0.25  # Includes artificial interp. window lag
    TAGGING_RECOVERY_TIME = 0.5
    AIM_PUNCH_DAMAGE_REFERENCE = 10.

    LOG10E = np.log10(np.e)
    DT64 = 1. / (64. - 1.)  # Tick-independent formula factor, standardised to 64 ticks

    OBJECT_DROP_VELOCITY = 28.43
    GRENADE_THROW_VELOCITY = OBJECT_DROP_VELOCITY * 2.
    HEALTH_AT_ROUND_START = 100.
    MONEY_AT_SIDE_START = 800
    MONEY_CAP = 16000

    SPAWN_ANGLES = np.array([deg*np.pi/180. for deg in [180., -108., -36., 36., 108.]])
    SPAWN_OFFSETS = np.array([[14. * np.cos(2.*np.pi*i/5.), 14. * np.sin(2.*np.pi*i/5.)] for i in np.arange(5.)])

    def __init__(self, player_id: int, inventory: Inventory, rng: np.random.Generator = None):
        Entity.__init__(self, player_id)
        PlayerEntity.__init__(self, player_id=player_id, rng=rng)

        self.inventory = inventory
        self.name = '    '
        self.role: int = GameID.ROLE_SPECTATOR
        self.latency = 0.
        self.global_buy_enabled = False

        # NOTE: The player is reassigned after initialisation
        self.team = GameID.GROUP_SPECTATORS
        self.position_id = player_id

        self.kills = 0
        self.deaths = 0
        self.money = 0
        self.health = 0.

        self.queued_damage = 0.
        self.recent_damage_taken = 0.
        self.time_since_damage_taken = 0.
        self.time_until_drawn = 0.
        self.time_to_reload = 0.
        self.time_off_trigger = 0.
        self.accumulated_firing_inaccuracy = 0.

        self.d_pos_recoil = np.array([0., 0.])
        self.d_angle_recoil = 0.

        self.pos_history: Deque[np.ndarray] = deque()

        self.slots: List[Union[Object, None]] = [None] * 9  # 1 armour slot + 8 drawable slots
        self.held_object: Object = None
        self.reload_events: Deque[Tuple[float, int]] = deque()

    def reset_side(self):
        """Reset money and carried objects."""

        self.money = self.MONEY_AT_SIDE_START

        knife = Knife(self.inventory.knife, self)

        self.slots = [None] * 9
        self.slots[Item.SLOT_KNIFE] = knife
        self.draw(knife)

    def reset_round(self, spawn_origin: np.ndarray, player_map: np.ndarray):
        """Reset health, position, movement, and inventory state."""

        # Reset position
        new_pos = spawn_origin + self.SPAWN_OFFSETS[(self.position_id-1) % 5]
        self.angle = self.SPAWN_ANGLES[(self.position_id-1) % 5]

        self.update_collider_map(
            player_map, self.pos, new_pos, claim_id=self.id, clear_id=Map.PLAYER_ID_NULL, check_cleared_area=True)

        self.pos = new_pos

        # Reset movement
        self.vel.fill(0.)
        self.acc.fill(0.)

        # Take away C4 if it was left unplanted
        if self.team == GameID.GROUP_TEAM_T:
            self.slots[Item.SLOT_OTHER] = None

        # Pistol given by default
        if self.slots[Item.SLOT_PISTOL] is None:
            pistol_item = self.inventory.pistol_t if self.team == GameID.GROUP_TEAM_T else self.inventory.pistol_ct
            pistol = Weapon(pistol_item, self, rng=self.rng)
            self.slots[Item.SLOT_PISTOL] = pistol

        # Start at full ammo
        for slot in Item.WEAPON_SLOTS:
            if self.slots[slot] is not None:
                self.slots[slot].reset_ammunition()

        # Reset decaying or queued variables
        self.queued_damage = 0.
        self.recent_damage_taken = 0.
        self.time_since_damage_taken = 0.
        self.time_until_drawn = 0.
        self.time_to_reload = 0.
        self.time_off_trigger = 0.
        self.accumulated_firing_inaccuracy = 0.

        self.d_pos_recoil.fill(0.)
        self.d_angle_recoil = 0.

        self.reload_events.clear()

        # Reset health
        self.health = self.HEALTH_AT_ROUND_START

    def set_name(self, name: str) -> Event:
        """Set name, returning event with current player data."""

        self.name = name

        return Event(Event.CTRL_PLAYER_CHANGED, (self.id, name, self.money, self.global_buy_enabled))

    def set_team(self, team: int, position_id: int) -> Event:
        """
        Set team and position within it, as determined by the session,
        returning the corresponding event.
        """

        self.team = team
        self.position_id = position_id

        return Event(Event.CTRL_PLAYER_MOVED, (self.id, team, position_id))

    def add_money(self, money: int):
        """Clip money loss or gain within expected range."""

        self.money = max(0, min(self.MONEY_CAP, self.money + money))

    def update(
        self,
        action: Action,
        players: Dict[int, 'Player'],
        objects: Dict[int, Object],
        map_: Map,
        timestamp: float,
        total_lag: float,
        grounded: bool
    ) -> Iterable[Event]:
        """
        Update own state according to an input and generate events to be
        handled by the global state update.

        Employs lag compensation:
        A deque of past timestamped positions is kept (down to `-MAX_LAG`)
        to verify shots. The closest timestamp following the target time
        is used to temporarily repopulate the collider map of players
        and trace the shot according to it.
        Lag compensation is not performed for actions that exceed `MAX_LAG`.
        """

        attack, walking, _, use, drop, rld, draw_id, _, cursor_y, force_w, force_d, \
            _, view, hovered_id, hovered_entity_id, d_angle = action.data

        # In buy phase, prevent movement and firing
        if grounded and not self.global_buy_enabled:
            force_w = 0
            force_d = 0
            attack = 0

        # NOTE: Messaged movement forces are strictly positive (B format) and need to be recentred
        else:
            force_w = force_w - 1
            force_d = force_d - 1

        # Get time since last update
        dt = (timestamp - self.states[-1].timestamp) if self.states else 0.

        # Decay/advance time-dependent variables
        self.decay(dt, recoil=True)

        events = deque()

        # Advance progress of ongoing reload
        if self.time_to_reload:
            rld_event = self.eval_reload(drop or draw_id)

            if rld_event is not None:
                events.append(rld_event)

        # Attack
        can_attack = not (
            self.time_until_drawn or self.time_to_reload or self.time_off_trigger < self.held_object.item.use_interval)

        if attack and can_attack and view == GameID.VIEW_WORLD:
            # Only weapons are eligible for lag compensation
            if self.held_object.item.slot in Item.WEAPON_SLOTS and total_lag and total_lag < self.MAX_ACCEPTABLE_LAG:
                timestamp_of_client_view = action.timestamp - total_lag

                # Rewind to state seen by the client
                for player in players.values():
                    if player.team != GameID.GROUP_SPECTATORS and player.id != self.id and player.health:
                        player.rewind(timestamp_of_client_view, map_.player_id)

                events.extend(self.attack(map_.height, map_.player_id, cursor_y, players))

                # Wind back to present state
                for player in players.values():
                    if player.team != GameID.GROUP_SPECTATORS and player.id != self.id and player.health:
                        player.wind_back(map_.player_id)

            else:
                events.extend(self.attack(map_.height, map_.player_id, cursor_y, players))

            self.time_off_trigger = 0.

        # Update plant
        if self.team == GameID.GROUP_TEAM_T:
            focus = self.held_object.item.id == GameID.ITEM_C4 and use and not attack and self.state == self.STATE_STILL

            c4 = self.slots[Item.SLOT_OTHER]

            if c4 is not None:
                c4_event = c4.try_plant(self.pos, map_.landmark, focus, action.timestamp)

                if c4_event is not None:
                    events.append(c4_event)

                    if c4_event.type == Event.C4_PLANTED:
                        events.extend(self.drop(c4, throw_len=0., throw_vel=0.))

        # Store data to update defuse in regular state update
        else:
            focus = hovered_id == GameID.ITEM_C4 and use and not attack
            kit_available = self.slots[Item.SLOT_OTHER] is not None

            while self.actions and (action.timestamp - self.actions[0].timestamp) > self.MAX_ACCEPTABLE_LAG:
                del self.actions[0]

            self.actions.append(
                Action(action.type, action.counter, action.timestamp, (self.pos, focus, kit_available), dt=dt))

        # Drop held item, pick up hovered item, or start reload sequence
        if not focus:
            if drop:
                drop_events = self.drop(self.held_object)

                if drop_events is not None:
                    events.extend(drop_events)

            if use:
                pickup_events = self.pick_up(hovered_entity_id, map_, objects)

                if pickup_events is not None:
                    events.extend(pickup_events)

                    # NOTE: Only cancel reload on successful pickup,
                    # otherwise it could be used to instantly cancel its 'animation'
                    self.time_to_reload = 0.

            if draw_id:
                item = self.inventory.get_item_by_id(draw_id)
                obj = self.slots[item.slot + item.subslot]

                if obj is not None and obj is not self.held_object:
                    self.draw(obj)
                    events.append(Event(Event.PLAYER_RELOAD, (self.id, Item.RLD_DRAW, self.held_object.item.id)))

            if rld:
                rld_event = self.reload()

                if rld_event is not None:
                    events.append(rld_event)

        # Max velocity determined by currently held item and recently taken damage
        max_vel = self.held_object.item.velocity_cap if not walking else (self.held_object.item.velocity_cap / 2.)
        max_vel *= self.get_tagging_factor()

        # Update position and movement state
        footstep_event = self.move(dt, force_w, force_d, d_angle, walking, max_vel, map_.height, map_.player_id)

        if footstep_event:
            events.append(Event(Event.FX_FOOTSTEP, self.id))

        self.update_collider_map(
            map_.player_id,
            self.states[-1].data if self.states else self.pos,
            self.pos,
            claim_id=self.id,
            clear_id=Map.PLAYER_ID_NULL)

        self.update_position_history(Entry(self.id, Entry.TYPE_STATE, action.counter, timestamp, self.pos))

        return events

    def eval_damage(
        self,
        dmg_event: Event,
        map_: Map,
        check_death: bool = True,
        recoil: bool = False,
        players: Dict[int, 'Player'] = None
    ) -> Iterable[Event]:
        """
        Calculate and apply effective damage, possibly extending into
        death evaluation.

        References:
        Dibolaris comments on 'How exactly does armor even work?' |
            https://old.reddit.com/r/GlobalOffensive/comments/2762i5/how_exactly_does_armor_even_work/chxrrxl/
        3kliksphilip: CS:GO - The Importance of Armour and when to rebuy |
            https://www.youtube.com/watch?v=n5xdMo6kj00
        """

        # Multiple damage events can be raised before their effect is evaluated,
        # which could lead to multiple deaths if left unhandled
        if not self.health:
            return Event.EMPTY_EVENT_LIST

        attacker_id, victim_id, item_id, dmg = dmg_event.data
        apen = self.inventory.get_item_by_id(item_id).armour_pen
        armour = self.slots[Item.SLOT_ARMOUR]

        # Apply armour penetration
        if armour is not None:
            armour.durability -= (1. - apen) * dmg / 2.

            if armour.durability < 0.5:
                self.slots[Item.SLOT_ARMOUR] = None

            dmg = apen * dmg

        # Apply effective damage
        self.health -= dmg

        # Add recoil
        if recoil:
            self.d_pos_recoil += self.get_aim_punch(players[attacker_id].pos, dmg)

        # Extend into death event
        if check_death and self.health < 0.5:
            excess = abs(0.5 - self.health)
            self.health = 0.

            death_event = Event(Event.PLAYER_DEATH, (attacker_id, victim_id, item_id, excess))

            return self.eval_death(death_event, map_)

        # Queue damage for `recent_damage_taken`
        else:
            self.queued_damage += dmg

        return Event.EMPTY_EVENT_LIST

    def eval_death(self, death_event: Event, _map: Map) -> Iterable[Event]:
        """Drop carried objects and clear occupied collider map area."""

        events = deque()
        events.append(death_event)

        # Drop all (droppable) objects
        for obj_idx in range(len(self.slots)):
            obj = self.slots[obj_idx]

            # Iter for multi-carried objects
            if obj is not None:
                while obj.carrying:
                    drop_events = self.drop(obj, rng=True)

                    if drop_events is not None:
                        events.extend(drop_events)
                    else:
                        break

                    # Dropping items resets their carried counter to 1 to allow them to be picked up,
                    # but it also messes up this loop without this check
                    if self.slots[obj_idx] is None:
                        break

        # Armour is not dropped, lose it explicitly
        if self.slots[Item.SLOT_ARMOUR] is not None:
            obj = self.slots[Item.SLOT_ARMOUR]
            self.slots[Item.SLOT_ARMOUR] = None
            events.append(Event(Event.OBJECT_ASSIGN, (self.id, obj.item.id, 0., 0, 0, 0, self.money, 0)))

        # Clear collider map
        _map.player_id[self.get_covered_indices()] = Map.PLAYER_ID_NULL

        # Reset decaying or queued variables
        self.queued_damage = 0.
        self.recent_damage_taken = 0.
        self.time_since_damage_taken = 0.
        self.time_until_drawn = 0.
        self.time_to_reload = 0.
        self.time_off_trigger = 0.
        self.accumulated_firing_inaccuracy = 0.

        self.d_pos_recoil.fill(0.)
        self.d_angle_recoil = 0.

        self.reload_events.clear()

        return events

    def get_aim_punch(self, incoming_pos: np.ndarray, effective_damage: float) -> np.ndarray:
        """Return offset of local viewing position after being hit."""

        d_pos = self.pos - incoming_pos

        # Rotate into local view
        angle = -(self.angle + np.pi/2.)

        rotmat = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]])

        d_pos = np.dot(rotmat, d_pos)

        return d_pos / max(np.linalg.norm(d_pos), 1.) * min(effective_damage, 33.) / self.AIM_PUNCH_DAMAGE_REFERENCE

    def get_tagging_factor(self):
        """
        Simplified tagging system, tying slow-down to all recent damage taken.

        NOTE: This simplification results in another discrepancy from CSGO,
        where incendiaries are not supposed to slow you down.

        References:
        https://counterstrike.fandom.com/wiki/Tagging
        https://steamcommunity.com/sharedfiles/filedetails/?id=412879303
        """

        return 1. if self.recent_damage_taken < 0.5 else (0.5 - 3e-3 * self.recent_damage_taken)

    def decay(self, dt: float, recoil: bool = False):
        """
        Advance time-dependent variables. Each tick-dependent formula
        uses normalised time arguments so that they are more consistent
        across different tick rates.

        References:
        BlackRetina & SlothSquadron: CSGO Weapon Spreadsheet |
            https://docs.google.com/spreadsheets/d/11tDzUNBq9zIX6_9Rel__fdAUezAQzSnh5AVYzCP060c/edit#gid=0
        """

        tick_norm = dt/self.DT64

        # Decay use obstructions
        self.time_to_reload = max(self.time_to_reload - dt, 0.)
        self.time_until_drawn = max(self.time_until_drawn - dt, 0.)
        self.time_off_trigger += dt

        # Decay tagging
        if self.queued_damage:
            self.recent_damage_taken += self.queued_damage
            self.time_since_damage_taken = 0.
            self.queued_damage = 0.

        else:
            self.time_since_damage_taken += dt
            self.recent_damage_taken *= np.exp(
                -self.time_since_damage_taken * tick_norm / (self.LOG10E * self.TAGGING_RECOVERY_TIME))

        # Decay innacuracy
        self.accumulated_firing_inaccuracy *= np.exp(
            -self.time_off_trigger * tick_norm / (self.LOG10E * self.held_object.item.recovery_time))

        # Decay recoil
        if recoil:
            recoil_recovery_time = self.held_object.item.use_interval * \
                (1. - np.exp(-self.held_object.item.recovery_time / self.held_object.item.use_interval))

            self.d_angle_recoil *= np.exp(
                -(self.time_off_trigger * tick_norm / (self.LOG10E * recoil_recovery_time)))

            self.d_pos_recoil *= (
                np.exp(-(self.time_off_trigger * tick_norm / (self.LOG10E * recoil_recovery_time)))
                + np.exp(-self.time_since_damage_taken * tick_norm / (self.LOG10E * self.TAGGING_RECOVERY_TIME)))

    def attack(
        self,
        height_map: np.ndarray,
        player_map: np.ndarray,
        cursor_y: float,
        players: Dict[int, 'Player'],
        recoil: bool = False
    ) -> Iterable[Event]:
        """
        Try to attack and relay any related events.

        NOTE: For simplicity, recoil is only used to produce client-side visual
        effects to simulate firing feedback with slight disorientation
        and does not affect the firing angle in any direct way,
        random or predictable.

        References:
        SlothSquadron comments on CSGO Innacuracy formulas |
            https://old.reddit.com/r/GlobalOffensive/comments/6s0ld4/csgo_inaccuracy_formulas/dl9633r/
        """

        # Fire weapon
        if self.held_object.item.slot in Item.WEAPON_SLOTS:
            self.accumulated_firing_inaccuracy += \
                self.held_object.item.firing_inaccuracy*0.1**(self.time_off_trigger/self.held_object.item.recovery_time)

            events = self.held_object.fire(
                self.pos, self.vel, self.angle, self.accumulated_firing_inaccuracy, height_map, player_map)

            # Single event means no clip, multiple means at least one attack
            if recoil and len(events) > 1:
                self.d_pos_recoil[1] += self.held_object.item.use_pos_offset
                self.d_angle_recoil += (self.rng.random()-0.5) * self.held_object.item.use_angle_var

        # Slash with knife
        elif self.held_object.item.slot == Item.SLOT_KNIFE:
            events = self.held_object.slash(self.pos, height_map, player_map, players)

            if recoil:
                self.d_pos_recoil[1] += self.held_object.item.use_pos_offset
                self.d_angle_recoil += (self.rng.random()-0.5) * self.held_object.item.use_angle_var

        # Throw grenade
        elif self.held_object.item.slot == Item.SLOT_UTILITY:
            throw_length = max(5., self.MAX_VIEW_RANGE - cursor_y)
            thrown_item_id = self.held_object.item.id

            events = self.drop(
                self.held_object, fused=True, throw_len=throw_length, throw_vel=self.GRENADE_THROW_VELOCITY)

            if events is not None:
                events.append(Event(Event.FX_ATTACK, (self.id, thrown_item_id)))

            else:
                events = Event.EMPTY_EVENT_LIST

            if recoil:
                self.d_pos_recoil[1] += self.held_object.item.use_pos_offset
                self.d_angle_recoil += (self.rng.random()-0.5) * self.held_object.item.use_angle_var

        # Other
        else:
            events = Event.EMPTY_EVENT_LIST

        return events

    def update_position_history(self, entry: Entry):
        """
        Store new state with timestamped position data
        and clear past ineligible states.
        """

        while self.states:
            if entry.timestamp - self.states[0].timestamp > self.MAX_ACCEPTABLE_LAG:
                del self.states[0]

            else:
                break

        self.states.append(entry)

    def rewind(self, target_timestamp: float, player_map: np.ndarray):
        """
        Find the position where the player was at the given time
        and accordingly update the player collider map.

        Should always be followed by a `wind_back`.
        """

        pos = None

        # Find closest timestamped position
        for entry in self.states:
            if entry.timestamp >= target_timestamp:
                pos = entry.data
                break

        # Pass if current position is the closest match
        if pos is None:
            return

        # Clear currently covered area, occupy past covered area (with consideration for other occupators)
        self.update_collider_map(
            player_map, self.pos, pos, claim_id=self.id, clear_id=Map.PLAYER_ID_NULL, check_claimed_area=True)

        self.pos = pos

    def wind_back(self, player_map: np.ndarray):
        """
        Set the player position to the last available state
        and accordingly update the player collider map.
        """

        pos = self.states[-1].data

        # Clear past occupied area (with consideration for other occupators), reclaim currently covered area
        # (passes if current position was the closest match)
        self.update_collider_map(
            player_map, self.pos, pos, claim_id=self.id, clear_id=Map.PLAYER_ID_NULL, check_cleared_area=True)

        self.pos = pos

    def buy(self, item_id: int) -> Union[Event, None]:
        """Try to buy an item with available money."""

        item = self.inventory.get_item_by_id(item_id)

        # Try to buy
        if self.money < item.price:
            return None

        self.add_money(-item.price)

        obj = self.slots[item.slot + item.subslot]

        # Assign new object
        if obj is None:
            obj = Weapon(item, self, rng=self.rng) if item.slot in Item.WEAPON_SLOTS else Object(item, self)
            self.slots[item.slot + item.subslot] = obj
            return Event(Event.OBJECT_ASSIGN, (self.id, item_id, *obj.get_values(), self.money, item.price))

        # Reset armour durability
        elif obj.item.id == GameID.ITEM_ARMOUR:
            obj.durability = obj.item.durability_cap
            return Event(Event.OBJECT_ASSIGN, (self.id, item_id, *obj.get_values(), self.money, item.price))

        # If buying over capacity, drop new object
        # If buying an item with an already occupied slot, drop new object
        elif obj.carrying == obj.item.carrying_cap or obj.item.id != item_id:
            obj = Weapon(item, self, rng=self.rng) if item.slot in Item.WEAPON_SLOTS else Object(item, self)
            return self.throw(obj, rng=True)

        # Buy an already carried item with free capacity
        else:
            obj.carrying += 1
            return Event(Event.OBJECT_ASSIGN, (self.id, item_id, *obj.get_values(), self.money, item.price))

    def pick_up(self, hovered_entity_id: int, _map: Map, objects: Dict[int, Object]) -> Union[Tuple[Event], None]:
        """
        Verify that the object seen by the client is eligible for pickup
        and handle its effects.
        """

        obj = objects.get(hovered_entity_id, None)

        # If no object in sight or object has a fuse or is out of range
        if obj is None or obj.lifetime != np.Inf or np.linalg.norm(self.pos - obj.pos) > self.MAX_PICKUP_RANGE:
            return None

        # CTs cannot pick up c4
        elif obj.item.id == GameID.ITEM_C4 and self.team == GameID.GROUP_TEAM_CT:
            return None

        # Ts cannot pick up dkits
        elif obj.item.id == GameID.ITEM_DKIT and self.team == GameID.GROUP_TEAM_T:
            return None

        # Get object in corresponding slot
        corr_obj = self.slots[obj.item.slot + obj.item.subslot]

        # Assign picked-up object to empty slot
        if corr_obj is None:
            obj.owner = self
            self.slots[obj.item.slot + obj.item.subslot] = obj
            return [
                Event(Event.OBJECT_ASSIGN, (self.id, obj.item.id, *obj.get_values(), self.money, 0)),
                Event(Event.OBJECT_EXPIRE, obj.id)]

        elif corr_obj.item.id == obj.item.id:
            # Picking up an object with cap over 1 and already carrying 1,
            # the existing object should simply have its counter incremented, without instancing anything anew
            if corr_obj.carrying < corr_obj.item.carrying_cap:
                corr_obj.carrying += 1
                return [
                    Event(Event.OBJECT_ASSIGN, (self.id, corr_obj.item.id, *corr_obj.get_values(), self.money, 0)),
                    Event(Event.OBJECT_EXPIRE, obj.id)]

            # Otherwise, the object is consumed and its resources pooled
            # (might be better to not consume the dropped object and only decrease its resources)
            elif (obj.durability > corr_obj.durability or ((obj.magazine + obj.reserve) > 0 and (
                corr_obj.reserve < corr_obj.item.reserve_cap or corr_obj.magazine < corr_obj.item.magazine_cap))
            ):
                corr_obj.durability = max(corr_obj.durability, obj.durability)
                max_magazine = max(corr_obj.magazine, obj.magazine)
                min_magazine = min(corr_obj.magazine, obj.magazine)
                corr_obj.magazine = max_magazine
                corr_obj.reserve = min(corr_obj.item.reserve_cap, corr_obj.reserve + obj.reserve + min_magazine)
                return [
                    Event(Event.OBJECT_ASSIGN, (self.id, corr_obj.item.id, *corr_obj.get_values(), self.money, 0)),
                    Event(Event.OBJECT_EXPIRE, obj.id)]

            # No pickup if the corresponding object's values are full
            else:
                return None

        # If corresponding slot is already occupied, drop the object currently occupying it and assign new object
        drop_events = self.drop(corr_obj, rng=True)
        obj.owner = self
        self.slots[obj.item.slot + obj.item.subslot] = obj

        return [
            *drop_events,
            Event(Event.OBJECT_ASSIGN, (self.id, obj.item.id, *obj.get_values(), self.money, 0)),
            Event(Event.OBJECT_EXPIRE, obj.id)]

    def draw(self, obj: Object):
        """Switch to given carried object."""

        if obj is self.held_object:
            return

        self.held_object = obj
        self.time_until_drawn = obj.item.draw_time

    def drop(
        self, obj: Object, fused: Optional[bool] = False, throw_len: float = None, throw_vel: float = None,
        rng: bool = False
    ) -> Union[List[Event], None]:
        """Try to drop the currently held object."""

        # Can't drop the knife or an empty slot
        if obj is None or obj.item.slot in Item.UNDROPPABLE_SLOTS:
            return None

        # Decrement carrying and possibly free its slot
        obj.carrying -= 1

        if not obj.carrying:
            self.slots[obj.item.slot + obj.item.subslot] = None

        # Create an object with finite lifetime
        if fused:
            if obj.item.id == GameID.ITEM_FLASH:
                dropped_obj = Flash(obj.item, self)
            elif obj.item.id == GameID.ITEM_EXPLOSIVE:
                dropped_obj = Explosive(obj.item, self)
            elif obj.item.subslot == Item.SUBSLOT_INCENDIARY:
                dropped_obj = Incendiary(obj.item, self)
            else:
                dropped_obj = Smoke(obj.item, self)

        # Create a new object if both thrown and kept, redrawing current object (this keeps other stats)
        elif obj.carrying:
            dropped_obj = Object(obj.item, self)

        # Drop given object
        else:
            dropped_obj = obj

        # Determine draw
        # If dropping currently held item with remaining capacity, redraw it
        # Otherwise, switch to knife
        if obj is self.held_object:
            self.draw(obj if obj.carrying else self.slots[Item.SLOT_KNIFE])

        # Reset carrying to 1 to indicate that it is a single, possibly pickupable, instance
        # Still, object values need to be obtained beforehand to indicate unassignment
        obj_values = obj.get_values()
        dropped_obj.carrying = 1

        return [
            Event(Event.OBJECT_ASSIGN, (self.id, obj.item.id, *obj_values, self.money, 0)),
            self.throw(dropped_obj, throw_len, throw_vel, rng)]

    def throw(
        self, obj: Object, throw_length: float = None, throw_velocity: float = None, rng: bool = False
    ) -> Event:
        """Throw an object wrt. throw strength-related parameters and own velocity."""

        throw_length = self.MAX_DROP_RANGE if throw_length is None else throw_length
        throw_velocity = self.OBJECT_DROP_VELOCITY if throw_velocity is None else throw_velocity

        # Add some randomness, so that the items don't land in the same place (e.g. when dying)
        angle = (self.angle + self.rng.random() - 0.5) if rng else self.angle

        land_pos = self.pos + np.array([np.cos(angle), np.sin(angle)]) * throw_length

        obj.throw(self.pos, land_pos, throw_velocity, init_vel=self.vel)

        return Event(Event.OBJECT_SPAWN, obj)

    def reload(self) -> Union[Event, None]:
        """Try to start a reloading sequence."""

        if self.time_to_reload or self.time_until_drawn:
            return None

        missing_mag = self.held_object.item.magazine_cap - self.held_object.magazine
        available_res = self.held_object.reserve

        if not (available_res and missing_mag):
            return None

        self.time_to_reload = self.held_object.item.reload_time
        self.reload_events.clear()
        self.reload_events.extend(self.held_object.item.reload_events)

        return Event(Event.PLAYER_RELOAD, (self.id, Item.RLD_START, self.held_object.item.id))

    def eval_reload(self, cancel: bool) -> Union[Event, None]:
        """
        Advance or stop the reloading sequence, i.e. a queue of reloading
        events that evaluate at specific times, returning at most one
        reloading event at a time.
        """

        # Check time threshold of upcoming reload event
        if self.reload_events and (self.held_object.item.reload_time - self.time_to_reload) >= self.reload_events[0][0]:
            reload_event_type = self.reload_events.popleft()[1]

            if reload_event_type == Item.RLD_ADD:
                missing_mag = self.held_object.item.magazine_cap - self.held_object.magazine

                # Add ammo
                if missing_mag:
                    available_res = self.held_object.reserve
                    used_res = min(missing_mag, available_res, self.held_object.item.ammo_per_restore)

                    self.held_object.magazine += used_res
                    self.held_object.reserve -= used_res

                    return Event(Event.PLAYER_RELOAD, (self.id, Item.RLD_ADD, self.held_object.item.id))

                # Fast-forward to reload end event
                else:
                    while self.reload_events:
                        if self.reload_events[0][1] != Item.RLD_END:
                            evtime = self.reload_events.popleft()[0]
                            self.time_to_reload = self.held_object.item.reload_time - evtime

                        else:
                            break

            else:
                return Event(Event.PLAYER_RELOAD, (self.id, Item.RLD_END, self.held_object.item.id))

        # Cancel on held object drop or switch
        elif cancel:
            self.time_to_reload = 0.

        return None
