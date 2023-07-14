"""Extendable live client for SDG"""

from argparse import Namespace
import struct
import random
from typing import Iterable, List, Tuple, Union

from numpy.random import default_rng

from sidegame.physics import fix_angle_range, update_collider_map, F_PI, F_2PI
from sidegame.effects import Colour, Mark, Explosion, Flame, Fog, Gunfire, Decal, Residual
from sidegame.networking import Entry, Action, Entity, LiveClient
from sidegame.game.shared import GameID, Map, Event, Message, Item, Object, Weapon, Incendiary, Smoke, Player, Session
from sidegame.game.client.simulation import Simulation
from sidegame.game.client.tracking import StatTracker


class SDGLiveClientBase(LiveClient):
    """
    A client for live communication with the SDG server.
    It leaves some of the underlying abstract methods unimplemented,
    so that the interface can be tailored to specific IO schemes.
    """

    CLIENT_MESSAGE_SIZE = 32
    SERVER_MESSAGE_SIZE = 64
    ENV_ID = Map.PLAYER_ID_NULL

    def __init__(self, args: Namespace):
        self.mmr = args.mmr
        self.name = args.name
        self.role_key: str = '0x' + args.role_key
        self.session_address = args.address
        self.rng = default_rng(args.seed)
        self.time_scale = args.time_scale

        super().__init__(
            tick_rate=args.tick_rate,
            polling_rate=args.polling_rate,
            sending_rate=args.sending_rate,
            server_address=(args.address, args.port),
            client_message_size=self.CLIENT_MESSAGE_SIZE,
            server_message_size=self.SERVER_MESSAGE_SIZE,
            matchmaking=bool(args.mmr),
            recording_path=args.recording_path,
            logging_path=args.logging_path,
            logging_level=args.logging_level,
            show_fps=args.show_fps,
            interp_ratio=args.interp_ratio
        )

        self.sim = Simulation(self.own_entity_id, args.tick_rate, args.volume, args.audio_device, rng=self.rng)
        self.session: Session = self.sim.session
        self.stats = StatTracker(self.session, self.own_entity) if args.track_stats else None
        random.seed(args.seed)

    def create_entity(self, entity_id: int) -> Entity:
        """
        Produce a local entity instance and prepare it
        for integration into the local state.

        Placeholders should be overriden in event handling,
        either in `OBJECT_SPAWN` or `CTRL_PLAYER_CONNECTED`
        """

        # NOTE: Object entities are distinguished by negative IDs
        if entity_id < 0:
            placeholder_item = Item()
            placeholder_owner = self
            new_entity = Object(placeholder_item, placeholder_owner, entity_id=entity_id)
        else:
            # NOTE: The inventory must be a placeholder, because the first call to `create_entity`
            # is made before `sim` (`sim.inventory`) is initialised
            placeholder_inventory = None
            new_entity = Player(entity_id, placeholder_inventory, rng=self.rng)

        return new_entity

    def update_own_entity(self, state_entry: Entry, timestamp: float):
        if not self.session.phase:
            return

        player: Player = self.own_entity

        # Overwrite state data, ignoring server-side recoil
        old_pos = player.pos.copy()

        player.pos[0], player.pos[1], player.vel[0], player.vel[1], player.angle, _, _, _, \
            player.time_since_damage_taken, time_until_drawn, time_to_reload, time_off_trigger, \
            held_object_id, held_object_magazine, held_object_reserve = state_entry.data

        if player.team != GameID.GROUP_SPECTATORS and player.health:
            update_collider_map(
                player.covered_indices, self.session.map.player_id, old_pos, player.pos, player.id, Map.PLAYER_ID_NULL)

        if self.stats is not None:
            self.stats.update_from_state(
                player.pos.copy() if player.pos is self.stats.last_pos else player.pos, timestamp, self.time_scale)

        # (Re)set held item
        if player.held_object is not None and held_object_id and held_object_id != player.held_object.item.id:
            held_object = player.inventory.get_item_by_id(held_object_id)
            carried_object = player.slots[held_object.slot + held_object.subslot]

            # Handle case when obj assign (drop) event would go through before state update
            # State update could then reference an item that has already been dropped, its state now occupied by `None`
            if carried_object is None:
                return

            player.held_object = carried_object

        if player.held_object is not None:
            player.held_object.reserve = held_object_reserve

            # Handle cases where client prediction seems ahead or behind due to tick/lag incidence
            time_diff = abs(state_entry.timestamp - (timestamp + self._clock_diff_tracker.value))
            values_are_close = abs(time_off_trigger - player.time_off_trigger) < time_diff

            client_side_triggered = time_off_trigger > (player.held_object.item.use_interval - time_diff) \
                and player.time_off_trigger < time_diff

            server_side_triggered = player.time_off_trigger > (player.held_object.item.use_interval - time_diff) \
                and time_off_trigger < time_diff

            if not (values_are_close or client_side_triggered or server_side_triggered):
                player.held_object.magazine = held_object_magazine
                player.time_until_drawn = time_until_drawn
                player.time_to_reload = time_to_reload
                player.time_off_trigger = time_off_trigger

            # NOTE: The other thing that can cause firing sound inconsistencies is that inputs can only be sent
            # and processed at discrete intervals, which don't necessarily match up nicely with specific use intervals,
            # so they will be suboptimally utilised, e.g. can't catch 0.15s use interval if tick is at 0.133s or 0.167s

    def handle_log(self, event_entry: Entry, timestamp: float):
        sim = self.sim
        session = self.session
        inventory = sim.inventory
        own_player = self.own_entity
        observed_player = session.players.get(sim.observed_player_id, own_player)
        queue_sound = sim.audio_system.queue_sound

        event_data: List[float, int] = event_entry.data
        event_id: int = event_data[-1]

        # When reconnecting, don't replay all effects (rough window with 5s grace period)
        accept_announced_fx = abs(timestamp + self._clock_diff_tracker.value - event_entry.timestamp) < 5.
        accept_experienced_fx = accept_announced_fx and observed_player.health

        # Handle game (env.) event
        if event_entry.id == self.ENV_ID:
            if event_id == Event.CTRL_MATCH_STARTED:
                map_id = event_data[-2]

                self.remove_object_entities()
                session.start_match(map_id=map_id, assign_c4=False)

                # NOTE: Spectators must enter manually (ESC), because it was a bit annoying when connecting mid-game
                if session.is_player(sim.own_player_id):
                    sim.enter_world()

            elif event_id == Event.CTRL_MATCH_ENDED:
                _, _, session.rounds_won_t, session.rounds_won_ct = event_data[-5:-1]
                session.stop_match()
                sim.exit_world()

            elif event_id == Event.CTRL_MATCH_PHASE_CHANGED:
                old_phase = session.phase
                new_phase = int(event_data[6])

                # Distribute rewards
                if new_phase == GameID.PHASE_RESET:
                    t_win = bool(event_data[0])
                    penalise_alive_ts = bool(event_data[1])
                    win_reward = int(event_data[2])

                    session.change_phase(
                        new_phase, t_win=t_win, win_reward=win_reward, penalise_alive_ts=penalise_alive_ts)

                    # Announcer sound
                    if accept_announced_fx:
                        queue_sound(sim.sounds['t_win' if t_win else 'ct_win'], own_player, own_player)

                    # Add chat entry
                    winning_team = GameID.GROUP_TEAM_T if t_win else GameID.GROUP_TEAM_CT
                    losing_team = GameID.GROUP_TEAM_CT if t_win else GameID.GROUP_TEAM_T

                    sim.add_chat_entry(Message(
                        GameID.NULL,
                        session.rounds_won_t + session.rounds_won_ct,
                        session.total_round_time,
                        [winning_team, GameID.TERM_KILL, losing_team, GameID.TERM_STOP]))

                # Reset side and/or round
                elif old_phase == GameID.PHASE_RESET and new_phase == GameID.PHASE_BUY:
                    self.remove_object_entities()
                    sim.effects.clear()
                    sim.fx_map.fill(0)

                    # Switch back to own player if viewing others when dead
                    if session.is_dead_player(sim.own_player_id):
                        sim.observed_player_id = sim.own_player_id

                    if (session.rounds_won_t + session.rounds_won_ct) == session.ROUNDS_TO_SWITCH:
                        session.reset_side(switch_sides=True)

                        # Announcer sound
                        if accept_announced_fx:
                            queue_sound(sim.sounds['reset_side'], own_player, own_player)

                        # Add chat entry
                        sender_id = GameID.NULL
                        msg_round = session.rounds_won_t + session.rounds_won_ct
                        msg_time = session.total_round_time

                        words = [GameID.GROUP_TEAM_T, GameID.TERM_MOVE, GameID.GROUP_TEAM_CT, GameID.TERM_STOP]
                        sim.add_chat_entry(Message(sender_id, msg_round, msg_time, words))

                        words = [GameID.GROUP_TEAM_CT, GameID.TERM_MOVE, GameID.GROUP_TEAM_T, GameID.TERM_STOP]
                        sim.add_chat_entry(Message(sender_id, msg_round, msg_time, words))

                    session.reset_round(assign_c4=False)

                    # Announcer sound
                    if accept_announced_fx:
                        queue_sound(sim.sounds['reset_round'], own_player, own_player)

                # Sync current match state
                session.loss_streak_t = int(event_data[3])
                session.loss_streak_ct = int(event_data[4])
                session.time = event_data[5]
                session.phase = int(event_data[6])
                session.rounds_won_t = event_data[-3]
                session.rounds_won_ct = event_data[-2]

            # Break client loop
            elif event_id == Event.CTRL_SESSION_ENDED:
                self.session_running = False

            # Handle player appeareance by adding them to the associated group
            # NOTE: If the player has disconnected later, the initial connection will still be acknowledged
            # to properly replay the events up to their disconnection
            elif event_id == Event.CTRL_PLAYER_CONNECTED:
                player_id = int(event_data[3])
                name = ''.join(chr(int(ordinal)) for ordinal in event_data[4:8])

                # Assign to entities if log was received before any state data, otherwise remove placeholder inventory
                if player_id in self.entities:
                    self.entities[player_id].inventory = inventory
                else:
                    self.entities[player_id] = Player(player_id, inventory, rng=session.rng)

                session.add_player(self.entities[player_id])
                self.entities[player_id].name = name

                # On connecting after match has started, need to sync game time
                if player_id == sim.own_player_id and session.phase:
                    phase_time, round_time, match_time = event_data[0:3]
                    session.time = phase_time
                    session.total_round_time = round_time
                    session.total_match_time = match_time

            # Handle player disappearance by removing them from the associated group
            elif event_id == Event.CTRL_PLAYER_DISCONNECTED:
                player_id = int(event_data[0])

                dc_player: Player = self.entities[player_id]

                if session.phase:
                    session.map.player_id[dc_player.get_covered_indices()] = Map.PLAYER_ID_NULL

                session.remove_player(dc_player)
                del self.entities[player_id]

                # Handle case where observed player disconnects
                if player_id == sim.observed_player_id:
                    sim.change_observed_player()

            # Handle team (re)assignment
            elif event_id == Event.CTRL_PLAYER_MOVED:
                player_id = int(event_data[0])
                team = event_data[-3]
                position_id = event_data[-2]

                session.move_player(player_id, team)

                # Has no effect if match is not running
                # If moving from spectator to player, reset observed player to own player
                # If moving from player to spectator, reset observed player to some other player
                # If no other players left, reset to lobby
                if player_id == sim.own_player_id:
                    sim.enter_world()

                # If other observed entity has been moved, select the next one from the available pool
                elif player_id == sim.observed_player_id:
                    sim.change_observed_player()

                # NOTE: Debugging; with proper unfolding of events, `position_id` is needlessly supplied
                assert position_id == session.players[player_id].position_id

            # Handle renaming, cheats, etc.
            elif event_id == Event.CTRL_PLAYER_CHANGED:
                player_id = int(event_data[0])
                name = ''.join(chr(int(ordinal)) for ordinal in event_data[1:5])
                money = int(event_data[5])
                dev_mode = bool(event_data[6])

                player: Player = session.players[player_id]
                player.name = name
                player.money = money
                player.dev_mode = dev_mode

            elif event_id == Event.CTRL_LATENCY_REQUESTED:
                requestor_id = event_data[10]

                if sim.own_player_id == requestor_id:
                    n_players = event_data[-2]
                    print_string = '\n\nPlayer round-trip latencies:\n'

                    for i in range(n_players):
                        a_player_id, a_player_latency, _, _ = struct.unpack('>4B', struct.pack('>f', event_data[i]))
                        a_player_name = session.players[a_player_id].name
                        print_string += f'{a_player_id:3d} | {a_player_name}: {a_player_latency:3d}ms\n'

                    print(print_string)

            # Handle object spawning as interpolated entities
            elif event_id == Event.OBJECT_SPAWN:
                obj_id = int(event_data[0])
                obj_owner_id = int(event_data[1])
                obj_lifetime = event_data[2]
                obj_item_id = event_data[-2]

                item = inventory.get_item_by_id(obj_item_id)
                owner = session.players.get(obj_owner_id, None)

                if owner is None:
                    owner = Player(Map.PLAYER_ID_NULL, sim.inventory, rng=self.rng)

                # NOTE: Incendiary/smoke need cover update methods, so replacing placeholders is not enough
                # Their lifetime also needs to be overridden to allow dropped objects to be hovered
                if obj_item_id == GameID.ITEM_INCENDIARY_T or obj_item_id == GameID.ITEM_INCENDIARY_CT:
                    obj = Incendiary(item, owner)
                    obj.lifetime = obj_lifetime
                elif obj_item_id == GameID.ITEM_SMOKE:
                    obj = Smoke(item, owner)
                    obj.lifetime = obj_lifetime
                else:
                    obj = Object(item, owner, lifetime=obj_lifetime)

                # NOTE: If the object has expired later, the initial spawn will still be acknowledged
                # to properly replay the events up to their expiry
                if obj_id in self.entities:
                    obj_prev: Object = self.entities[obj_id]
                    obj.states.extend(obj_prev.states)

                obj.id = obj_id
                self.entities[obj_id] = obj
                session.objects[obj_id] = obj

            # Interpolated object expiry is akin to entity "disconnection"
            elif event_id == Event.OBJECT_EXPIRE:
                obj_id = int(event_data[0])
                obj = session.objects[obj_id]

                session.map.object_id[obj.get_position_indices(obj.pos)] = Map.OBJECT_ID_NULL

                # NOTE: Due to `accept_experienced_fx`, cover indices can remain unset up to this point
                if Item.SUBSLOT_INCENDIARY <= obj.item.subslot <= Item.SUBSLOT_SMOKE and obj.cover_indices is not None:
                    obj.clear_zone_cover(session.map.zone, session.map.zone_id)

                del session.objects[obj_id]
                del self.entities[obj_id]

            # Add carried object, remove it, or (re)set its values
            elif event_id == Event.OBJECT_ASSIGN:
                obj_owner_id = int(event_data[0])
                obj_item_id = event_data[-2]

                durability = event_data[1]
                magazine = int(event_data[2])
                reserve = int(event_data[3])
                carrying = int(event_data[4])
                money = int(event_data[5])
                spending = int(event_data[6])

                player = session.players[obj_owner_id]
                player.money = money
                item = inventory.get_item_by_id(obj_item_id)
                full_item_slot = item.slot + item.subslot

                if player.slots[full_item_slot] is None and carrying:
                    player.slots[full_item_slot] = \
                        Weapon(item, player, rng=player.rng) if item.slot in Item.WEAPON_SLOTS else Object(item, player)

                elif not carrying and not spending:
                    player.slots[full_item_slot] = None

                if player.slots[full_item_slot] is not None and carrying:
                    player.slots[full_item_slot].set_values(durability, magazine, reserve, carrying)

                # SFX
                if accept_experienced_fx:
                    queue_sound(sim.sounds['get' if carrying else 'drop'], observed_player, player)

            elif event_id == Event.OBJECT_TRIGGER and accept_experienced_fx:
                obj_id = int(event_data[0])

                # SFX
                source = session.objects[obj_id]
                sound = inventory.get_item_by_id(source.item.id).sounds['detonate']
                queue_sound(sound, observed_player, source)

                # Handle VFX (update covered indices and zone map)
                # NOTE: Zone map is set according to current zone map state, but not updated retroactively,
                # i.e. old incendiaries will be fully displayed (ideally below) the new effect
                item_id = source.item.id

                if item_id == GameID.ITEM_FLASH:
                    vfx = Explosion(source.pos, source.item.radius, 0.2)

                elif item_id == GameID.ITEM_EXPLOSIVE:
                    vfx = Explosion(source.pos, source.item.radius/2., 0.5)

                elif item_id == GameID.ITEM_SMOKE:
                    vfx = Fog((0, 0), source.item.radius, source.item.duration)

                    source.set_zone_cover(session.map.wall, session.map.zone, session.map.zone_id)
                    vfx.world_indices = vfx.cover_indices = source.cover_indices

                else:
                    vfx = Flame((0, 0), source.item.radius, source.item.duration)

                    source.set_zone_cover(session.map.wall, session.map.zone, session.map.zone_id)
                    vfx.world_indices = vfx.cover_indices = source.cover_indices

                sim.add_effect(vfx)

            elif event_id == Event.C4_DETONATED and accept_announced_fx:
                obj_id = int(event_data[0])
                source = session.objects[obj_id]

                queue_sound(inventory.c4.sounds['explode'], observed_player, source)
                sim.add_effect(Explosion(source.pos, source.item.radius/2., 3.))

            elif event_id == Event.C4_PLANTED:
                planter_id = int(event_data[0])

                planter: Player = session.players[planter_id]
                planter.money += 300

                # Announcer sound
                if accept_announced_fx:
                    queue_sound(sim.sounds['planted'], own_player, own_player)

                    # Object sound
                    if accept_experienced_fx:
                        queue_sound(inventory.c4.sounds['plant'], observed_player, planter)

                # Add chat entry
                sim.add_chat_entry(Message(
                    GameID.NULL,
                    session.rounds_won_t + session.rounds_won_ct,
                    session.total_round_time,
                    [planter.position_id, GameID.TERM_HOLD, GameID.ITEM_C4, GameID.TERM_STOP]))

            elif event_id == Event.C4_DEFUSED:
                obj_id = int(event_data[0])
                defuser_id = int(event_data[1])

                defuser: Player = session.players[defuser_id]
                defuser.money += 300
                session.distribute_rewards(session.players_t.values(), [], 800)

                # Announcer sound
                if accept_announced_fx:
                    queue_sound(sim.sounds['defused'], own_player, own_player)

                    # Object sound
                    if accept_experienced_fx:
                        queue_sound(inventory.c4.sounds['disarmed'], observed_player, defuser)

                # Add chat entry
                sim.add_chat_entry(Message(
                    GameID.NULL,
                    session.rounds_won_t + session.rounds_won_ct,
                    session.total_round_time,
                    [defuser.position_id, GameID.ITEM_DKIT, GameID.ITEM_C4, GameID.TERM_STOP]))

            elif event_id == Event.FX_C4_TOUCHED and accept_experienced_fx:
                obj_id = int(event_data[0])

                queue_sound(inventory.c4.sounds['disarming'], observed_player, session.objects[obj_id])

            elif event_id == Event.FX_FOOTSTEP and accept_experienced_fx:
                player_id = int(event_data[0])
                player = session.players[player_id]

                pos_y, pos_x = player.get_position_indices(player.pos)
                terrain_key = int(session.map.sound[pos_y, pos_x])

                # Terrain sound
                sound = random.choice(sim.footsteps[terrain_key])
                queue_sound(sound, observed_player, player)

                # Movement sound
                sound = random.choice(sim.movements)
                queue_sound(sound, observed_player, player)

            elif event_id == Event.FX_C4_KEY_PRESS and accept_experienced_fx:
                planter_id = int(event_data[0])

                sound = random.choice(sim.keypresses)
                queue_sound(sound, observed_player, session.players[planter_id])

            elif event_id == Event.FX_C4_INIT and accept_experienced_fx:
                obj_owner_id = int(event_data[0])

                queue_sound(inventory.c4.sounds['init'], observed_player, session.players[obj_owner_id])

            elif event_id == Event.FX_C4_BEEP and accept_experienced_fx:
                obj_id = int(event_data[0])
                obj = session.objects[obj_id]
                queue_sound(inventory.c4.sounds['beep_a'], observed_player, obj)
                sim.add_effect(Colour(Colour.get_disk_indices(3), sim.COLOUR_RED, 0.1, obj.pos[1], obj.pos[0], 0.4))

            elif event_id == Event.FX_C4_BEEP_DEFUSING and accept_experienced_fx:
                obj_id = int(event_data[0])
                obj = session.objects[obj_id]

                # NOTE: Universally different sound for `BEEP_DEFUSING` would make it impossible to bluff
                # It's only meant to inform the defuser and their team, anyway (and spectators)
                if observed_player.team == GameID.GROUP_TEAM_T:
                    sound = inventory.c4.sounds['beep_a']
                    sim.add_effect(
                        Colour(Colour.get_disk_indices(3), sim.COLOUR_RED, 0.1, obj.pos[1], obj.pos[0], 0.4))
                else:
                    sound = inventory.c4.sounds['beep_b']
                    sim.add_effect(
                        Colour(Colour.get_disk_indices(3), sim.COLOUR_WHITE, 0.1, obj.pos[1], obj.pos[0], 0.4))

                queue_sound(sound, observed_player, obj)

            elif event_id == Event.FX_BOUNCE and accept_experienced_fx:
                obj_id = int(event_data[0])
                source = session.objects[obj_id]

                # NOTE: Dropped objects can land/bounce, but don't have dedicated sfx of their own
                sound = inventory.get_item_by_id(source.item.id).sounds.get('bounce', inventory.flash.sounds['bounce'])
                queue_sound(sound, observed_player, source)

            elif event_id == Event.FX_LAND and accept_experienced_fx:
                obj_id = int(event_data[0])
                source = session.objects[obj_id]

                sound = inventory.get_item_by_id(source.item.id).sounds.get('land', inventory.flash.sounds['land'])
                queue_sound(sound, observed_player, source)

            elif event_id == Event.FX_C4_NVG and accept_announced_fx:
                obj_id = int(event_data[0])

                queue_sound(inventory.c4.sounds['nvg'], observed_player, session.objects[obj_id])

            elif event_id == Event.FX_ATTACK and accept_experienced_fx:
                attacker_id = int(event_data[0])
                item_id = event_data[-2]
                item = inventory.get_item_by_id(item_id)

                # NOTE: Own player wall hits are already predicted and grenade throw SFX is handled by drop (assign)
                if attacker_id != sim.own_player_id and item.slot != Item.SLOT_UTILITY:
                    source = session.players[attacker_id]
                    queue_sound(item.sounds['attack'], observed_player, source, override=True)

                    # Add gunfire
                    if item.slot in Item.WEAPON_SLOTS:
                        sim.add_effect(Gunfire(source.pos, source.angle, item.flash_level, item.use_interval))

            elif event_id == Event.FX_CLIP_LOW and accept_experienced_fx:
                attacker_id = int(event_data[0])

                queue_sound(sim.sounds['clip_low'], observed_player, session.players[attacker_id])

            elif event_id == Event.FX_CLIP_EMPTY and accept_experienced_fx:
                attacker_id = int(event_data[0])

                queue_sound(sim.sounds['clip_empty'], observed_player, session.players[attacker_id])

            elif event_id == Event.FX_EXTINGUISH and accept_experienced_fx:
                obj_id = int(event_data[0])

                source = session.objects[obj_id]
                queue_sound(inventory.get_item_by_id(source.item.id).sounds['extinguish'], observed_player, source)

            elif event_id == Event.FX_FLASH and accept_experienced_fx:
                flashed_id = int(event_data[1])
                debuff = event_data[2]
                debuff_duration = event_data[3]

                # NOTE: Not `sim.observed_player_id`
                if flashed_id == sim.own_player_id:
                    # Residual image effect
                    if sim.last_world_frame is not None:
                        sim.add_effect(Residual(sim.last_world_frame, debuff_duration, flash=debuff, opacity=debuff))

                    # SFX
                    if debuff > 0.1:
                        if debuff > 0.7:
                            sound = sim.sounds['sine_max']
                        elif debuff > 0.4:
                            sound = sim.sounds['sine_mid']
                        elif debuff >= 0.1:
                            sound = sim.sounds['sine_min']

                        queue_sound(sound, observed_player, observed_player)

            elif event_id == Event.FX_WALL_HIT and accept_experienced_fx:
                pos_x = event_data[0]
                pos_y = event_data[1]
                attacker_id = int(event_data[2])
                item_id = int(event_data[-2])

                # NOTE: Own player wall hits are already predicted
                if attacker_id != sim.own_player_id:
                    sim.add_effect(Decal(pos_y, pos_x, lifetime=10.))

                    if item_id == GameID.ITEM_KNIFE:
                        queue_sound(inventory.knife.sounds['front_hit'], observed_player, session.players[attacker_id])

            elif event_id == Event.PLAYER_RELOAD:
                reloader_id = int(event_data[0])
                rld_event_type = event_data[-3]
                rld_item_id = event_data[-2]

                # NOTE: Own reload events are already predicted
                if reloader_id != sim.own_player_id and accept_experienced_fx:
                    if rld_event_type == Item.RLD_START:
                        sound_key = 'reload_start'
                    elif rld_event_type == Item.RLD_ADD:
                        sound_key = 'reload_add'
                    elif rld_event_type == Item.RLD_END:
                        sound_key = 'reload_end'
                    else:
                        sound_key = 'draw'

                    sound = inventory.get_item_by_id(rld_item_id).get_sound(sound_key)

                    if sound is not None:
                        queue_sound(sound, observed_player, session.players[reloader_id])

            elif event_id == Event.PLAYER_DAMAGE:
                attacker_id = int(event_data[0])
                damaged_id = int(event_data[1])
                damage = event_data[2]
                item_id = event_data[-2]

                dmg_event = Event(event_id, (attacker_id, damaged_id, item_id, damage))
                player = session.players[damaged_id]

                # NOTE: Foreign entity recoil is interpolated instead
                recoil = player.id == sim.own_player_id
                player.eval_damage(dmg_event, session.map, check_death=True, recoil=recoil, players=session.players)

                if accept_experienced_fx:
                    # Hit sound
                    queue_sound(sim.sounds['hit'], observed_player, player)

                    # Add blood splatter
                    pos_x, pos_y = player.pos
                    sim.add_effect(Decal(pos_y, pos_x, colour=Decal.COLOUR_RED, lifetime=20.))

                    if item_id == GameID.ITEM_KNIFE:
                        knife = inventory.knife
                        sound = knife.sounds['front_hit' if damage <= knife.base_damage else 'back_hit']
                        queue_sound(sound, observed_player, player)

            elif event_id == Event.PLAYER_DEATH:
                attacker_id = int(event_data[0])
                victim_id = int(event_data[1])
                excess = event_data[2]
                item_id = event_data[-2]

                death_event = Event(event_id, (attacker_id, victim_id, item_id, excess))

                player = session.players[victim_id]
                player.eval_death(death_event, session.map)
                session.handle_player_death(death_event)

                # Death sound
                if accept_experienced_fx or (victim_id == sim.observed_player_id and accept_announced_fx):
                    queue_sound(sim.sounds['death'], observed_player, player)

                # C4 deaths don't give rewards or recognition
                attacker_pos_id = \
                    session.players[attacker_id].position_id if item_id != GameID.ITEM_C4 else GameID.ITEM_C4

                # Add chat entry
                sim.add_chat_entry(Message(
                    GameID.NULL,
                    session.rounds_won_t + session.rounds_won_ct+1,
                    session.total_round_time,
                    [attacker_pos_id, GameID.TERM_KILL, player.position_id, GameID.TERM_STOP]))

                if victim_id == sim.own_player_id:
                    sim.observer_lock_time = 0.5

        # Update chat history with player message
        elif session.check_player_message_access(sim.own_player_id, event_entry.id):
            msg_time = event_data[8]
            words = [int(event_data[i]) for i in range(9, 13)]
            msg_round = event_data[-2]
            sender_position_id = event_data[-1]
            sender_id = event_entry.id

            # Add chat entry
            sim.add_chat_entry(Message(sender_position_id, msg_round, msg_time, words, sender_id=sender_id))

            if accept_announced_fx:
                # Distinguish between normal messages and marks
                mark_included = False

                # Spawn ping mark effects
                for i, word in enumerate(words):
                    if GameID.MARK_T1 <= word <= GameID.MARK_CT5:
                        pos_x, pos_y = event_data[2*i], event_data[2*i+1]

                        sim.add_effect(Mark(pos_y, pos_x, associated_id=word, lifetime=10.))
                        mark_included = True

                # Add SFX
                queue_sound(sim.sounds['mark_received' if mark_included else 'msg_received'], own_player, own_player)

        if self.stats is not None:
            self.stats.update_from_event(event_entry.id, event_id, event_data, timestamp)

    def remove_object_entities(self):
        """Remove client-side-only (object) entities from general entities."""
        for entity_id in self.session.objects:
            if entity_id in self.entities:
                del self.entities[entity_id]

    def predict_state(self, action: Action):
        player: Player = self.own_entity
        session = self.session
        sim = self.sim

        # Unpack action
        attack, walking, _, _, drop, rld, draw_id, _, cursor_y, force_w, force_d, _, view, _, _, d_angle = action.data
        dt = action.dt * self.time_scale

        # Increment game time
        if session.phase and not action.processed:
            session.time += dt
            session.total_round_time += dt
            session.total_match_time += dt

        # Prediction for own entity when observed and alive
        if session.phase and player.team != GameID.GROUP_SPECTATORS and player.health:
            player.decay(dt, recoil=(not action.processed))

            if player.time_to_reload and not action.processed:
                rld_event = player.eval_reload(drop or draw_id)

                if rld_event is not None:
                    sound = player.held_object.item.get_sound(
                        'reload_end' if rld_event.data[1] == Item.RLD_END else 'reload_add')

                    if sound is not None:
                        sim.audio_system.queue_sound(sound, player, player)

            # In buy phase, prevent movement and firing
            if session.phase == GameID.PHASE_BUY and not player.dev_mode:
                force_w = 0
                force_d = 0
                attack = 0

            # NOTE: Messaged movement forces are strictly positive (B format) and need to be recentred
            else:
                force_w = force_w - 1
                force_d = force_d - 1

            # Max velocity determined by currently drawn item and recently taken damage
            max_v = player.held_object.item.velocity_cap if not walking else (player.held_object.item.velocity_cap / 2.)
            max_v *= player.get_tagging_factor()

            # Update position and movement state
            old_pos = player.pos

            player.move(dt, force_w, force_d, d_angle, walking, max_v, session.map.height, session.map.player_id)

            update_collider_map(
                player.covered_indices, session.map.player_id, old_pos, player.pos, player.id, Map.PLAYER_ID_NULL)

            # Execute draw
            if draw_id:
                item = player.inventory.get_item_by_id(draw_id)
                obj = player.slots[item.slot + item.subslot]

                if obj is not None and obj is not player.held_object and not action.processed:
                    sim.audio_system.queue_sound(item.sounds['draw'], player, player)

                player.draw(obj)

            # No prediction is done for planting / defusing or picking / dropping stuff up,
            # i.e. these things are conveyed via lagged server state updates
            # (no particular benefit of prediction here, only harder reconciliation)
            if rld:
                rld_event = player.reload()

                if rld_event is not None and not action.processed:
                    sound = player.held_object.item.get_sound('reload_start')

                    if sound is not None:
                        sim.audio_system.queue_sound(sound, player, player)

            # Fire
            if attack and view == GameID.VIEW_WORLD:
                can_attack = not (
                    player.time_until_drawn or player.time_to_reload or
                    player.time_off_trigger < player.held_object.item.use_interval)

                if can_attack:
                    # NOTE: Prediction should not drop items, otherwise there can be a mismatch with the server
                    # that might not be reconciled until death
                    if player.held_object.item.slot == Item.SLOT_UTILITY:
                        events = [Event(Event.FX_ATTACK, (player.id, player.held_object.item.id))]
                        player.d_pos_recoil[1] += player.held_object.item.use_pos_offset
                        player.d_angle_recoil += player.rng.normal(0., player.held_object.item.use_angle_std)

                    else:
                        events = player.attack(
                            session.map.height, session.map.player_id, cursor_y, session.players, recoil=True)

                    player.time_off_trigger = 0.

                    if not action.processed:
                        knife_hit_processed = False

                        for event in events:
                            if event.type == Event.FX_ATTACK:
                                item_id = event.data[1]
                                item = player.inventory.get_item_by_id(item_id)

                                # Throws already handled by drop (assign)
                                if item.slot != Item.SLOT_UTILITY:
                                    sim.audio_system.queue_sound(item.sounds['attack'], player, player, override=True)

                                    # Add muzzle flash
                                    if item.slot in Item.WEAPON_SLOTS:
                                        sim.add_effect(
                                            Gunfire(player.pos, player.angle, item.flash_level, item.use_interval))

                            elif event.type == Event.FX_WALL_HIT:
                                pos_x, pos_y = event.data[0]
                                item_id = event.data[2]
                                item = player.inventory.get_item_by_id(item_id)

                                sim.add_effect(Decal(pos_y, pos_x, lifetime=10.))

                                if item_id == GameID.ITEM_KNIFE and not knife_hit_processed:
                                    knife_hit_processed = True
                                    sim.audio_system.queue_sound(item.sounds['front_hit'], player, player)

        if not action.processed:
            action.processed = True

    def interpolate_foreign_entity(self, entity: Entity, state_ratio: float, state1: Entry, state2: Entry):
        if not self.session.phase:
            return

        # NOTE: Negative entity IDs denote object entities
        if entity.id < 0:
            if entity.id not in self.session.objects:
                return

            obj: Object = entity

            pos_x_1, pos_y_1 = state1.data[:2]
            pos_x_2, pos_y_2 = state2.data[:2]

            # Interpolate movement
            obj.pos[0] = state_ratio * pos_x_2 + (1. - state_ratio) * pos_x_1
            obj.pos[1] = state_ratio * pos_y_2 + (1. - state_ratio) * pos_y_1

        else:
            player: Player = entity

            if entity.id not in self.session.players or player.team == GameID.GROUP_SPECTATORS or not player.health:
                return

            old_pos = player.pos.copy()

            pos_x_1, pos_y_1, angle_1 = state1.data[0], state1.data[1], state1.data[4]
            r_angle_1, r_pos_x_1, r_pos_y_1 = state1.data[5], state1.data[6], state1.data[7]

            pos_x_2, pos_y_2, _, _, angle_2, r_angle_2, r_pos_x_2, r_pos_y_2, _, _, _, _, \
                held_object_id, held_object_magazine, held_object_reserve = state2.data

            # Interpolate position
            player.pos[0] = state_ratio * pos_x_2 + (1. - state_ratio) * pos_x_1
            player.pos[1] = state_ratio * pos_y_2 + (1. - state_ratio) * pos_y_1

            update_collider_map(
                player.covered_indices, self.session.map.player_id, old_pos, player.pos, player.id, Map.PLAYER_ID_NULL)

            # When crossing -pi/pi, the negative angle needs to be brought into the positive range (or vice versa),
            # so that convex combination can be performed
            if angle_1*angle_2 < 0. and abs(angle_1 - angle_2) > F_PI:
                if angle_1 < 0.:
                    angle_1 += F_2PI
                else:
                    angle_2 += F_2PI

            angle = state_ratio * angle_2 + (1. - state_ratio) * angle_1
            player.angle = fix_angle_range(angle)

            # Interpolate recoil
            player.d_pos_recoil[0] = state_ratio * r_pos_x_2 + (1. - state_ratio) * r_pos_x_1
            player.d_pos_recoil[1] = state_ratio * r_pos_y_2 + (1. - state_ratio) * r_pos_y_1

            r_angle = state_ratio * r_angle_2 + (1. - state_ratio) * r_angle_1
            player.d_angle_recoil = fix_angle_range(r_angle)

            # (Re)set held item
            if player.held_object is not None and held_object_id and held_object_id != player.held_object.item.id:
                held_object = player.inventory.get_item_by_id(held_object_id)
                carried_object = player.slots[held_object.slot + held_object.subslot]

                if carried_object is None:
                    return

                player.held_object = carried_object
                player.held_object.magazine = held_object_magazine
                player.held_object.reserve = held_object_reserve

    def get_connection_request(self, interp_ratio: float) -> bytes:
        fractured_role_key = struct.unpack('>4B', struct.pack('>L', int(self.role_key, 16)))
        fractured_mmr = struct.unpack('>4B', struct.pack('>f', self.mmr))
        ord_name = [ord(char) for char in self.name]
        data = [*ord_name, *fractured_mmr, *fractured_role_key, 0, 0, 0, interp_ratio]

        return self.pack_input_data(Action(Action.TYPE_INIT, 0, 0, data), 0)

    def unpack_redirection_address(self, message: bytes) -> Union[Tuple[str, int], None]:
        subport = int(self.unpack_single(message)[0].data[0])
        return (self.session_address, subport) if subport else None

    def unpack_connection_reply(self, message: bytes) -> Tuple[int, float, float]:
        entry = self.unpack_single(message)[0]
        assert entry.type == Entry.TYPE_INIT

        client_id = entry.id
        update_rate = entry.data[0]
        server_clock = entry.data[1]

        return client_id, update_rate, server_clock

    def unpack_server_data(self, data: Iterable[bytes]) -> Tuple[Iterable[Entry], int]:
        """
        Unpack state updates and local log request counter.

        Server update rate is expected to be on par with or lower than
        the client's tick rate, so there should not be an over-abundance of
        state updates. Otherwise, squashing could have been used to avoid
        too many local state updates and foreign entity interpolations.
        """

        entries, counters = zip(*(self.unpack_single(packet) for packet in data))
        local_log_request_counter = max(counters)

        return entries, local_log_request_counter

    @staticmethod
    def unpack_single(data: bytes) -> Tuple[Entry, int]:
        """Unpack a single incoming data packet into a state entry and a log request counter."""

        # HhBLf (2+2+1+4+4=13) + 12f3B (12*4+3*1=51) -> 64B
        packet = struct.unpack('>HhBLf12f3B', data)

        local_log_request_counter, entry_id, entry_type, counter, timestamp = packet[:5]
        entry_data = packet[5:]

        return Entry(entry_id, entry_type, counter, timestamp, entry_data), local_log_request_counter

    def pack_input_data(self, action: Action, global_log_request_counter: int) -> bytes:
        # HBLf (2+1+4+4=11) + 13B2hf (13*1+2*2+4=21) = 32B
        return struct.pack(
            '>HBLf13B2hf', global_log_request_counter, action.type, action.counter, action.timestamp, *action.data)
