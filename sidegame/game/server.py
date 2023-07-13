"""SDG server"""

import struct
from argparse import Namespace
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Tuple, Union
from numpy.random import default_rng
from sidegame.networking import Entry, Action, Entity, Server
from sidegame.game.shared import GameID, Map, Event, Message, Inventory, Object, C4, Player, Session


class SDGServer(Server):
    """A single authority for the state of a SDG multi-player session."""

    CLIENT_MESSAGE_SIZE = 32
    SERVER_MESSAGE_SIZE = 64
    MAX_TIME_TO_ASSEMBLE = 20.
    ENV_ID = Map.PLAYER_ID_NULL

    def __init__(self, args: Namespace, assigned_teams: Dict[str, int] = None, ip_config: dict[str, list[str]] = None):
        self.rng = default_rng(args.seed)
        self.time_scale = args.time_scale

        super().__init__(
            args.tick_rate,
            args.update_rate,
            (args.address, args.port),
            self.CLIENT_MESSAGE_SIZE,
            self.SERVER_MESSAGE_SIZE,
            logging_name='Session' if args.session_id is None else args.session_id,
            logging_path=args.logging_path,
            logging_level=args.logging_level,
            show_ticks=args.show_ticks,
            ip_config=ip_config)

        self.roles = {
            int(args.admin_key, 16): GameID.ROLE_ADMIN,
            int(args.player_key, 16): GameID.ROLE_PLAYER,
            int(args.spectator_key, 16): GameID.ROLE_SPECTATOR}

        self.auto: bool = assigned_teams is not None
        self.auto_init_timestamp: float = None
        self.assigned_teams = assigned_teams

        self.session = Session(rng=self.rng)
        self.inventory = Inventory()
        self.gathered_data_cache: Deque[Entry] = deque()

        self.bot_counter = 0

    def unpack_connection_request(self, data: bytes, client_id: int) -> float:
        request = self.unpack_single(data)[0].data

        player: Player = self.entities[client_id]
        name = ''.join(chr(ordinal) for ordinal in request[:4])
        role_key = struct.unpack('>L', struct.pack('>4B', *request[8:12]))[0]

        if role_key in self.roles:
            player.name = name
            player.role = self.roles[role_key]

        return request[-1]

    def get_connection_reply(self, client_id: int, update_rate: float, current_clock: float) -> bytes:
        data = [update_rate, current_clock, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0, 0]
        entry = Entry(client_id, Entry.TYPE_INIT, 0, 0., data)

        return self.pack_state_data((entry,), 0)[0]

    def create_entity(self, entity_id: int) -> Entity:
        return Player(entity_id, self.inventory, rng=self.rng)

    def handle_missing_entities(self, missing_entity_ids: List[int]) -> Iterable[Tuple[int, Any]]:
        removal_logs = deque()

        for entity_id in missing_entity_ids:
            # Remove from session
            if entity_id in self.session.players:
                player = self.session.players[entity_id]
                self.session.remove_player(player)

                if self.session.map is not None:
                    self.session.map.player_id[player.get_covered_indices()] = Map.PLAYER_ID_NULL

                # Generate disconnection log for clients
                data = [
                    float(entity_id), 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0, Event.CTRL_PLAYER_DISCONNECTED]

                # NOTE: These logs bypass queued events
                # (because they are handled after send, after queued events have already been logged)
                removal_logs.append((self.ENV_ID, data))

            # Remove from known entities
            del self.entities[entity_id]

        return removal_logs

    def unpack_client_data(self, data: Iterable[bytes]) -> Tuple[Iterable[Action], int]:
        actions, counters = zip(*(self.unpack_single(packet) for packet in data))
        global_log_request_counter = max(counters)

        # Get oldest log (most probable to be the next requested), discard the rest
        logs = [action for action in actions if action.type == Action.TYPE_LOG]
        log = min(logs, key=self.counter_key) if logs else None

        # Get squashed action state
        states = [action for action in actions if action.type == Action.TYPE_STATE]

        if len(states) == 1:
            state = states[0]

        elif len(states) > 1:
            # Transpose, because a list of lists doesn't allow "column" indexing
            data_groups = list(zip(*(state.data for state in states)))

            # Keep order in consideration
            ctrs = [state.counter for state in states]

            squashed_input = [self.squash_by_index(idx, data_group, ctrs) for idx, data_group in enumerate(data_groups)]

            counter = max(state.counter for state in states)
            timestamp = max(state.timestamp for state in states)
            state = Action(Action.TYPE_STATE, counter, timestamp, squashed_input)

        else:
            state = None

        return tuple(action for action in (log, state) if action is not None), global_log_request_counter

    @staticmethod
    def unpack_single(data: bytes) -> Tuple[Action, int]:
        """Unpack a single incoming data packet into an action and a log request counter."""

        # HBLf (2+1+4+4=11) + 13B2hr (13*1+2*2+4=21) -> 32B
        packet = struct.unpack('>HBLf13B2hf', data)

        global_log_request_counter, action_type, counter, timestamp = packet[:4]
        action_data = packet[4:]

        return Action(action_type, counter, timestamp, action_data), global_log_request_counter

    @staticmethod
    def counter_key(action: Action) -> int:
        """Allow sorting actions by their counter."""

        return action.counter

    def squash_by_index(self, idx: int, vals: Iterable[Union[int, float]], ctrs: List[int]) -> Union[int, float]:
        """
        Produce a single value from a possibly unordered sequence of values
        according to the nature of its content, distinguished by its index
        in the packet structure.
        """

        # Trigger flags are squashed with `max` to check for presence of trigger
        if idx < 6:
            return max(vals)

        # Multiple options with overriding effects have null values filtered out, then the latest one is returned
        elif idx == 6:
            ctrs_vals = sorted(ctr_val for ctr_val in zip(ctrs, vals) if ctr_val[1])
            return ctrs_vals[-1][1] if ctrs_vals else 0

        # Multiple options that are orderable are reduced to their median,
        # specifically, 'high median', where the middle pair in even-length lists is not interpolated,
        # instead returning one existing value of the pair
        elif idx < 9:
            return sorted(vals)[len(vals)//2]

        # Signed trigger flags are summed and clipped to get the effective trigger
        # NOTE: Values are strictly positive (B format) and need to be offset
        elif idx < 12:
            return max(-1, min(1, sum(vals)-len(vals))) + 1

        # Multiple options without order or clear indication of priority are reduced to their mode
        elif idx == 12:
            return max(vals, key=vals.count)

        # Combinations of filtering out null values and getting the mode
        elif idx == 13:
            vals = [val for val in vals if val != GameID.NULL]
            return max(vals, key=vals.count) if vals else GameID.NULL

        elif idx == 14:
            vals = [val for val in vals if val != Map.PLAYER_ID_NULL]
            return max(vals, key=vals.count) if vals else Map.PLAYER_ID_NULL

        # Accumulated differences are simply added together
        else:
            return sum(vals)

    def pack_state_data(self, state: Iterable[Entry], local_log_request_counter: int) -> Iterable[bytes]:
        # HhBLf (2+2+1+4+4=13) + 12f3B (12*4+3*1=51) -> 64B
        return [struct.pack(
            '>HhBLf12f3B', local_log_request_counter, entry.id, entry.type, entry.counter, entry.timestamp, *entry.data)
            for entry in state]

    def gather_state_data(self, entity: Entity) -> Iterable[Entry]:
        return self.gathered_data_cache

    def get_gathered_data_cache(self, timestamp: float):
        """
        Decompose the state of the match, participating entities, and tracked
        objects into a sequence of fixed-length data entries.

        At 30Hz update rate, is is assumed that every entity (other than
        spectators, who don't actively participate in the match) has changes on
        each update tick. Out of them, entities out of line of sight of the
        gathering entity should have their data sent to it, as well, because
        emitted sounds are tied to source positions. Thus, every gathering
        entity is sent data from every observed entity on each update tick
        (line of sight etc. is determined client-side). The (state) data is
        collected in advance to not repeat the gathering process unnecessarily.

        NOTE: Because gathering entities all access the data cache in the same
        way, the data being sent is the same for all clients. However,
        there are no guarantees when individual clients will receive, process,
        or reply to the server, so the environment is still asynchronous.
        """

        self.gathered_data_cache.clear()
        skip_spectators = self.session.players_t or self.session.players_ct

        for player in self.session.players.values():
            if player.team == GameID.GROUP_SPECTATORS and skip_spectators:
                continue

            data = [
                player.pos[0], player.pos[1], player.vel[0], player.vel[1], player.angle, player.d_angle_recoil,
                player.d_pos_recoil[0], player.d_pos_recoil[1],
                player.time_since_damage_taken, player.time_until_drawn, player.time_to_reload, player.time_off_trigger,
                player.held_object.item.id if player.held_object is not None else 0,
                player.held_object.magazine if player.held_object is not None else 0,
                player.held_object.reserve if player.held_object is not None else 0]

            self.gathered_data_cache.append(Entry(player.id, Entry.TYPE_STATE, player.action_counter, timestamp, data))

        for obj in self.session.objects.values():
            data = [obj.pos[0], obj.pos[1], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0, 0]

            # Use approximate tick counter as the object entity's state counter
            self.gathered_data_cache.append(
                Entry(obj.id, Entry.TYPE_STATE, int(self._update_rate*timestamp), timestamp, data))

    def apply_action(
        self,
        entity: Entity,
        action: Action,
        timestamp: float,
        lag: float,
        lerp: float
    ) -> Iterable[Any]:

        session = self.session
        player: Player = entity

        # NOTE: In physically close networks, lag could potentially come out slightly negative (-0.0005, -0.0009, ...)
        # due to the client and server's compensation of differences between the times
        # when messages were sent and when they were processed (depends on tick intervals)
        # Thus, it needs to be clipped to prevent lag compensation from being affected (undercutting lerp)
        player.latency = max(0., lag)

        # Assign to a group on first action eval after connecting
        if player.id not in session.players:
            session.add_player(player)

            return [Event(Event.CTRL_PLAYER_CONNECTED, (
                session.time, session.total_round_time, session.total_match_time, player.id, player.name))]

        # Only active players, i.e. not spectators or killed players, have their update called
        # Others still gather data and can issue logs, but cannot otherwise interact with the world until (re)spawning
        elif not session.phase or not player.health or player.team == GameID.GROUP_SPECTATORS:
            return Event.EMPTY_EVENT_LIST

        # In buy phase, prevent movement and firing
        grounded = session.phase == GameID.PHASE_BUY

        return player.update(
            action, session.players, session.objects, session.map, timestamp, lag+lerp, grounded, self.time_scale)

    def update_state(
        self,
        dt: float,
        timestamp: float,
        queued_logs: Deque[Action],
        queued_events: Deque[Event]
    ) -> Iterable[Tuple[int, Any]]:

        # Default session control flag
        flag = Session.FLAG_NULL

        # Self-start or self-termination
        if self.auto and not self.session.phase:

            # Set time of initialisation
            if self.auto_init_timestamp is None:
                self.auto_init_timestamp = timestamp

            # A failsafe to break if nothing happens after a set amount of time
            # Should also allow to self-terminate after end of match is reached, because the time should be far exceeded
            elif timestamp - self.auto_init_timestamp > self.MAX_TIME_TO_ASSEMBLE:

                # Break main loop
                self.session_running = False

                # Signal clients to end the processes on their side as well
                # NOTE: In practice (on Windows, at least), breaking the main loop will make
                # the server's socket unreachable to clients, possibly before they receive the signal,
                # causing a 'Lost connection to the server' message instead of the expected exit
                # (despite the socket being UDP; see the docstring of `networking.core::ServerSocket.recv`)
                queued_events.append(Event(Event.CTRL_SESSION_ENDED, None))

            # Attempt self-start
            elif len(self.session.players) >= len(self.assigned_teams):
                # Distinguish between actors based on name
                # This is relayed to and through the matchmaker and upon connecting to the session server
                assigned_teams = {
                    player.id: self.assigned_teams[player.name]
                    for player in self.session.players.values() if player.name in self.assigned_teams}

                # Match is started if all named players are present
                if len(assigned_teams) == len(self.assigned_teams):
                    flag = Session.FLAG_START_MATCH
                    self.session.assigned_teams = assigned_teams

        # Handle commands/update global history (use queued action logs to add actual logs)
        for log in queued_logs:
            event = self.handle_user_log(log)

            # Check for session control flags, otherwise add to standard game events
            if event is not None:
                if event.type == Event.CTRL_MATCH_STARTED:
                    flag = Session.FLAG_START_MATCH

                elif event.type == Event.CTRL_MATCH_ENDED:
                    flag = Session.FLAG_END_MATCH

                # TODO: Temporary fix to update clientside money display
                elif event.type == Event.OBJECT_SPAWN and log.data[1] == GameID.LOG_BUY:
                    queued_events.append(event)

                    obj = event.data
                    queued_events.append(Event(
                        Event.OBJECT_ASSIGN,
                        (obj.owner.id, obj.item.id, 0., 0, 0, 0, obj.owner.money, obj.item.price)))

                else:
                    queued_events.append(event)

        # Update session
        self.session.update(dt * self.time_scale, queued_events, flag=flag)

        # Cache current state data
        self.get_gathered_data_cache(timestamp)

        # Iter events, make new logs to update global history
        new_global_logs = [self.create_global_log(event) for event in queued_events]

        return new_global_logs

    def handle_user_log(self, log: Action) -> Union[Event, None]:
        """Verify and evaluate user commands."""

        log_data = log.data
        player_id = log_data[0]
        log_id = log_data[1]
        user_role = self.entities[player_id].role

        # Anyone can try to change their role or name
        if log_id == GameID.CMD_SET_ROLE:
            player: Player = self.entities[player_id]
            role_key = struct.unpack('>L', struct.pack('>4B', *log_data[2:6]))[0]

            if role_key in self.roles:
                player.role = self.roles[role_key]

            self.logger.debug('Elevated client %d to role %d.', player_id, player.role)
            return None

        elif log_id == GameID.CMD_SET_NAME:
            player: Player = self.session.players[player_id]
            name = ''.join(chr(ordinal) for ordinal in log_data[2:6])

            return player.set_name(name)

        elif user_role < GameID.ROLE_PLAYER:
            return None

        # Player elevated commands
        if log_id == GameID.CMD_SET_TEAM:
            moved_player_id, team = log_data[2:4]

            # Correction to allow only the admin to move other players
            if user_role < GameID.ROLE_ADMIN and moved_player_id != player_id:
                return None

            return self.session.move_player(moved_player_id, team)

        elif log_id == GameID.LOG_BUY:
            can_buy = self.session.check_player_buy_eligibility(player_id)

            if can_buy:
                player: Player = self.session.players[player_id]
                item_id = log_data[2]
                return player.buy(item_id)

            return None

        elif log_id == GameID.LOG_MESSAGE:
            player: Player = self.session.players[player_id]

            if player.team == GameID.GROUP_SPECTATORS:
                return None

            sender_position_id = player.position_id
            msg_round = self.session.rounds_won_t + self.session.rounds_won_ct + \
                (1 if self.session.phase != GameID.PHASE_RESET else 0)
            msg_time = self.session.total_round_time
            words = log_data[2:6]
            marks = [log_data[6:8], log_data[8:10], log_data[10:12], log_data[12:14]]

            msg = Message(sender_position_id, msg_round, msg_time, words, marks=marks, sender_id=player_id)

            return Event(Event.PLAYER_MESSAGE, msg)

        elif log_id == GameID.CMD_GET_LATENCY:
            latencies = [
                (a_player.id, int(min(a_player.latency * 1000., 255.)))
                for a_player in self.session.players.values() if a_player.team != GameID.GROUP_SPECTATORS][:10]

            return Event(Event.CTRL_LATENCY_REQUESTED, (player_id, latencies))

        elif user_role < GameID.ROLE_ADMIN:
            return None

        # Admin/dev elevated commands
        if log_id == GameID.CMD_START_MATCH:
            return Event(Event.CTRL_MATCH_STARTED, None)

        elif log_id == GameID.CMD_END_MATCH:
            return Event(Event.CTRL_MATCH_ENDED, None)

        elif log_id == GameID.CMD_END_SESSION:
            self.session_running = False
            return Event(Event.CTRL_SESSION_ENDED, None)

        elif log_id == GameID.CHEAT_END_ROUND:
            # Runs down the timer in the buy or planting phase
            if self.session.phase == GameID.PHASE_BUY:
                self.session.time = Session.TIME_TO_BUY

            elif self.session.phase == GameID.PHASE_PLANT:
                self.session.time = Session.TIME_TO_PLANT

        elif log_id == GameID.CHEAT_DEV_MODE:
            player: Player = self.entities[player_id]
            player.dev_mode = not player.dev_mode
            return Event(Event.CTRL_PLAYER_CHANGED, (player_id, player.name, player.money, player.dev_mode))

        elif log_id == GameID.CHEAT_MAX_MONEY:
            player: Player = self.entities[player_id]
            player.money = Player.MONEY_CAP
            return Event(Event.CTRL_PLAYER_CHANGED, (player_id, player.name, player.money, player.dev_mode))

        elif log_id == GameID.CMD_ADD_BOT:
            if self.bot_counter == 256:
                return None

            bot_counter = self.bot_counter
            self.bot_counter += 1

            client_id = self._socket._node_counter
            self._socket._node_counter += 1

            bot: Player = self.create_entity(client_id)
            bot.name = f'{"bot"[:4-len(str(bot_counter))]}{bot_counter}'
            bot.role = GameID.ROLE_PLAYER
            bot.dev_mode = bool(log_data[2])
            bot.action_counter = 0

            self.session.add_player(bot)

            self.logger.info('Client %d connected (bot %d).', client_id, bot_counter)

            return Event(Event.CTRL_PLAYER_CONNECTED, (
                self.session.time, self.session.total_round_time, self.session.total_match_time, bot.id, bot.name))

        elif log_id == GameID.CMD_KICK_NAME:
            kicked_name = ''.join(chr(ordinal) for ordinal in log_data[2:6])
            kicked_id = None
            kicked_player = None

            for player_id, player in self.session.players.items():
                if player.name == kicked_name:
                    kicked_id = player.id
                    kicked_player = player
                    break

            if kicked_id is None:
                return None

            else:
                self.session.remove_player(kicked_player)

                if self.session.map is not None:
                    self.session.map.player_id[kicked_player.get_covered_indices()] = Map.PLAYER_ID_NULL

            if kicked_id in self.entities:
                del self.entities[kicked_id]

                if self._blocked_ips is None:
                    self._blocked_ips = []

                for client_key, client in tuple(self._clients.items()):
                    if client.id == kicked_id:
                        self._blocked_ips.append(client.address[0])
                        del self._clients[client_key]
                        break

            else:
                self.bot_counter -= 1

            return Event(Event.CTRL_PLAYER_DISCONNECTED, kicked_id)

        return None

    def create_global_log(self, event: Event) -> Tuple[int, Any]:
        """
        Prepare event data of variable size for packing
        by converting it to a list of predetermined length and structure.
        """

        event_type = event.type

        if event_type == Event.CTRL_MATCH_STARTED:
            map_id = event.data
            data = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, map_id, event_type]

        elif event_type == Event.CTRL_MATCH_ENDED:
            time, phase, rounds_won_t, rounds_won_ct = event.data
            data = [time, float(phase), 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., rounds_won_t, rounds_won_ct, event_type]

        elif event_type == Event.CTRL_MATCH_PHASE_CHANGED:
            t_win, penalise_alive_ts, win_reward, loss_streak_t, loss_streak_ct, \
                time, phase, rounds_won_t, rounds_won_ct = event.data

            data = [
                float(t_win), float(penalise_alive_ts), float(win_reward),
                float(loss_streak_t), float(loss_streak_ct),
                time, float(phase), 0., 0., 0., 0., 0., rounds_won_t, rounds_won_ct, event_type]

        elif event_type == Event.CTRL_SESSION_ENDED:
            data = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0, event_type]

        elif event_type == Event.CTRL_PLAYER_CONNECTED:
            phase_time, round_time, match_time, player_id, name = event.data

            data = [
                phase_time, round_time, match_time, float(player_id),
                float(ord(name[0])), float(ord(name[1])), float(ord(name[2])), float(ord(name[3])),
                0., 0., 0., 0., 0, 0, event_type]

        elif event_type == Event.CTRL_PLAYER_DISCONNECTED:
            entity_id = event.data

            data = [float(entity_id), 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0, event_type]

        elif event_type == Event.CTRL_PLAYER_MOVED:
            player_id, team, position_id = event.data
            data = [float(player_id), 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., team, position_id, event_type]

        elif event_type == Event.CTRL_PLAYER_CHANGED:
            player_id, name, money, dev_mode = event.data

            data = [
                float(player_id),
                float(ord(name[0])), float(ord(name[1])), float(ord(name[2])), float(ord(name[3])),
                float(money), float(dev_mode), 0., 0., 0., 0., 0., 0, 0, event_type]

        elif event_type == Event.CTRL_LATENCY_REQUESTED:
            requestor_id, latencies = event.data

            data = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., float(requestor_id), 0., 0, len(latencies), event_type]

            for i, (a_player_id, a_player_latency) in enumerate(latencies):
                data[i] = struct.unpack('>f', struct.pack('>4B', a_player_id, a_player_latency, 0, 0))[0]

        elif event_type == Event.OBJECT_SPAWN:
            obj: Object = event.data
            data = [
                float(obj.id), float(obj.owner.id), obj.lifetime,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0, obj.item.id, event_type]

        elif event_type == Event.OBJECT_ASSIGN:
            obj_owner_id, obj_item_id, durability, magazine, reserve, carrying, money, spending = event.data
            data = [
                float(obj_owner_id),
                durability, float(magazine), float(reserve), float(carrying),
                float(money), float(spending), 0., 0., 0., 0., 0., 0, obj_item_id, event_type]

        elif event_type == Event.C4_PLANTED:
            obj: C4 = event.data
            data = [float(obj.owner.id), 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0, event_type]

        elif event_type == Event.C4_DEFUSED:
            obj: C4 = event.data
            data = [float(obj.id), float(obj.defused_by), 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0, event_type]

        elif event_type in Event.ID_EVENTS:
            entity_id = event.data
            data = [float(entity_id), 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0, event_type]

        elif event_type == Event.FX_ATTACK:
            attacker_id, item_id = event.data
            data = [float(attacker_id), 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, item_id, event_type]

        elif event_type == Event.FX_FLASH:
            attacker_id, flashed_id, debuff, duration = event.data

            data = [
                float(attacker_id), float(flashed_id), debuff, duration,
                0., 0., 0., 0., 0., 0., 0., 0., 0, 0, event_type]

        elif event_type == Event.FX_WALL_HIT:
            (pos_x, pos_y), attacker_id, item_id = event.data
            data = [pos_x, pos_y, float(attacker_id), 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, item_id, event_type]

        elif event_type == Event.PLAYER_RELOAD:
            reloader_id, rld_evtype, rld_item_id = event.data
            data = [float(reloader_id), 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., rld_evtype, rld_item_id, event_type]

        elif event_type == Event.PLAYER_DAMAGE:
            attacker_id, damaged_id, item_id, damage = event.data

            data = [
                float(attacker_id), float(damaged_id), damage,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0, item_id, event_type]

        elif event_type == Event.PLAYER_DEATH:
            attacker_id, damaged_id, item_id, excess = event.data
            data = [
                float(attacker_id), float(damaged_id), excess,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0, item_id, event_type]

        elif event_type == Event.PLAYER_MESSAGE:
            msg: Message = event.data

            data = [
                *(float(coord) for coord in sum(msg.marks, tuple())),
                msg.time, float(msg.words[0]), float(msg.words[1]),
                float(msg.words[2]), msg.words[3], msg.round, msg.position_id]

            return msg.sender_id, data

        else:
            # There shouldn't be any events missed by previous branches, but still, to cover the else case
            data = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0, event_type]

        return self.ENV_ID, data
