"""Rules around a match of SDG"""

from itertools import chain
from collections import deque
from typing import Callable, Deque, Dict, Iterable, Tuple, Union
import numpy as np
from sidegame.game.shared.core import GameID, Map, Event
from sidegame.game.shared.objects import Object, C4
from sidegame.game.shared.player import Player


class Session:
    """
    Describes the states and flow of the game.

    References:
    https://counterstrike.fandom.com/wiki/Competitive
    https://www.gamersrdy.com/counter-strike-global-offensive/global-offensive-how-does-a-competitive-match-work/
    """

    ROUNDS_TO_SWITCH = 15
    ROUNDS_TO_WIN = 16

    TIME_TO_BUY = 20.
    TIME_TO_PLANT = 115.
    TIME_TO_DEFUSE = 40.  # Actually handled by planted C4 object
    TIME_TO_RESET = 7.

    FLAG_NULL = 0
    FLAG_START_MATCH = 1
    FLAG_END_MATCH = -1

    C4_EVENT_TYPES: Tuple[int] = (Event.C4_PLANTED, Event.C4_DETONATED, Event.C4_DEFUSED)

    def __init__(self, rng: np.random.Generator = None):
        self.rng = np.random.default_rng() if rng is None else rng

        # Init groups and subgroups
        self.objects: Dict[int, Object] = {}
        self.players: Dict[int, Player] = {}

        self.spectators: Dict[int, Player] = {}
        self.players_t: Dict[int, Player] = {}
        self.players_ct: Dict[int, Player] = {}

        self.groups: Dict[int, Dict[int, Player]] = {
            GameID.GROUP_SPECTATORS: self.spectators,
            GameID.GROUP_TEAM_T: self.players_t,
            GameID.GROUP_TEAM_CT: self.players_ct}

        # Preset team consitutions for auto mode
        self.assigned_teams: Dict[int, int] = None

        # Init match vars
        self.map: Map = None
        self.total_match_time = 0.
        self.total_round_time = 0.
        self.rounds_won_t: int = None
        self.rounds_won_ct: int = None
        self.loss_streak_t: int = None
        self.loss_streak_ct: int = None
        self.time = 0.
        self.phase: int = GameID.NULL
        self.object_counter: int = None

    def update(self, dt: float, events: Deque[Event], flag: int = FLAG_NULL):
        """
        Evaluate match-affecting events and advance the match,
        checking for and relaying any further changes in match state.
        """

        if self.phase == GameID.NULL and flag == self.FLAG_START_MATCH:
            events.extend(self.start_match())

        if self.phase != GameID.NULL:
            # Update objects
            for obj in self.objects.values():
                events.extend(obj.update(dt, chain(self.players_t.values(), self.players_ct.values()), self.map))

            # Iter and extend events
            idx = 0
            c4_event = None

            while idx < len(events):
                event = events[idx]
                event_type = event.type

                # Add newly spawned object to tracked objects
                if event_type == Event.OBJECT_SPAWN:
                    obj: Object = event.data
                    self.object_counter -= 1
                    new_obj_id = self.object_counter

                    obj.id = new_obj_id
                    self.objects[new_obj_id] = obj
                    self.map.object_id[obj.get_position_indices(obj.pos)] = new_obj_id

                # Remove expired object from tracked objects
                elif event_type == Event.OBJECT_EXPIRE:
                    object_id = event.data
                    obj = self.objects[object_id]
                    del self.objects[object_id]
                    self.map.object_id[obj.get_position_indices(obj.pos)] = Map.OBJECT_ID_NULL

                # Set C4 event
                elif event_type in self.C4_EVENT_TYPES:
                    c4_event = event

                # Evaluate player hit
                elif event_type == Event.PLAYER_DAMAGE:
                    damaged_player_id = event.data[1]
                    events.extend(self.players[damaged_player_id].eval_damage(
                        event, self.map, recoil=True, players=self.players))

                # Evaluate player death
                elif event_type == Event.PLAYER_DEATH:
                    self.handle_player_death(event)

                idx += 1

            # Check phase change conditions
            self.eval_state(dt, events, c4_event)

        if self.phase and flag == self.FLAG_END_MATCH:
            events.append(self.stop_match())

    def start_match(self, map_id: int = None, assign_c4: bool = True) -> Iterable[Event]:
        """Initialise the match on a specific map."""

        self.map = Map(map_id=map_id, rng=self.rng)
        self.total_match_time = 0.
        self.rounds_won_t = 0
        self.rounds_won_ct = 0

        self.time = 0.
        self.phase = GameID.PHASE_BUY

        events = deque()

        # If teams are predetermined, clear and refill them accordingly
        if self.assigned_teams is not None:
            for player_id in tuple(chain(self.players_t, self.players_ct)):
                events.append(self.move_player(player_id, GameID.GROUP_SPECTATORS))

            for player_id, team in self.assigned_teams.items():
                move_event = self.move_player(player_id, team)

                if move_event is not None:
                    events.append(move_event)

        events.append(Event(Event.CTRL_MATCH_STARTED, self.map.id))

        self.reset_side()
        c4_assignment = self.reset_round(assign_c4=assign_c4)

        if c4_assignment is not None:
            events.append(c4_assignment)

        return events

    def reset_side(self, switch_sides: bool = False):
        """
        Reset side-related variables per active player.
        If switching sides, positions within the team remain consistent.
        """

        # Teams begin with effectively one loss streak to mitigate effects of initial loss
        self.loss_streak_t = 1
        self.loss_streak_ct = 1

        if switch_sides:
            players_t_to_move = tuple(self.players_t.items())
            players_ct_to_move = tuple(self.players_ct.items())

            # Switch scores
            self.rounds_won_t, self.rounds_won_ct = self.rounds_won_ct, self.rounds_won_t

            # Move Ts to CT side
            # NOTE: CT position IDs offset T position IDs by 10
            for player_id, player in players_t_to_move:
                self.move_player(player_id, GameID.GROUP_TEAM_CT, position_id=(player.position_id+10))

            # Move CTs to T side
            # NOTE: T position IDs offset CT position IDs by -10
            for player_id, player in players_ct_to_move:
                self.move_player(player_id, GameID.GROUP_TEAM_T, position_id=(player.position_id-10))

        else:
            for player in self.players_t.values():
                player.reset_side()

            for player in self.players_ct.values():
                player.reset_side()

    def reset_round(self, assign_c4: bool = True) -> Union[Event, None]:
        """Reset round-related variables per active player."""

        self.total_round_time = 0.
        self.objects.clear()
        self.object_counter = 0
        self.map.reset()

        for player in self.players_t.values():
            player.reset_round(self.map.spawn_origin_t, self.map.player_id)

        for player in self.players_ct.values():
            player.reset_round(self.map.spawn_origin_ct, self.map.player_id)

        # Give C4 to a random T-side player
        if assign_c4 and self.players_t:
            player = self.players_t[self.rng.choice(tuple(self.players_t.keys()))]
            item = player.inventory.c4
            obj = C4(item, player)

            player.slots[item.slot + item.subslot] = obj
            return Event(Event.OBJECT_ASSIGN, (player.id, item.id, *obj.get_values(), player.money, 0))

        return None

    def eval_state(self, dt, events: Deque[Event], c4_event: Union[Event, None]):
        """Check match state and extend events with potential changes."""

        # NOTE: Clients should see time as decrementing wrt. phase time limits
        self.time += dt
        self.total_round_time += dt
        self.total_match_time += dt

        # Buy time over
        if self.phase == GameID.PHASE_BUY and self.time >= self.TIME_TO_BUY:
            events.append(self.change_phase(GameID.PHASE_PLANT))

        elif self.phase == GameID.PHASE_PLANT:
            if c4_event is not None and c4_event.type == Event.C4_PLANTED:
                events.append(self.change_phase(GameID.PHASE_DEFUSE))

                # +300 for the player who plants or defuses
                c4_event.data.owner.add_money(300)

            # CTs win by running down the clock
            # Loss bonus doesn't apply to Ts who didn't die before the timer ran out
            elif self.time >= self.TIME_TO_PLANT:
                events.append(self.change_phase(GameID.PHASE_RESET, t_win=False, penalise_alive_ts=True))

            # CTs win by aceing Ts
            elif self.check_lives(any, self.players_ct.values()) and not self.check_lives(any, self.players_t.values()):
                events.append(self.change_phase(GameID.PHASE_RESET, t_win=False))

            # Ts win by aceing CTs
            elif self.check_lives(any, self.players_t.values()) and not self.check_lives(any, self.players_ct.values()):
                events.append(self.change_phase(GameID.PHASE_RESET, t_win=True))

        elif self.phase == GameID.PHASE_DEFUSE:
            # Ts win by detonation
            # Winning by defusal or detonation grants 3500 base reward instead of 3250
            if c4_event is not None and c4_event.type == Event.C4_DETONATED:
                events.append(self.change_phase(GameID.PHASE_RESET, t_win=True, win_reward=3500))

            # CTs win by defusing
            # Winning by defusal or detonation grants 3500 base reward instead of 3250
            elif c4_event is not None and c4_event.type == Event.C4_DEFUSED:
                events.append(self.change_phase(GameID.PHASE_RESET, t_win=False, win_reward=3500))

                # +300 for the player who plants or defuses
                self.players[c4_event.data.defused_by].add_money(300)

                # +800 per player for Ts losing but planting before
                self.distribute_rewards(self.players_t.values(), [], 800)

            # Ts win by aceing CTs
            elif self.check_lives(any, self.players_t.values()) and not self.check_lives(any, self.players_ct.values()):
                events.append(self.change_phase(GameID.PHASE_RESET, t_win=True))

        # NOTE: Waits for reset time to pass in all cases, never instantly shuts down or freezes
        elif self.phase == GameID.PHASE_RESET and self.time >= self.TIME_TO_RESET:
            # Check for match end
            if self.rounds_won_t == self.ROUNDS_TO_WIN or self.rounds_won_ct == self.ROUNDS_TO_WIN:
                events.append(self.stop_match())

            # Check for side reset
            elif (self.rounds_won_t + self.rounds_won_ct) == self.ROUNDS_TO_SWITCH:
                self.reset_side(switch_sides=True)
                c4_assignment = self.reset_round()
                events.append(self.change_phase(GameID.PHASE_BUY))

                if c4_assignment is not None:
                    events.append(c4_assignment)

            # Standard round reset
            else:
                c4_assignment = self.reset_round()
                events.append(self.change_phase(GameID.PHASE_BUY))

                if c4_assignment is not None:
                    events.append(c4_assignment)

    def change_phase(
        self,
        phase: int,
        t_win: bool = None,
        win_reward: int = 3250,
        penalise_alive_ts: bool = False
    ) -> Event:
        """
        Transition between match phases and distribute rewards after
        the outcome of a round was determined.

        NOTE: Losing streaks are not reset upon winning, only decremented,
        to prevent situations where teams can be worse off when winning
        after a string of losses.
        See: https://blog.counter-strike.net/index.php/2019/03/23488/
        """

        self.time = 0.
        self.phase = phase

        if t_win is not None:
            if t_win:
                self.rounds_won_t += 1
                self.loss_streak_t = max(self.loss_streak_t - 1, 0)
                self.loss_streak_ct = min(self.loss_streak_ct + 1, 5)
                self.distribute_rewards(
                    self.players_t.values(), self.players_ct.values(), win_reward, self.loss_streak_ct)

            else:
                self.rounds_won_ct += 1
                self.loss_streak_ct = max(self.loss_streak_ct - 1, 0)
                self.loss_streak_t = min(self.loss_streak_t + 1, 5)
                players_t_values = (player for player in self.players_t.values() if not player.health) \
                    if penalise_alive_ts else self.players_t.values()
                self.distribute_rewards(self.players_ct.values(), players_t_values, win_reward, self.loss_streak_t)

        data = (
            t_win if t_win is not None else 0,
            penalise_alive_ts if penalise_alive_ts is not None else 0,
            win_reward,
            self.loss_streak_t,
            self.loss_streak_ct,
            self.time,
            self.phase,
            self.rounds_won_t,
            self.rounds_won_ct)

        return Event(Event.CTRL_MATCH_PHASE_CHANGED, data)

    def stop_match(self) -> Event:
        """Reset time and phase flag and relay final scores."""

        self.time = 0.
        self.phase = GameID.NULL

        return Event(Event.CTRL_MATCH_ENDED, (self.time, self.phase, self.rounds_won_t, self.rounds_won_ct))

    def check_lives(self, eval_fn: Callable, players: Iterable[Player]) -> bool:
        """
        Evaluate if a given subset of players remains alive.
        Expects a boolean function, i.e. `any` or `all`.
        """

        return eval_fn(player.health > 0. for player in players)

    def distribute_rewards(
        self, winners: Iterable[Player], losers: Iterable[Player], win_reward: int, loss_streak: int = 0
    ):
        """Add money per player depending on win/loss and loss streak."""

        for player in winners:
            player.add_money(win_reward)

        for player in losers:
            player.add_money(1400 + (loss_streak - 1) * 500)

    def handle_player_death(self, event: Event):
        """Update the scoreboard and add a reward (or penalty)."""

        killer_id, victim_id, item_id, _ = event.data

        killer = self.players[killer_id]
        victim = self.players[victim_id]

        if victim_id == killer_id or item_id == GameID.ITEM_C4:
            pass

        # -300 penalty on teamkill
        elif victim.team == killer.team:
            killer.add_money(-300)
            killer.kills = max(killer.kills-1, 0)

        # Individual weapon frag reward
        else:
            killer.add_money(killer.inventory.get_item_by_id(item_id).reward)
            killer.kills += 1

        victim.deaths += 1

    def add_player(self, player: Player):
        """Add an unsorted player to the session."""

        self.players[player.id] = player
        self.groups[player.team][player.id] = player

    def remove_player(self, player: Player):
        """Remove a sorted player from the session."""

        del self.groups[player.team][player.id]
        del self.players[player.id]

    def move_player(self, player_id: int, team: int, position_id: int = None) -> Union[Event, None]:
        """
        Assign the player a new team and a position within it.
        Returns `None` if no positions are free.
        """

        player = self.players[player_id]

        # Get position id
        if position_id is None:
            if team == GameID.GROUP_SPECTATORS:
                position_id = player_id

            else:
                id_range = range(GameID.PLAYER_T1, GameID.PLAYER_T5+1) if team == GameID.GROUP_TEAM_T else \
                    range(GameID.PLAYER_CT1, GameID.PLAYER_CT5+1)

                position_id = min(
                    set(id_range) - set(a_player.position_id for a_player in self.groups[team].values()), default=None)

                if position_id is None:
                    return None

        # Update groups
        del self.groups[player.team][player_id]
        self.groups[team][player_id] = player

        # Update player-side variables and return resulting team change event
        old_team = player.team
        player_moved_event = player.set_team(team, position_id)

        # When (and if) a player (re)connects, they should start from scratch,
        # i.e. as a new (reset) player in the middle of the round
        if self.phase:
            if team != GameID.GROUP_SPECTATORS:
                player.reset_side()
                player.reset_round(
                    self.map.spawn_origin_t if team == GameID.GROUP_TEAM_T else self.map.spawn_origin_ct,
                    self.map.player_id)
            elif old_team != GameID.GROUP_SPECTATORS:
                player.reset_side()
                self.map.player_id[player.get_covered_indices()] = Map.PLAYER_ID_NULL

        return player_moved_event

    def check_player_buy_eligibility(self, player_id: int) -> bool:
        """Check if the player can buy wrt. distance to spawn point, match phase, and time."""

        if not self.phase:
            return False

        player = self.players[player_id]

        if player.team == GameID.GROUP_SPECTATORS or not player.health:
            return False

        spawn_point = self.map.spawn_origin_t if player.team == GameID.GROUP_TEAM_T else self.map.spawn_origin_ct

        # Check phase/time and distance to spawn point
        valid_distance = np.linalg.norm(player.pos - spawn_point) <= Player.MAX_BUY_RANGE

        return player.global_buy_enabled or (valid_distance and (
            self.phase == GameID.PHASE_BUY or (self.phase == GameID.PHASE_PLANT and self.time < self.TIME_TO_BUY)))

    def check_player_message_access(self, player_id: int, sender_id: int) -> bool:
        """Check if player A can see a message from player B."""

        observer = self.players.get(player_id, None)
        sender = self.players.get(sender_id, None)

        return observer is not None and sender is not None and observer.team in (sender.team, GameID.GROUP_SPECTATORS)

    def is_player(self, player_id: int) -> bool:
        """Check if a player is participating in the match."""

        player: Player = self.players.get(player_id, None)
        return player is not None and player.team != GameID.GROUP_SPECTATORS

    def is_dead_player(self, player_id: int) -> bool:
        """Check if a player is spectating others, but otherwise participating in the match."""

        player: Player = self.players.get(player_id, None)
        return player is not None and (player.team != GameID.GROUP_SPECTATORS and not player.health)

    def is_spectator(self, player_id: int) -> bool:
        """Check if a player belongs to the spectator group."""

        player: Player = self.players.get(player_id, None)
        return player is not None and player.team == GameID.GROUP_SPECTATORS

    def is_dead_or_spectator(self, player_id: int) -> bool:
        """Check if a player is spectating other players."""

        player: Player = self.players.get(player_id, None)
        return player is not None and (player.team == GameID.GROUP_SPECTATORS or not player.health)
