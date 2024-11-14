"""Extraction of game scores, markings, and utilisation."""

import os
import json
import struct
from logging import Logger
from time import asctime
from typing import List, Tuple, Union

import numpy as np
import cv2
import psutil

from sidegame.utils_jit import vec2_norm2
from sidegame.assets import ImageBank, MapID, ASSET_DIR
from sidegame.graphics import draw_image
from sidegame.networking.core import Recorder
from sidegame.physics import get_position_indices
from sidegame.game import GameID, EventID
from sidegame.game.shared import Player, Session
from sidegame.game.client.simulation import Simulation


DATA_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'user_data'))


class StatTracker:
    """Keeps and updates statistics accumulated in past sessions and during the game."""

    def __init__(self, session: Session, own_player: Player):
        self.session = session
        self.own_player = own_player
        self.data_dir = DATA_DIR

        self.last_timestamp: float = None
        self.last_pos: np.ndarray = None
        self.temp_killer_info: Tuple[int, float] = None

        self.temp_scores = {
            'kills_this_round': 0,
            'kills_to_clutch': 0,
            'allies_planted': 0,
            'enemies_planted': 0,
            'allies_defused': 0,
            'planted': 0,
            'defused': 0,
            'kast_triggered': 0,
            'opening_triggered': 0,
            'damage': 0.,
            'total_team_damage': 0.}

        self.temp_player_damage = {
            GameID.PLAYER_T1: 0.,
            GameID.PLAYER_T2: 0.,
            GameID.PLAYER_T3: 0.,
            GameID.PLAYER_T4: 0.,
            GameID.PLAYER_T5: 0.,
            GameID.PLAYER_CT1: 0.,
            GameID.PLAYER_CT2: 0.,
            GameID.PLAYER_CT3: 0.,
            GameID.PLAYER_CT4: 0.,
            GameID.PLAYER_CT5: 0.}

        self.temp_last_contact_time = {
            GameID.PLAYER_T1: 0.,
            GameID.PLAYER_T2: 0.,
            GameID.PLAYER_T3: 0.,
            GameID.PLAYER_T4: 0.,
            GameID.PLAYER_T5: 0.,
            GameID.PLAYER_CT1: 0.,
            GameID.PLAYER_CT2: 0.,
            GameID.PLAYER_CT3: 0.,
            GameID.PLAYER_CT4: 0.,
            GameID.PLAYER_CT5: 0.}

        self.tracked_item_damage = {
            GameID.ITEM_RIFLE_T: 0.,
            GameID.ITEM_RIFLE_CT: 0.,
            GameID.ITEM_SMG_T: 0.,
            GameID.ITEM_SMG_CT: 0.,
            GameID.ITEM_SHOTGUN_T: 0.,
            GameID.ITEM_SHOTGUN_CT: 0.,
            GameID.ITEM_SNIPER: 0.,
            GameID.ITEM_PISTOL_T: 0.,
            GameID.ITEM_PISTOL_CT: 0.,
            GameID.ITEM_KNIFE: 0.,
            GameID.ITEM_C4: 0.,
            GameID.ITEM_FLASH: 0.,
            GameID.ITEM_EXPLOSIVE: 0.,
            GameID.ITEM_INCENDIARY_T: 0.,
            GameID.ITEM_INCENDIARY_CT: 0.,
            GameID.ITEM_SMOKE: 0.}

        self.tracked_item_buys = {
            GameID.ITEM_ARMOUR: 0,
            GameID.ITEM_RIFLE_T: 0,
            GameID.ITEM_RIFLE_CT: 0,
            GameID.ITEM_SMG_T: 0,
            GameID.ITEM_SMG_CT: 0,
            GameID.ITEM_SHOTGUN_T: 0,
            GameID.ITEM_SHOTGUN_CT: 0,
            GameID.ITEM_SNIPER: 0,
            GameID.ITEM_PISTOL_T: 0,
            GameID.ITEM_PISTOL_CT: 0,
            GameID.ITEM_KNIFE: 0,
            GameID.ITEM_DKIT: 0,
            GameID.ITEM_FLASH: 0,
            GameID.ITEM_EXPLOSIVE: 0,
            GameID.ITEM_INCENDIARY_T: 0,
            GameID.ITEM_INCENDIARY_CT: 0,
            GameID.ITEM_SMOKE: 0}

        self.tracked_item_uses = {
            GameID.ITEM_ARMOUR: 0,
            GameID.ITEM_RIFLE_T: 0,
            GameID.ITEM_RIFLE_CT: 0,
            GameID.ITEM_SMG_T: 0,
            GameID.ITEM_SMG_CT: 0,
            GameID.ITEM_SHOTGUN_T: 0,
            GameID.ITEM_SHOTGUN_CT: 0,
            GameID.ITEM_SNIPER: 0,
            GameID.ITEM_PISTOL_T: 0,
            GameID.ITEM_PISTOL_CT: 0,
            GameID.ITEM_KNIFE: 0,
            GameID.ITEM_DKIT: 0,
            GameID.ITEM_FLASH: 0,
            GameID.ITEM_EXPLOSIVE: 0,
            GameID.ITEM_INCENDIARY_T: 0,
            GameID.ITEM_INCENDIARY_CT: 0,
            GameID.ITEM_SMOKE: 0}

        self.tracked_heatmap = np.zeros((64, 64))

        self.tracked_scores = {
            'damage': 0.,
            'kills': 0,
            'deaths': 0,
            'assists': 0,
            'own_team_kills': 0,
            'suicides': 0,
            'multikill_rounds': 0,
            'multikills': 0,
            'clutch_rounds': 0,
            'clutches': 0,
            'clutchkills': 0,
            'plants': 0,
            'defuses': 0,
            'money_spent': 0,
            'distance': 0.,
            'footsteps': 0,
            'game_time': 0.,
            'messages': 0,
            'remaining_health': 0,
            'afterplant_rounds': 0,
            'afterplants': 0,
            'retake_rounds': 0,
            'retakes': 0,
            't_rounds': 0,
            'takes': 0,
            'ct_rounds': 0,
            'holds': 0,
            't_rounds_won': 0,
            'ct_rounds_won': 0,
            'kast_rounds': 0,
            'round_win_shares': 0.,
            'openings': 0,
            'opening_tries': 0,
            'matches_won': 0,
            'matches': 0}

    def save(self) -> str:
        """Add currently tracked stats to all-time stats and save them."""

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        path_to_stats: str = os.path.join(self.data_dir, f'{self.own_player.name}_stats.json')
        path_to_heatmap: str = os.path.join(self.data_dir, f'{self.own_player.name}_heatmap.npy')

        # Get stats
        if os.path.exists(path_to_stats):
            with open(path_to_stats, 'r') as stats_file:
                stats = json.load(stats_file)
        else:
            stats = {}

        if not stats or not stats['alltime_scores']:
            stats = {
                'alltime_item_damage': {str(key): 0 for key in self.tracked_item_damage},
                'alltime_item_buys': {str(key): 0 for key in self.tracked_item_buys},
                'alltime_item_uses': {str(key): 0 for key in self.tracked_item_uses},
                'alltime_scores': {str(key): 0 for key in self.tracked_scores}}

        # Update all-time stats with currently tracked stats
        for key in self.tracked_item_damage:
            stats['alltime_item_damage'][str(key)] += self.tracked_item_damage[key]

        for key in self.tracked_item_buys:
            stats['alltime_item_buys'][str(key)] += self.tracked_item_buys[key]

        for key in self.tracked_item_uses:
            stats['alltime_item_uses'][str(key)] += self.tracked_item_uses[key]

        for key in self.tracked_scores:
            stats['alltime_scores'][str(key)] += self.tracked_scores[key]

        # Add currently tracked stats
        entry_indices = [int(entry.split('_')[2]) for entry in stats if entry.startswith('session_scores')]
        entry_idx = (max(entry_indices)+1) if entry_indices else 0

        stats[f'session_item_damage_{entry_idx:03d}'] = self.tracked_item_damage
        stats[f'session_item_buys_{entry_idx:03d}'] = self.tracked_item_buys
        stats[f'session_item_uses_{entry_idx:03d}'] = self.tracked_item_uses
        stats[f'session_scores_{entry_idx:03d}'] = self.tracked_scores
        stats[f'session_endtime_{entry_idx:03d}'] = asctime()

        # Get heatmap
        heatmap = np.load(path_to_heatmap) if os.path.exists(path_to_heatmap) else np.zeros((64, 64, 1))

        # Update all-time heatmap with currently tracked heatmap
        heatmap[..., 0] = heatmap[..., 0] + self.tracked_heatmap

        # Add currently tracked heatmap
        heatmap = np.concatenate((heatmap, self.tracked_heatmap[..., None]), axis=-1)

        # Save updated stats
        with open(path_to_stats, 'w') as stats_file:
            json.dump(stats, stats_file)

        np.save(path_to_heatmap, heatmap)

        return path_to_stats

    def reset_temp(self):
        """Reset round specific (temporary) stats."""

        for key in self.temp_scores:
            self.temp_scores[key] *= 0

        for key in self.temp_player_damage:
            self.temp_player_damage[key] = 0.
            self.temp_last_contact_time[key] = 0.

        self.temp_killer_info = None

    def reset_full(self):
        """Reset stats tracked since the session was started."""

        self.reset_temp()

        for key in self.tracked_item_damage:
            self.tracked_item_damage[key] *= 0

        for key in self.tracked_item_buys:
            self.tracked_item_buys[key] *= 0

        for key in self.tracked_item_uses:
            self.tracked_item_uses[key] *= 0

        for key in self.tracked_scores:
            self.tracked_scores[key] *= 0

        self.tracked_heatmap.fill(0.)
        self.last_timestamp = None

    def update_from_state(self, pos: float, timestamp: float, time_scale: float = 1.):
        """
        Update uptime, distance travelled, and heatmap of positions
        (time spent at rounded location).
        """

        if self.own_player.team == GameID.GROUP_SPECTATORS:
            return

        dt = 0. if self.last_timestamp is None else (timestamp - self.last_timestamp) * time_scale
        self.last_timestamp = timestamp

        if self.session.phase != GameID.PHASE_BUY and self.own_player.health:
            self.tracked_heatmap[get_position_indices(self.own_player.pos / 10.)] += dt
            self.tracked_scores['game_time'] += dt

            self.tracked_scores['distance'] += vec2_norm2(pos - self.last_pos) if self.last_pos is not None else 0.
            self.last_pos = pos

    def update_from_event(self, event_src: int, event_id: int, event_data: List[Union[int, float]], timestamp: float):
        """Update stats wrt. in-game event."""

        if self.own_player.team == GameID.GROUP_SPECTATORS:
            return

        # Messaging
        elif event_src != MapID.PLAYER_ID_NULL:
            sender_position_id = event_data[-1]

            if sender_position_id == self.own_player.position_id:
                self.tracked_scores['messages'] += 1

            return

        # Money spent on self-assignment
        if event_id == EventID.OBJECT_ASSIGN:
            assignee_id = int(event_data[0])
            obj_item_id = event_data[-2]

            carrying = int(event_data[4])
            spending = int(event_data[6])

            if carrying and spending and assignee_id == self.own_player.id:
                item = self.own_player.inventory.get_item_by_id(obj_item_id)
                self.tracked_scores['money_spent'] += spending
                self.tracked_item_buys[item.id] += 1

        # Flags for objective success rate
        elif event_id == EventID.C4_PLANTED and self.session.phase == GameID.PHASE_PLANT:
            planter_id = int(event_data[0])

            if planter_id in self.session.groups[self.own_player.team]:
                self.temp_scores['allies_planted'] = 1

                if planter_id == self.own_player.id:
                    self.temp_scores['planted'] = 1
                    self.tracked_scores['plants'] += 1

            else:
                self.temp_scores['enemies_planted'] = 1

        elif event_id == EventID.C4_DEFUSED:
            defuser_id = int(event_data[1])

            # Flag for RWS
            if defuser_id in self.session.groups[self.own_player.team]:
                self.temp_scores['allies_defused'] = 1

                if defuser_id == self.own_player.id:
                    self.temp_scores['defused'] = 1
                    self.tracked_scores['defuses'] += 1

        elif event_id == EventID.PLAYER_DAMAGE:
            attacker_id = int(event_data[0])
            victim_id = int(event_data[1])
            damage = event_data[2]
            item_id = event_data[-2]

            attacker = self.session.players[attacker_id]
            victim = self.session.players[victim_id]

            if victim.team != self.own_player.team:
                # RWS data
                if attacker.team == self.own_player.team and self.session.phase != GameID.PHASE_RESET:
                    self.temp_scores['total_team_damage'] += damage

                # Other damage-associated data
                if attacker_id == self.own_player.id:
                    self.tracked_scores['damage'] += damage
                    self.tracked_item_damage[item_id] += damage

                    if self.session.phase != GameID.PHASE_RESET:
                        self.temp_scores['damage'] += damage
                        self.temp_player_damage[victim.position_id] += damage
                        self.temp_last_contact_time[victim.position_id] = timestamp

        elif event_id == EventID.FX_ATTACK:
            attacker_id = int(event_data[0])
            item_id = event_data[-2]

            if attacker_id == self.own_player.id:
                self.tracked_item_uses[item_id] += 1

        elif event_id == EventID.FX_FOOTSTEP:
            player_id = int(event_data[0])

            if player_id == self.own_player.id:
                self.tracked_scores['footsteps'] += 1

        # Keep track of flash assists, count debuff as flash damage
        elif event_id == EventID.FX_FLASH and self.session.phase != GameID.PHASE_RESET:
            attacker_id = int(event_data[0])
            victim_id = int(event_data[1])
            debuff = event_data[2]

            if attacker_id == self.own_player.id:
                victim = self.session.players[victim_id]

                self.temp_last_contact_time[victim.position_id] = timestamp

                # Offset for own teammates debuffed
                if victim.team == self.own_player.team:
                    self.tracked_item_damage[GameID.ITEM_FLASH] -= debuff
                else:
                    self.tracked_item_damage[GameID.ITEM_FLASH] += debuff

        elif event_id == EventID.PLAYER_DEATH:
            attacker_id = int(event_data[0])
            victim_id = int(event_data[1])

            enemy_team = GameID.GROUP_TEAM_T if self.own_player.team == GameID.GROUP_TEAM_CT else GameID.GROUP_TEAM_CT
            alive_ts = [player.id for player in self.session.groups[self.own_player.team].values() if player.health]
            alive_cts = [player.id for player in self.session.groups[enemy_team].values() if player.health]

            attacker = self.session.players[attacker_id]
            victim = self.session.players[victim_id]

            if self.session.phase != GameID.PHASE_RESET:
                # Clutch flag
                if len(alive_ts) == 1 and alive_ts[0] == self.own_player.id and not self.temp_scores['kills_to_clutch']:
                    self.temp_scores['kills_to_clutch'] = len(alive_cts)
                    self.tracked_scores['clutch_rounds'] += 1

                elif len(alive_cts) == 1 and alive_cts[0] == self.own_player.id and (
                    not self.temp_scores['kills_to_clutch']
                ):
                    self.temp_scores['kills_to_clutch'] = len(alive_ts)
                    self.tracked_scores['clutch_rounds'] += 1

                # Opening
                if not self.temp_scores['opening_triggered'] and attacker.team != victim.team:
                    self.temp_scores['opening_triggered'] = 1

                    if attacker_id == self.own_player.id:
                        self.tracked_scores['openings'] += 1
                        self.tracked_scores['opening_tries'] += 1
                    elif victim_id == self.own_player.id:
                        self.tracked_scores['opening_tries'] += 1

            if victim_id == self.own_player.id:
                # KAST flag
                # If killed in death event, store attacker id and timestamp of death
                # On subsequent death event, if attacker dies and timestamp of that is within 5 seconds, log as trade
                self.temp_killer_info = (attacker_id, timestamp)

            if victim.team == enemy_team and attacker_id != self.own_player.id:
                damage_to_victim = self.temp_player_damage[victim.position_id]

                # Assist check, recent damage (or flash) counts towards assists as well
                if damage_to_victim > 40. or (timestamp - self.temp_last_contact_time[victim.position_id]) < 5.:
                    self.tracked_scores['assists'] += 1
                    self.temp_scores['kast_triggered'] = 1

                # KAST flag (trade)
                if self.temp_killer_info is not None and victim_id == self.temp_killer_info[0] and (
                    timestamp - self.temp_killer_info[1]
                ) < 5.:
                    self.temp_scores['kast_triggered'] = 1

            # Friendly fire check
            elif victim.team == self.own_player.team and attacker_id == self.own_player.id:
                if victim_id != self.own_player.id:
                    self.tracked_scores['own_team_kills'] += 1
                else:
                    self.tracked_scores['deaths'] += 1
                    self.tracked_scores['suicides'] += 1

            # KAST flag
            elif attacker_id == self.own_player.id:
                self.tracked_scores['kills'] += 1
                self.temp_scores['kills_this_round'] += 1
                self.temp_scores['kast_triggered'] = 1

            elif victim_id == self.own_player.id:
                self.tracked_scores['deaths'] += 1

        # NOTE: Only considers phases that preceded round win, i.e. excluding reset phase
        elif event_id == EventID.CTRL_MATCH_PHASE_CHANGED:
            new_phase = int(event_data[6])

            if new_phase == GameID.PHASE_RESET:
                t_win = bool(event_data[0])

                win = (t_win and self.own_player.team == GameID.GROUP_TEAM_T) or \
                    ((not t_win) and self.own_player.team == GameID.GROUP_TEAM_CT)

                # Completed matches
                if (
                    (self.session.rounds_won_t + self.session.rounds_won_ct) == (2*Session.ROUNDS_TO_SWITCH) or
                    self.session.rounds_won_t == Session.ROUNDS_TO_WIN or
                    self.session.rounds_won_ct == Session.ROUNDS_TO_WIN
                ):
                    self.tracked_scores['matches'] += 1

                # Completed rounds on a side
                if self.own_player.team == GameID.GROUP_TEAM_T:
                    self.tracked_scores['t_rounds'] += 1
                else:
                    self.tracked_scores['ct_rounds'] += 1

                # Objective success rate
                if self.temp_scores['allies_planted']:
                    self.tracked_scores['afterplant_rounds'] += 1
                    self.tracked_scores['takes'] += 1

                elif self.temp_scores['enemies_planted']:
                    self.tracked_scores['retake_rounds'] += 1

                if win:
                    if self.temp_scores['kills_to_clutch']:
                        self.tracked_scores['clutches'] += 1.
                        self.tracked_scores['clutchkills'] += self.temp_scores['kills_to_clutch']

                    if self.temp_scores['allies_planted']:
                        self.tracked_scores['afterplants'] += 1

                    elif self.temp_scores['enemies_planted']:
                        self.tracked_scores['retakes'] += 1

                    elif self.own_player.team == GameID.GROUP_TEAM_CT:
                        self.tracked_scores['holds'] += 1

                    if self.own_player.team == GameID.GROUP_TEAM_T:
                        self.tracked_scores['t_rounds_won'] += 1

                        if self.session.rounds_won_t == Session.ROUNDS_TO_WIN:
                            self.tracked_scores['matches_won'] += 1

                    else:
                        self.tracked_scores['ct_rounds_won'] += 1

                        if self.session.rounds_won_ct == Session.ROUNDS_TO_WIN:
                            self.tracked_scores['matches_won'] += 1

                    # RWS
                    if self.temp_scores['total_team_damage'] == 0.:
                        damage_contribution = 1. / len(self.session.groups[self.own_player.team])
                    else:
                        damage_contribution = self.temp_scores['damage'] / self.temp_scores['total_team_damage']

                    if self.temp_scores['allies_defused'] or self.temp_scores['allies_planted']:
                        rws = 0.7 * damage_contribution + \
                            (0.3 if self.temp_scores['defused'] or self.temp_scores['planted'] else 0.)
                    else:
                        rws = damage_contribution

                    self.tracked_scores['round_win_shares'] += rws

                # Kills
                if self.temp_scores['kills_this_round'] > 1:
                    self.tracked_scores['multikill_rounds'] += 1
                    self.tracked_scores['multikills'] += self.temp_scores['kills_this_round']

                # KAST
                if self.own_player.health:
                    self.temp_scores['kast_triggered'] = 1

                if self.temp_scores['kast_triggered']:
                    self.tracked_scores['kast_rounds'] += 1

                # Other
                self.tracked_scores['remaining_health'] += self.own_player.health

                # Reset temporary stats
                self.reset_temp()

    def summary(self, soft: bool = True) -> List[Tuple[str, Union[float, int]]]:
        """
        Get the stats associated with own player's activity during the
        running session.

        Some stats are skipped in this print-out, e.g. the numerous item stats,
        but can still be accessed and used externally.

        References for RWS and Rating 2:
        https://support.esea.net/hc/en-us/articles/360008740634-What-is-RWS-
        https://www.hltv.org/news/4094/what-is-that-rating-thing-in-stats
        https://www.hltv.org/news/20695/introducing-rating-20
        https://flashed.gg/posts/reverse-engineering-hltv-rating/
        """

        tracked_scores = self.tracked_scores

        finished_rounds = tracked_scores['t_rounds'] + tracked_scores['ct_rounds'] - \
            (1 if self.session.phase == GameID.PHASE_RESET else 0)

        won_rounds = tracked_scores['t_rounds_won'] + tracked_scores['ct_rounds_won']

        # Using lowest possible (-7) round fraction instead of 1 to soften the drop in stats into the next round
        soft_played_rounds = max(1., finished_rounds + self.session.total_round_time / 180.)
        hard_played_rounds = finished_rounds + 1
        played_rounds = soft_played_rounds if soft else hard_played_rounds

        # Basic stats
        kpr = tracked_scores['kills'] / played_rounds
        apr = tracked_scores['assists'] / played_rounds
        dpr = tracked_scores['deaths'] / played_rounds
        adr = tracked_scores['damage'] / played_rounds

        # Approx. impact and rating 2, KAST, RWS
        impact = 2.14*kpr + 0.42*apr - 0.41
        kast = tracked_scores['kast_rounds'] / played_rounds
        rating2 = 0.0073*kast + 0.3591*kpr - 0.5329*dpr + 0.2372*impact + 0.0032*adr + 0.1587
        rws = tracked_scores['round_win_shares'] / max(1, won_rounds)

        # Utility stats
        util_damage = sum(self.tracked_item_damage[key] for key in (
                GameID.ITEM_FLASH, GameID.ITEM_EXPLOSIVE,
                GameID.ITEM_INCENDIARY_T, GameID.ITEM_INCENDIARY_CT, GameID.ITEM_SMOKE))

        util_use = sum(self.tracked_item_uses[key] for key in (
                GameID.ITEM_FLASH, GameID.ITEM_EXPLOSIVE,
                GameID.ITEM_INCENDIARY_T, GameID.ITEM_INCENDIARY_CT, GameID.ITEM_SMOKE))

        util_damage_per_throw = util_damage / max(1, util_use)
        util_throws_per_round = util_use / played_rounds

        return [
            ('kills', tracked_scores['kills']),
            ('deaths', tracked_scores['deaths']),
            ('assists', tracked_scores['assists']),
            ('kill_death_ratio', tracked_scores['kills'] / max(1., tracked_scores['deaths'])),
            ('kdassist_ratio', (tracked_scores['kills']+tracked_scores['assists']) / max(1., tracked_scores['deaths'])),
            ('kills_per_round', kpr),
            ('deaths_per_round', dpr),
            ('assists_per_round', apr),
            ('avg_damage_per_round', adr),
            ('multikills_per_round', tracked_scores['multikill_rounds'] / max(1, finished_rounds)),
            ('average_multikills', tracked_scores['multikills'] / max(1, tracked_scores['multikill_rounds'])),
            ('util_dmg_per_throw', util_damage_per_throw),
            ('util_throws_per_round', util_throws_per_round),
            ('money_spent_per_kill', max(1., tracked_scores['money_spent'] / max(1, tracked_scores['kills']))),
            ('distance_per_kill', tracked_scores['distance'] / max(1., tracked_scores['kills'])),
            ('average_speed', tracked_scores['distance'] / max(1., tracked_scores['game_time'])),
            ('noise_steps_per_round', tracked_scores['footsteps'] / played_rounds),
            ('health_at_round_end', tracked_scores['remaining_health'] / max(1, finished_rounds)),
            ('messages_per_round', tracked_scores['messages'] / played_rounds),
            ('opening_success', tracked_scores['opening_tries'] / played_rounds),
            ('clutch_success', tracked_scores['clutches'] / max(1, tracked_scores['clutch_rounds'])),
            ('kill_assist_survival_trade', kast),
            ('impact', impact),
            ('rating_2', rating2),
            ('round_win_share', rws),
            ('success_take', tracked_scores['takes'] / max(1, tracked_scores['t_rounds'])),
            ('success_afterplant', tracked_scores['afterplants'] / max(1, tracked_scores['afterplant_rounds'])),
            ('success_t_side', tracked_scores['t_rounds_won'] / max(1, tracked_scores['t_rounds'])),
            ('success_hold', tracked_scores['holds'] / max(1, tracked_scores['ct_rounds'])),
            ('success_retake', tracked_scores['retakes'] / max(1, tracked_scores['retake_rounds'])),
            ('success_ct_side', tracked_scores['ct_rounds_won'] / max(1, tracked_scores['ct_rounds']))]


class FocusTracker:
    """
    Keeps track of an alternative cursor during a replay, storing or redrawing
    its position. Intended to highlight points of interest in a recording or
    to manually label frames with approximate eye focus in real time
    as a recording is replayed.

    NOTE:
    - The recording can be freely paused and sped up (or slowed down).
    - Jumping forward with active labelling will cause some labels to be missed.
    - Jumping backward will not remove labels that were already made.
    """

    MODE_NULL: int = 0
    MODE_READ: int = 1
    MODE_WRITE: int = 2

    def __init__(self, path: str = None, mode: int = MODE_NULL, start_active: bool = False):
        self.recorder = Recorder(file_path=path)
        self.mode = self.MODE_NULL if path is None else mode

        if self.mode == self.MODE_READ:
            self.recorder.read()

        self.icon_inactive = self._load_cursor_image()
        self.icon_active = self.icon_inactive.copy()
        self.icon_inactive[..., :3] = ImageBank.COLOURS['t_cyan']
        self.icon_active[..., :3] = ImageBank.COLOURS['t_red']

        self.y, self.x = Simulation.WORLD_FRAME_CENTRE
        self.active = False if path is None else start_active
        self.hidden = True

    @staticmethod
    def _load_cursor_image() -> np.ndarray:
        """Wrapper around `cv2.imread` to minimise path specification."""

        with open(os.path.join(ASSET_DIR, 'sheets', 'slices.json'), 'r') as f:
            i, j, h, w = json.load(f)['icons']['pointer_cursor']

        icon_sheet = cv2.imread(os.path.join(ASSET_DIR, 'sheets', 'icons.png'), flags=cv2.IMREAD_UNCHANGED)

        return icon_sheet[i:i+h, j:j+w].copy()

    def update(self, yrel: float, xrel: float):
        """Update focal coordinates according to relative movement."""

        self.y = np.clip(self.y + yrel, 2., 141.)
        self.x = np.clip(self.x + xrel, 2., 253.)

    def register(self, window: Union[np.ndarray, None], tick_counter: int):
        """Draw focal cursor and log its position."""

        if self.mode == self.MODE_NULL or (self.mode == self.MODE_READ and (not self.active or self.hidden)):
            return

        if window is not None:
            draw_image(
                window, self.icon_active if self.active else self.icon_inactive, round(self.y)-2, round(self.x)-2)

        # Check for mode and rewind
        if self.mode != self.MODE_WRITE or tick_counter < self.recorder.counter:
            return

        # Update meta
        self.recorder.counter = tick_counter

        data = struct.pack('>2fB', self.y, self.x, int(not self.active))
        self.recorder.append(data)

        # Cache and squeeze records
        self.recorder.cache_chunks()
        self.recorder.squeeze()

    def finish(self, logger: Logger = None):
        """Save logged focal coordinates."""

        if self.mode != self.MODE_WRITE:
            return

        self.recorder.restore_chunks()
        self.recorder.squeeze(all_=True)
        self.recorder.save()

        if logger is not None:
            logger.info("Recording saved to: '%s'.", self.recorder.file_path)

    def get(self, tick_counter: int):
        """
        Get the focal coordinates corresponding to the current replay tick.
        If no correspondance is found (e.g. due to skipped frames when recording
        focus), current coordinates are preserved.
        """

        if self.mode != self.MODE_READ:
            return

        while self.recorder.buffer:
            (_, ctr, _), data = self.recorder.split_meta(self.recorder.buffer[0])

            if ctr > tick_counter:
                break

            elif ctr == tick_counter:
                self.y, self.x, hidden = struct.unpack('>2fB', data)
                self.hidden = bool(hidden)

            del self.recorder.buffer[0]


class PerfMonitor:
    """
    Performance monitoring for ticking processes.
    Monitor for FPS, CPU perc. utilisation, and memory usage.
    """

    def __init__(self, pid: int = None, path: str = None):
        self.data = {
            k: 0. for k in (
                'fps_sum', 'fps_sum2', 'fps_min', 'fps_max', 'fps_avg', 'fps_std',
                'cpu_sum', 'cpu_sum2', 'cpu_min', 'cpu_max', 'cpu_avg', 'cpu_std',
                'mem_sum', 'mem_sum2', 'mem_min', 'mem_max', 'mem_avg', 'mem_std', 'num')}

        self.data['num'] = 0

        self.sum_keys = ('fps_sum', 'cpu_sum', 'mem_sum')
        self.sum2_keys = ('fps_sum2', 'cpu_sum2', 'mem_sum2')
        self.min_keys = ('fps_min', 'cpu_min', 'mem_min')
        self.max_keys = ('fps_max', 'cpu_max', 'mem_max')
        self.avg_keys = ('fps_avg', 'cpu_avg', 'mem_avg')
        self.std_keys = ('fps_std', 'cpu_std', 'mem_std')
        self.data_keys = (self.sum2_keys, self.sum_keys, self.min_keys, self.max_keys)
        self.stat_keys = (self.sum2_keys, self.sum_keys, self.avg_keys, self.std_keys)

        for k in self.min_keys:
            self.data[k] = float('inf')

        self.proc = psutil.Process(os.getpid() if pid is None else pid)
        self.path = path

    def update_data(self, fps: float = 0.):
        """Update monitoring data."""

        cpu_perc = self.proc.cpu_percent()
        mem_res = self.proc.memory_info().rss / 1024**2

        for ksum2, ksum, kmin, kmax, val in zip(*self.data_keys, (fps, cpu_perc, mem_res)):

            # Unrolled std components
            self.data[ksum2] += val**2
            self.data[ksum] += val

            # Peaks/drops
            if val < self.data[kmin] and val != 0.:
                self.data[kmin] = val

            if val > self.data[kmax]:
                self.data[kmax] = val

        self.data['num'] += 1

    def update_stats(self):
        """Infer mean and standard deviation per monitored quantity."""

        num = self.data['num']

        if not num:
            return

        for ksum2, ksum, kavg, kstd in zip(*self.stat_keys):
            self.data[kavg] = self.data[ksum] / num
            self.data[kstd] = self.get_std(self.data[ksum2], self.data[ksum], self.data[kavg], num)

    @staticmethod
    def get_std(sum2: float, sum1: float, avg: float, num: int) -> float:
        """Infer standard deviation from unrolled term."""

        return max(0., (sum2 - 2*sum1*avg + num*avg**2) / num)**0.5

    def save(self):
        """Save monitoring stats and data."""

        if self.path is None:
            return

        if os.path.exists(self.path):
            with open(self.path, 'r') as data_file:
                data = json.load(data_file)

        else:
            data = {}

        data[asctime()] = self.data

        with open(self.path, 'w') as data_file:
            json.dump(data, data_file)
