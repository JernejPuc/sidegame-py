"""Extraction of success-oriented values from the game state in SDG."""

import os
import json
from time import asctime
from typing import List, Tuple, Union
import numpy as np
from sidegame.game.shared import GameID, Event, Map, Player, Session


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
            'assists': 0,
            'own_team_kills': 0,
            'suicides': 0,
            'multikill_rounds': 0,
            'multikills': 0,
            'clutch_rounds': 0,
            'clutches': 0,
            'clutchkills': 0,
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
            't_wins': 0,
            'ct_wins': 0,
            'kast_rounds': 0,
            'round_win_shares': 0.,
            'openings': 0,
            'opening_tries': 0}

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

    def update_from_state(self, pos: float, timestamp: float):
        """
        Update uptime, distance travelled, and heatmap of positions
        (time spent at rounded location).
        """

        if self.own_player.team == GameID.GROUP_SPECTATORS:
            return

        dt = 0. if self.last_timestamp is None else (timestamp - self.last_timestamp)
        self.last_timestamp = timestamp

        if self.session.phase != GameID.PHASE_BUY and self.own_player.health:
            self.tracked_heatmap[self.own_player.get_position_indices(self.own_player.pos / 10.)] += dt
            self.tracked_scores['game_time'] += dt

            self.tracked_scores['distance'] += np.linalg.norm(pos - self.last_pos) if self.last_pos is not None else 0.
            self.last_pos = pos

    def update_from_event(self, event_src: int, event_id: int, event_data: List[Union[int, float]], timestamp: float):
        """Update stats wrt. in-game event."""

        if self.own_player.team == GameID.GROUP_SPECTATORS:
            return

        # Messaging
        elif event_src != Map.PLAYER_ID_NULL:
            sender_position_id = event_data[-1]

            if sender_position_id == self.own_player.position_id:
                self.tracked_scores['messages'] += 1

            return

        # Money spent on self-assignment
        if event_id == Event.OBJECT_ASSIGN:
            assignee_id = int(event_data[0])
            obj_item_id = event_data[-2]

            carrying = int(event_data[4])
            spending = int(event_data[6])

            if carrying and spending and assignee_id == self.own_player.id:
                item = self.own_player.inventory.get_item_by_id(obj_item_id)
                self.tracked_scores['money_spent'] += spending
                self.tracked_item_buys[item.id] += 1

        # Flags for objective success rate
        elif event_id == Event.C4_PLANTED:
            planter_id = int(event_data[0])

            if planter_id in self.session.groups[self.own_player.team] and self.session.phase == GameID.PHASE_PLANT:
                self.temp_scores['allies_planted'] = 1

                if planter_id == self.own_player.id:
                    self.temp_scores['planted'] = 1
            else:
                self.temp_scores['enemies_planted'] = 1

        elif event_id == Event.C4_DEFUSED:
            defuser_id = int(event_data[0])

            # Flag for RWS
            if defuser_id in self.session.groups[self.own_player.team]:
                self.temp_scores['allies_defused'] = 1

                if defuser_id == self.own_player.id:
                    self.temp_scores['defused'] = 1

        elif event_id == Event.PLAYER_DAMAGE:
            attacker_id = int(event_data[0])
            victim_id = int(event_data[1])
            damage = event_data[2]
            item_id = event_data[-2]

            attacker = self.session.players[attacker_id]
            victim = self.session.players[victim_id]

            if victim.team != self.own_player.team:
                # RWS data
                if attacker.team == self.own_player.team:
                    self.temp_scores['total_team_damage'] += damage

                # Other damage-associated data
                if attacker_id == self.own_player.id:
                    self.temp_scores['damage'] += damage
                    self.tracked_scores['damage'] += damage
                    self.temp_player_damage[victim.position_id] += damage
                    self.temp_last_contact_time[victim.position_id] = timestamp
                    self.tracked_item_damage[item_id] += damage

        elif event_id == Event.FX_ATTACK:
            attacker_id = int(event_data[0])
            item_id = event_data[-2]

            if attacker_id == self.own_player.id:
                self.tracked_item_uses[item_id] += 1

        elif event_id == Event.FX_FOOTSTEP:
            player_id = int(event_data[0])

            if player_id == self.own_player.id:
                self.tracked_scores['footsteps'] += 1

        # Keep track of flash assists, count debuff as flash damage
        elif event_id == Event.FX_FLASH:
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

        elif event_id == Event.PLAYER_DEATH:
            attacker_id = int(event_data[0])
            victim_id = int(event_data[1])

            enemy_team = GameID.GROUP_TEAM_T if self.own_player.team == GameID.GROUP_TEAM_CT else GameID.GROUP_TEAM_CT
            alive_ts = [player.id for player in self.session.groups[self.own_player.team].values() if player.health]
            alive_cts = [player.id for player in self.session.groups[enemy_team].values() if player.health]

            attacker = self.session.players[attacker_id]
            victim = self.session.players[victim_id]

            # Clutch flag
            if len(alive_ts) == 1 and alive_ts[0] == self.own_player.id and not self.temp_scores['kills_to_clutch']:
                self.temp_scores['kills_to_clutch'] = alive_cts
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

            else:
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
                        self.tracked_scores['suicides'] += 1

                # KAST flag
                elif attacker_id == self.own_player.id:
                    self.temp_scores['kast_triggered'] = 1

        # NOTE: Only considers phases that preceded round win, i.e. excluding reset phase
        elif event_id == Event.CTRL_MATCH_PHASE_CHANGED:
            new_phase = int(event_data[6])

            if new_phase == GameID.PHASE_RESET:
                t_win = bool(event_data[0])

                win = (t_win and self.own_player.team == GameID.GROUP_TEAM_T) or \
                    (not t_win and self.own_player.team == GameID.GROUP_TEAM_CT)

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
                        self.tracked_scores['t_wins'] += 1
                    else:
                        self.tracked_scores['ct_wins'] += 1

                    # RWS
                    if self.temp_scores['allies_defused'] or self.temp_scores['allies_planted']:
                        rws = 0.7 * self.temp_scores['damage'] / max(1., self.temp_scores['total_team_damage']) + \
                            (0.3 if self.temp_scores['defused'] or self.temp_scores['planted'] else 0.)
                    else:
                        rws = self.temp_scores['damage'] / max(1., self.temp_scores['total_team_damage'])

                    self.tracked_scores['round_win_shares'] += rws

                # Kills
                if self.temp_scores['kills_this_round'] > 1:
                    self.tracked_scores['multikill_rounds'] += 1.
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

        own_player = self.own_player
        tracked_scores = self.tracked_scores

        finished_rounds = self.session.rounds_won_t + self.session.rounds_won_ct - \
            (1 if self.session.phase == GameID.PHASE_RESET else 0)

        # Using lowest possible (-7) round fraction instead of 1 to soften the drop in stats into the next round
        soft_played_rounds = max(1., finished_rounds + self.session.total_round_time / 180.)
        hard_played_rounds = finished_rounds + 1
        played_rounds = soft_played_rounds if soft else hard_played_rounds

        # Basic stats
        kpr = own_player.kills / played_rounds
        apr = tracked_scores['assists'] / played_rounds
        dpr = own_player.deaths / played_rounds
        adr = tracked_scores['damage'] / played_rounds

        # Approx. impact and rating 2, KAST, RWS
        impact = 2.14*kpr + 0.42*apr - 0.41
        kast = tracked_scores['kast_rounds'] / played_rounds
        rating2 = 0.0073*kast + 0.3591*kpr - 0.5329*dpr + 0.2372*impact + 0.0032*adr + 0.1587
        rws = tracked_scores['round_win_shares'] / max(1, finished_rounds)

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
            ('kills', own_player.kills),
            ('deaths', own_player.deaths),
            ('assists', tracked_scores['assists']),
            ('kill_death_ratio', own_player.kills / max(1., own_player.deaths)),
            ('kdassist_ratio', (own_player.kills + tracked_scores['assists']) / max(1., own_player.deaths)),
            ('kills_per_round', kpr),
            ('deaths_per_round', dpr),
            ('assists_per_round', apr),
            ('avg_damage_per_round', adr),
            ('multikills_per_round', tracked_scores['multikill_rounds'] / max(1, finished_rounds)),
            ('average_multikills', tracked_scores['multikills'] / max(1, tracked_scores['multikill_rounds'])),
            ('util_dmg_per_throw', util_damage_per_throw),
            ('util_throws_per_round', util_throws_per_round),
            ('money_spent_per_kill', max(1., tracked_scores['money_spent'] / max(1, own_player.kills))),
            ('distance_per_kill', tracked_scores['distance'] / max(1., own_player.kills)),
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
            ('success_t_side', tracked_scores['t_wins'] / max(1, tracked_scores['t_rounds'])),
            ('success_hold', tracked_scores['holds'] / max(1, tracked_scores['ct_rounds'])),
            ('success_retake', tracked_scores['retakes'] / max(1, tracked_scores['retake_rounds'])),
            ('success_ct_side', tracked_scores['ct_wins'] / max(1, tracked_scores['ct_rounds']))]