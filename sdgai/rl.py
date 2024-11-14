"""Train a SDG AI agent with reinforcement learning."""

from argparse import ArgumentParser, Namespace
from collections import deque
from random import Random

import numpy as np
import torch
from torch import nn, Tensor

from discit.distr import MultiCategorical, FixedVarNormal
from discit.func import symexp
from discit.optim import NAdamW, LRScheduler
from discit.rl import ActorCritic, PPO
from discit.track import CheckpointTracker

from sidegame.assets import ImageBank, SoundBank
from sidegame.audio import get_mel_basis, spectrify
from sidegame.networking import Action, Entry, Event
from sidegame.utils import get_logger
from sidegame.game import GameID, EventID
from sidegame.game.shared import Inventory, Player, Session
from sidegame.game.client.simulation import Simulation
from sidegame.game.client.tracking import StatTracker
from sidegame.game.server import SDGServer

from sdgai.actor import SDGBaseActor, prepare_inputs, SAMPLING_RATE, HRIR_LEN


# ------------------------------------------------------------------------------
# MARK: RewardMap

REWARD_MAP = {
    'round_won': 1.,
    'match_won': 5.,
    'dist_diff': 0.001,
    'damage_pts': 0.0025,
    'kills': 0.2,
    'ownkills': -1.,
    'death': -0.2,
    'planted': 0.5,
    'defused': 1.}


# ------------------------------------------------------------------------------
# MARK: ObservationSpace

OBS_AUDIO_SIZE = 128    # 2 channels * 64 mels
OBS_MKBD_SIZE = 20      # 14 bin. btn., 1 num. btn., 1 m. wheel, 2 m. mvmt., 2 m. pos.
OBS_AUX_SIZE = 16       # T wins, CT wins, team id., kills, deaths, health, money, 2 pos., 7 items held

OBS_IMG_SHAPE = (3, 144, 256)
OBS_VEC_SIZE = OBS_AUDIO_SIZE + OBS_MKBD_SIZE
STATE_VEC_SIZE = OBS_VEC_SIZE + OBS_AUX_SIZE


# ------------------------------------------------------------------------------
# MARK: ActionSpace

# Mouse movement
ACT_MOUSE_MVMT = tuple([(v,) for v in SDGBaseActor.MOUSE_BINS[3:-3]])

# Mouse buttons
ACT_MOUSE_BTN = (
    (0., 0.),   # None
    (1., 0.),   # Left click | Fire
    (0., 1.),   # Right click | Walk
    (1., 1.))   # Both

# Keyboard buttons
# Interaction
ACT_KBD_SEGR = (
    (0., 0., 0., 0.),   # None
    (1., 0., 0., 0.),   # S | Message
    (0., 1., 0., 0.),   # E | Use
    (0., 0., 1., 0.),   # G | Drop
    (0., 0., 0., 1.))   # R | Reload

# Movement
ACT_KBD_DAWS = (
    (0., 0., 0., 0.),   # Idle
    (1., 0., 0., 0.),   # D | Right
    (0., 1., 0., 0.),   # A | Left
    (0., 0., 1., 0.),   # W | Fwd.
    (0., 0., 0., 1.),   # S | Back
    (1., 0., 1., 0.),   # W+D | Fwd. + right
    (0., 1., 1., 0.),   # W+A | Fwd. + left
    (1., 0., 0., 1.),   # S+D | Back + right
    (0., 1., 0., 1.))   # S+A | Back + left

# View
ACT_KBD_TXCB = (
    (0., 0., 0., 0.),   # World
    (1., 0., 0., 0.),   # Tab | Map/stats
    (0., 1., 0., 0.),   # X | Com. wheel (terms)
    (0., 0., 1., 0.),   # C | Com. wheel (items)
    (0., 0., 0., 1.))   # B | Buy menu

# Item
ACT_KBD_NUM = (
    (0., 0., 0., 0., 0.),   # None
    (1., 0., 0., 0., 0.),   # 1 | Main
    (0., 1., 0., 0., 0.),   # 2 | Pistol
    (0., 0., 1., 0., 0.),   # 3 | Knife
    (0., 0., 0., 1., 0.),   # 4 | C4
    (0., 0., 0., 0., 1.))   # 5 | Cycle utils.

# 67 = 4 + 5 + 9 + 5 + 6 + 19 + 19
ACT_VALUES = ACT_MOUSE_BTN, ACT_KBD_SEGR, ACT_KBD_DAWS, ACT_KBD_TXCB, ACT_KBD_NUM, ACT_MOUSE_MVMT, ACT_MOUSE_MVMT
ACT_SPLIT = tuple([len(v) for v in ACT_VALUES])
ACT_SIZE = sum(ACT_SPLIT)


# ------------------------------------------------------------------------------
# MARK: VisEncoder

class VisEncoder(nn.Module):
    CHN_SIZES = (OBS_IMG_SHAPE[0], 32, 192, 576)
    OUT_SIZE = CHN_SIZES[-1]

    def __init__(self):
        super().__init__()

        self.activ = nn.CELU(0.1)

        # 256x144x3 -> 64x36x32
        self.conv0 = nn.Conv2d(self.CHN_SIZES[0], self.CHN_SIZES[1], 4, stride=4)

        # 64x36x32 -> 16x12x192
        self.conv1 = nn.Conv2d(self.CHN_SIZES[1], self.CHN_SIZES[2], (3, 4), stride=(3, 4))

        # 16x12x192 -> 4x3x576
        self.conv2 = nn.Conv2d(self.CHN_SIZES[2], self.CHN_SIZES[3], 4, stride=4)

        # 4x3x576 -> 1x1x576
        self.conv3 = nn.Conv2d(self.CHN_SIZES[3], self.OUT_SIZE, (3, 4))

    def forward(self, x: Tensor) -> Tensor:
        x = self.activ(self.conv0(x))
        x = self.activ(self.conv1(x))
        x = self.activ(self.conv2(x))
        x = self.conv3(x)

        return x.flatten(1)


# ------------------------------------------------------------------------------
# MARK: RNNBase

class RNNBase(nn.Module):
    def __init__(self, ipt_size: int, enc_size: int, mem_size: int, out_size: int):
        super().__init__()

        self.fcin = nn.Linear(ipt_size, enc_size)
        self.activ = nn.Tanh()

        self.rnn = nn.GRUCell(enc_size, mem_size)
        self.mem = nn.Parameter(torch.zeros(1, mem_size).uniform_(-1., 1.))

        self.fcout = nn.Linear(mem_size, out_size)

    def forward(self, x: Tensor, mem: Tensor) -> tuple[Tensor, Tensor]:
        x = self.activ(self.fcin(x))
        mem = self.rnn(x, mem)
        x = self.fcout(mem)

        return x, mem


# ------------------------------------------------------------------------------
# MARK: SDGNet

class SDGNet(ActorCritic):
    ENC_SIZE = 512
    MEM_SIZE = 256

    def __init__(self):
        super().__init__()

        # 256x144x3 -> 576
        self.visencoder = VisEncoder()

        # 576 + 148 -> 512 -> 256 -> 67
        self.policy = RNNBase(VisEncoder.OUT_SIZE + OBS_VEC_SIZE, self.ENC_SIZE, self.MEM_SIZE, ACT_SIZE)

        self.act_value_tpl = nn.ParameterList([nn.Parameter(torch.tensor(v), requires_grad=False) for v in ACT_VALUES])

        # 576 + 164 -> 512 -> 256 -> 1
        self.valuator = RNNBase(VisEncoder.OUT_SIZE + STATE_VEC_SIZE, self.ENC_SIZE, self.MEM_SIZE, 1)

    def init_mem(self, n_actors: int = 1, n_envs: int = None) -> tuple[Tensor, Tensor]:
        mem_p = self.policy.mem.detach().expand(n_actors, -1).clone()
        mem_v = self.valuator.mem.detach().expand(n_actors, -1).clone()

        return mem_p, mem_v

    def reset_mem(
        self,
        mem: tuple[Tensor, Tensor],
        nonreset_mask: Tensor
    ) -> tuple[Tensor, Tensor]:

        mem_p, mem_v = mem

        mem_p = torch.lerp(self.policy.mem, mem_p, nonreset_mask)
        mem_v = torch.lerp(self.valuator.mem, mem_v, nonreset_mask)

        return mem_p, mem_v

    def unwrap_sample(self, sample: tuple[Tensor, ...], aux: None) -> tuple[Tensor]:
        return sample[0],

    def get_distr(self, args: Tensor | tuple[Tensor, ...], from_raw: bool = False) -> MultiCategorical:
        if from_raw:
            logits = args
            mcat = MultiCategorical.from_raw(self.act_value_tpl, logits, logits_are_split=False, joint_prob=False)

        else:
            log_prob_tpl = args
            mcat = MultiCategorical(self.act_value_tpl, log_prob_tpl, joint_prob=False)

        return mcat

    def act(
        self,
        obs: tuple[Tensor, ...],
        mem: tuple[Tensor, ...],
        sample: bool = None
    ) -> tuple[Tensor, tuple[Tensor, ...]]:

        obs_img, obs_vec, _ = obs
        mem_p, mem_v = mem

        obs_img = self.visencoder.forward(obs_img)
        obs_p = torch.cat((obs_img, obs_vec), dim=-1)

        p, mem_p = self.policy(obs_p, mem_p)

        p = self.get_distr(p, from_raw=True)
        a = p.sample()[0] if sample else p.mode

        return a, (mem_p, mem_v)

    def collect(
        self,
        obs: tuple[Tensor, ...],
        mem: tuple[Tensor, ...],
        sample: tuple[Tensor, ...] = None
    ) -> tuple[dict[str, Tensor | tuple[Tensor, ...]], None, tuple[Tensor, Tensor]]:

        if sample is None:
            obs_img, obs_vec, obs_aux = obs
            obs_img = self.visencoder.forward(obs_img)

            obs_p = torch.cat((obs_img, obs_vec), dim=-1)
            obs_v = torch.cat((obs_p, obs_aux), dim=-1)
            obs = obs_p, obs_v

        obs_p, obs_v = obs
        mem_p, mem_v = mem

        p, mem_p = self.policy(obs_p, mem_p)
        v, mem_v = self.valuator(obs_v, mem_v)

        p = self.get_distr(p, from_raw=True)
        v = FixedVarNormal(symexp(v)).mean

        if sample is None:
            sample = p.sample()

        data = {
            'act': sample,
            'args': p.args,
            'val': v,
            'mem': mem,
            'obs': obs}

        return data, None, (mem_p, mem_v)

    def forward(
        self,
        obs: tuple[Tensor, ...],
        mem: tuple[Tensor, ...],
        sample: tuple[Tensor, ...] = None,
        detach: bool = False
    ) -> dict[str, MultiCategorical | FixedVarNormal | tuple[Tensor, ...]]:

        obs_p, obs_v = obs
        mem_p, mem_v = mem

        p, mem_p = self.policy(obs_p, mem_p)
        v, mem_v = self.valuator(obs_v, mem_v)

        act = self.get_distr(p, from_raw=True)
        val = FixedVarNormal(symexp(v))

        return {
            'act': act,
            'val': val,
            'aux': (),
            'mem': (mem_p, mem_v)}


# ------------------------------------------------------------------------------
# MARK: SDGSyncActor

class SDGSyncActor(SDGBaseActor):
    """Actor client without initialised networking."""

    def __init__(self, session: Session, sim: Simulation, name: str = 'ai00'):
        self.session = session
        self.sim = sim
        self.name = name

        self.entities = session.players
        self.own_entity = session.players[sim.own_player_id]
        self.own_entity.name = name
        self.stats = StatTracker(session, self.own_entity)

        self.rng = session.rng
        self.rng_py = Random(self.rng.random())

        self.value = 0.
        self._clock_diff_tracker = self

        self.action_skip = 0
        self.focus = None

        self.player = self.own_entity
        self.observations = deque()
        self.actions = deque()
        self.reset_action_state()

        self.last_scores = {k: 0 for k in REWARD_MAP}
        self.last_scores['dist_diff'] = -1.
        self.site_x_centre = (session.map.site_a_centre[0] + session.map.site_b_centre[0]) / 2.

    def get_reward(self) -> float:
        scores = self.stats.tracked_scores
        last_scores = self.last_scores
        reward = 0.

        n_rounds_won = scores['ct_rounds_won'] + scores['t_rounds_won']
        diff_to_sites = abs(self.player.pos[0] - self.site_x_centre)

        if n_rounds_won > last_scores['round_won']:
            reward += REWARD_MAP['round_won']
            last_scores['round_won'] = n_rounds_won

        if scores['matches_won'] > last_scores['match_won']:
            reward += REWARD_MAP['match_won']
            last_scores['match_won'] = scores['matches_won']

        if last_scores['dist_diff'] < 0.:
            last_scores['dist_diff'] = diff_to_sites

        elif diff_to_sites != last_scores['dist_diff']:
            reward += REWARD_MAP['dist_diff'] * (last_scores['dist_diff'] - diff_to_sites)
            last_scores['dist_diff'] = diff_to_sites

        if scores['damage'] > last_scores['damage_pts']:
            reward += REWARD_MAP['damage_pts'] * (scores['damage'] - last_scores['damage_pts'])
            last_scores['damage_pts'] = scores['damage']

        if scores['kills'] > last_scores['kills']:
            reward += REWARD_MAP['kills'] * (scores['kills'] - last_scores['kills'])
            last_scores['kills'] = scores['kills']

        if scores['own_team_kills'] > last_scores['ownkills']:
            reward += REWARD_MAP['ownkills'] * (scores['own_team_kills'] - last_scores['ownkills'])
            last_scores['ownkills'] = scores['own_team_kills']

        if scores['deaths'] > last_scores['death']:
            reward += REWARD_MAP['death']
            last_scores['death'] = scores['deaths']

        if scores['plants'] > last_scores['planted']:
            reward += REWARD_MAP['planted']
            last_scores['planted'] = scores['plants']

        if scores['defuses'] > last_scores['defused']:
            reward += REWARD_MAP['defused']
            last_scores['defused'] = scores['defuses']

        return reward


# ------------------------------------------------------------------------------
# MARK: SDGSyncEnv

class SDGSyncEnv:
    """
    Lock-step simulation replacing both server and clients for AI actors.

    Agents are all directly updated in a local process in fixed-time iterations,
    avoiding the issues and dynamic conditions of real-time networking.

    The process can thus be simplified, without time constraints,
    client-side prediction, entity interpolation, or server-side lag compensation.
    """

    ENV_ID = SDGServer.ENV_ID

    def __init__(self, n_agents: int, tick_rate: float, frame_skip: int, rng: np.random.Generator = None, idx: int = 0):
        self.n_agents = n_agents
        self.updates_per_step = frame_skip + 1
        self.dt_srv = 1. / tick_rate
        self.dt_act = self.dt_srv / self.updates_per_step
        rng = np.random.default_rng(rng)

        self.session = Session(rng=rng)
        images = ImageBank()
        sounds = SoundBank(tick_rate)
        inventory = Inventory(images, sounds.item_sounds)
        assets = (images, sounds, inventory)

        self.actors: list[SDGSyncActor] = []

        for agent_idx in range(n_agents):
            self.session.add_player(Player(agent_idx, inventory, rng=rng))
            sim = Simulation(agent_idx, tick_rate, session=self.session, assets=assets)

            self.actors.append(SDGSyncActor(self.session, sim, name=f'e{idx}a{agent_idx}'))

        self.entities = self.session.players

        self.counter = 0
        self.timestamp = 0.
        self.queued_events: deque[Event] = deque()

        self.reset = True
        self.start()

    def start(self):
        self.session.assigned_teams = {
            player.id: (GameID.GROUP_TEAM_CT if i % 2 else GameID.GROUP_TEAM_T)
            for i, player in self.session.players.items()}

        self.session.update(self.dt_srv, self.queued_events, flag=GameID.CMD_START_MATCH)

        for actor in self.actors:
            actor.reset_action_state()
            actor.last_scores['dist_diff'] = -1.

    def step(self):
        self.reset = False
        session = self.session

        # Apply inputs
        for actor in self.actors:
            actor.stats.update_from_state(actor.player.pos.copy(), self.timestamp)

            state, log = actor.poll_user_input(self.timestamp)
            action = Action(Action.TYPE_STATE, self.counter, self.timestamp, state)

            if actor.player.health:
                self.queued_events.extend(
                    actor.player.update(
                        action,
                        session.players, session.objects, session.c4, session.map,
                        self.timestamp, 0., session.phase == GameID.PHASE_BUY))

            if log is None:
                continue

            log = Action(Action.TYPE_LOG, self.counter, self.timestamp, log)
            SDGServer.handle_user_log(self, log, self.queued_events)

        # Update global state
        event_idx = 0

        for _ in range(self.updates_per_step):
            event_idx = self.session.update(self.dt_srv, self.queued_events, event_idx=event_idx)
            self.timestamp += self.dt_srv
            self.counter += 1

        # Sync. local states
        while self.queued_events:
            event = self.queued_events.popleft()

            if event.type == EventID.CTRL_MATCH_ENDED:
                self.reset = True
                self.start()

            elif event.type in (EventID.CTRL_PLAYER_MOVED, EventID.OBJECT_SPAWN, EventID.OBJECT_EXPIRE):
                continue

            log_id, log_data = SDGServer.create_global_log(self, event)
            log = Entry(log_id, Entry.TYPE_LOG, self.counter, self.timestamp, log_data)

            for actor in self.actors:
                actor.handle_log(log, self.timestamp, predicting_state=False)

    def generate_outputs(self) -> tuple[list[np.ndarray], list[np.ndarray], list[list[float]], list[list[float]], bool]:
        session = self.session
        rounds_won_t = session.rounds_won_t
        rounds_won_ct = session.rounds_won_ct

        n_rounds = rounds_won_t + rounds_won_ct + int(self.session.phase != GameID.PHASE_RESET)

        max_pos_val = max(session.map.bounds) / 2
        max_item_id = GameID.ITEM_SNIPER

        frames, sounds, mkbds, states = [], [], [], []

        for actor in self.actors:
            sim = actor.sim
            player = actor.player

            sim.eval_effects(self.dt_act)
            frames.append(sim.get_frame())
            sounds.append(sim.get_sound())
            mkbds.append(actor.mkbd_state)

            state = [
                rounds_won_t / Session.ROUNDS_TO_WIN,
                rounds_won_ct / Session.ROUNDS_TO_WIN,
                float(player.team == GameID.GROUP_TEAM_CT),
                player.kills / n_rounds,
                player.deaths / n_rounds,
                player.health / player.HEALTH_AT_ROUND_START,
                player.money * 3 / player.MONEY_CAP,
                player.pos[0] / max_pos_val - 1.,
                player.pos[1] / max_pos_val - 1.,
                0. if player.slots[0] is None else 1.,
                0. if player.slots[1] is None else player.slots[1].id / max_item_id]

            state += [float(slot is not None) for slot in player.slots[4:]]

            states.append(state)

        return frames, sounds, mkbds, states, self.reset


# ------------------------------------------------------------------------------
# MARK: SDGSyncRunner

class SDGSyncRunner:
    """CLI launcher for training and evaluation."""

    EMPTY_DICT = {}

    def __init__(self, args: Namespace):
        self.test: bool = args.test
        self.sampling_prob: float = args.sampling_prob

        time_per_round = Session.TIME_TO_BUY + Session.TIME_TO_PLANT + Session.TIME_TO_DEFUSE + Session.TIME_TO_RESET
        time_per_match = 2 * Session.ROUNDS_TO_SWITCH * time_per_round

        self.n_matches_to_complete: int = args.n_matches
        self.n_steps = int(args.n_matches * time_per_match * args.tick_rate / ((args.frame_skip + 1) * args.n_envs))

        self.n_envs: int = args.n_envs
        self.n_agents_per_env: int = args.n_agents_per_env
        self.n_all_agents = self.n_agents_per_env * self.n_envs

        self.ckpter = CheckpointTracker(args.model_name, args.data_dir, args.device, args.seed)

        self.envs = [
            SDGSyncEnv(args.n_agents_per_env, args.tick_rate, args.frame_skip, self.ckpter.rng, i)
            for i in range(self.n_envs)]

        self.mel_basis = get_mel_basis(SAMPLING_RATE)
        self.ham_window: np.ndarray = np.hamming(int(SAMPLING_RATE/args.tick_rate + (HRIR_LEN-1)*2))
        self.db_ref = self.ham_window.sum()**2 / 2.
        self.reward_map = np.array(REWARD_MAP.values())

        self.logger = get_logger('AITester' if self.test else 'AITrainer', path=args.logging_path)

    @classmethod
    def from_args(cls):
        parser = ArgumentParser(description='Argument parser for the SDG RL runner.')

        parser.add_argument(
            '--test', action='store_true',
            help='Evaluate the named model at its latest checkpoint instead of training it (further).')
        parser.add_argument(
            '--n_matches', type=int, default=100,
            help='Matches to complete before ending the process.')
        parser.add_argument(
            '--n_envs', type=int, default=10,
            help='Number of matches to run in parallel.')
        parser.add_argument(
            '--n_agents_per_env', type=int, default=10,
            help='Number of agent copies to spawn per match.')

        parser.add_argument(
            '--data_dir', type=str, default='models',
            help='Path to the directory where model checkpoints will be saved.')
        parser.add_argument(
            '--model_name', type=str, required=True,
            help='Name under which model checkpoints and logs will be saved.')
        parser.add_argument(
            '--device', type=str, default='cuda',
            help='Processing device used to run the model and store data for training or evaluation.')
        parser.add_argument(
            '--seed', type=int, default=42,
            help='Seed for initialising random number generators.')

        parser.add_argument(
            '--tick_rate', type=float, default=60.,
            help='Rate of updating the local game state in ticks (frames) per second.')
        parser.add_argument(
            '--frame_skip', type=int, default=1,
            help='Number of env. ticks to skip before generating data for inference and applying actions.')
        parser.add_argument(
            '--sampling_prob', type=float, default=1.,
            help='Probability of sampling to get actions from probabilities instead of argmax. '
            '0 corresponds to argmax and 1 to sampling on every step.')

        parser.add_argument(
            '--logging_path', type=str, default=None,
            help='If given, execution logs are written to a file at the specified location instead of stdout.')

        return cls(parser.parse_args())

    def run(self):
        try:
            self.logger.info('Launching...')

            self.eval() if self.test else self.train()

            self.logger.info('Done.')

        except KeyboardInterrupt:
            self.logger.info('Interrupted.')

    def train(self):
        model = SDGNet()
        optimizer = NAdamW((param for param in model.parameters() if param.requires_grad), device=self.ckpter.device)

        model.to(self.ckpter.device)
        self.ckpter.load_model(model, optimizer)

        scheduler = LRScheduler(optimizer, starting_step=self.ckpter.meta['update_step'])

        rl_algo = PPO(
            self.step,
            self.ckpter,
            scheduler,
            self.n_all_agents,
            self.n_steps,
            n_rollout_steps=256,
            n_truncated_steps=30,
            n_passes_per_step=10)

        try:
            rl_algo.run()

        except KeyboardInterrupt:
            rl_algo.writer.close()
            raise

    def eval(self):
        model = SDGNet()

        model.to(self.ckpter.device)
        self.ckpter.load_model(model)

        mem = model.init_mem(self.n_all_agents)

        with torch.inference_mode():
            obs = self.step()[0]
            n_matches_completed = 0

            while n_matches_completed < self.n_matches_to_complete:
                if self.sampling_prob:
                    sample = self.ckpter.rng.random() < self.sampling_prob

                else:
                    sample = False

                actions, mem = model.act(obs, mem, sample)
                obs, data, _ = self.step(actions)

                if not data['nrst'].all().item():
                    mem = model.reset_mem(mem, data['nrst'])

                    for env in self.envs:
                        if env.reset:
                            n_matches_completed += 1

                    self.logger.info('Completed %d of %d matches.', n_matches_completed, self.n_matches_to_complete)

        for env in self.envs:
            for actor in env.actors:
                path_to_stats = actor.stats.save()

                self.logger.info('Stats for actor %s saved to: %s', actor.name.upper(), path_to_stats)

    def step(
        self,
        actions: Tensor = None
    ) -> tuple[tuple[Tensor, ...], dict[str, Tensor], dict[str, float]]:

        # Apply actions
        if actions is not None:
            actions = actions.reshape(self.n_envs, self.n_agents_per_env, -1).cpu().numpy()

            for env, acts in zip(self.envs, actions):
                for actor, act in zip(env.actors, acts):
                    mkbd, mmoty, mmotx, mwhl = act[:19].astype(np.int64), act[19], act[20], 0

                    actor.actions.append((*actor.cutout_centre_indices, mkbd, mmoty, mmotx, mwhl))

        # Update env. states
        for env in self.envs:
            env.step()

        # Render graphics and gather observations
        all_frames, all_sounds, all_mkbds, all_states, resets = [], [], [], [], []

        for env in self.envs:
            frames, sounds, mkbds, states, rst = env.generate_outputs()

            all_frames.extend(frames)
            all_sounds.extend(sounds)
            all_mkbds.extend(mkbds)
            all_states.extend(states)
            resets.append(rst)

        # Batch observations
        frames = np.stack(all_frames)
        sounds = np.stack(all_sounds)
        mkbds = np.stack(all_mkbds)
        states = np.stack(all_states)
        resets = np.array(resets)

        # Preprocess
        spectra = spectrify(sounds, self.mel_basis, self.ham_window, SAMPLING_RATE, ref=self.db_ref)

        obs_img, obs_snd, obs_mkbd, obs_aux = prepare_inputs(frames, spectra, mkbds, states, self.ckpter.device)
        obs_vec = torch.cat((obs_snd.flatten(1), obs_mkbd), dim=1)
        obs_aux = (obs_aux + 1.).log_()

        # Interleave resets
        nrst_mask = np.repeat(~resets, self.n_agents_per_env)[:, None]
        nrst_mask = torch.from_numpy(nrst_mask).to(self.ckpter.device, dtype=torch.float32)

        # Get reward from tracked stats
        rewards = torch.tensor(
            [actor.get_reward() for env in self.envs for actor in env.actors],
            dtype=torch.float32,
            device=self.ckpter.device)

        return (obs_img, obs_vec, obs_aux), {'rwd': rewards, 'nrst': nrst_mask}, self.EMPTY_DICT


if __name__ == '__main__':
    runner = SDGSyncRunner.from_args()
    runner.run()
