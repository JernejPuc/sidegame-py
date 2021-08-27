"""
An attempt of reinforcement learning in SDG. The example is local and requires
at least further optimisations to be practically usable.
"""

import os
import argparse
from collections import deque
from typing import Deque, Dict, List, Hashable, Tuple, Union
from time import sleep, perf_counter
from datetime import timedelta
import threading
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from sdgai.model import PCNet
from sdgai.actor import SDGRemoteActor, logits_to_pi
from sidegame.networking.core import get_logger


INTERRUPT_CHECK_PERIOD = 1.
MAIN_RANK = 0
RANK_DELAY = 0.05
MAX_DISP_SECONDS = 99*24*3600


class Inferent:
    """
    Groups inference processes tied to the same device. Can be used for
    reinforcement learning with multiple actor instances per agent model
    or evaluation of singular actors or entire teams.

    Uses a semi-simultaneous pool of workers to issue calls for model inference.
    Each worker loops and waits until enough data (observations from different
    actors, wrt. their keys) is available to create a batch. Observations are
    handled in chronological order (FIFO). Through batching, a single inferent
    can handle multiple actor requests at once, while concurrent processes are
    used for better device utilisation.

    After running a forward pass on a batch, the results are distributed and
    made available to corresponding actors through mutually accessible queues.
    When actors finish (or are disconnected), the controller adds their keys
    to a synchronised list. Associated inferents use it to clear corresponding
    queues and states.

    The loop also reloads model parameters if they are found to have been updated.
    This is determined by a version value shared with the optimisers.
    """

    N_DELAY = 6
    N_PROCS = N_DELAY

    def __init__(
        self,
        args: argparse.Namespace,
        keys: List[Hashable],
        model_version: mp.Value,
        model_branch: mp.Value,
        manager: mp.Manager,
        device: str,
        termination_event: mp.Event
    ):
        self.key_lock = manager.Lock()
        self.request_queue: List[Hashable] = manager.list()
        self.removal_queue: List[Hashable] = manager.list()
        self.observation_queues: Dict[Hashable, mp.Queue] = {key: manager.Queue() for key in keys}
        self.action_queues: Dict[Hashable, mp.Queue] = {key: manager.Queue() for key in keys}

        # Single initial instance so that managed locks and dicts are shared between workers
        if args.seed is not None:
            torch.manual_seed(args.seed)

        model = PCNet(manager=manager, critic=True)

        self.workers = [
            mp.Process(
                target=self.worker,
                args=(
                    model,
                    args,
                    device,
                    model_version,
                    model_branch,
                    self.key_lock,
                    self.request_queue,
                    self.removal_queue,
                    self.observation_queues,
                    self.action_queues,
                    termination_event),
                daemon=True)
            for _ in range(self.N_PROCS)]

        for worker in self.workers:
            worker.start()

    @staticmethod
    def worker(
        model: PCNet,
        args: argparse.Namespace,
        device: str,
        model_version: mp.Value,
        model_branch: mp.Value,
        key_lock: mp.Lock,
        request_queue: List[int],
        removal_queue: List[int],
        observation_queues: Dict[int, mp.Queue],
        action_queues: Dict[int, mp.Queue],
        termination_event: mp.Event
    ):
        if args.checkpoint_path is not None:
            model = model.load(args.checkpoint_path, device=device, strict=args.strict)

        model.eval()

        local_version = -1
        batch_size = len(observation_queues)

        while not termination_event.is_set():
            global_version = model_version.get()

            # Reload model from state dict
            if global_version != local_version:
                model = model.load(
                    os.path.join(args.model_dir, args.model_name + f'_v{model_branch.get():03d}.pth'), device=device)

                local_version = global_version

            with key_lock:
                # Remove deprecated keys
                if removal_queue:
                    model.clear(keys=removal_queue)

                    for _ in range(len(removal_queue)):
                        del removal_queue[0]

                # Get unique keys in queue, preserving order
                unique_keys = list(dict.fromkeys(request_queue))

                if len(unique_keys) > batch_size:
                    unique_keys = unique_keys[:batch_size]

                # Confirm requests
                if len(unique_keys) == batch_size:
                    for _ in unique_keys:
                        del request_queue[0]

            if len(unique_keys) < batch_size:
                sleep(1e-3)
                continue

            images, spectra, mkbds, foci, actor_keys = list(
                zip(*[observation_queues[key].get() for key in unique_keys]))

            x_visual = torch.cat(images).to(device=device)
            x_audio = torch.cat(spectra).to(device=device)
            x_mkbd = torch.cat(mkbds).to(device=device)
            focus_coords = torch.cat(foci).to(device=device)

            # Run forward pass
            with torch.no_grad():
                inf_focus, inf_mkbd, inf_val = model(
                    x_visual, x_audio, x_mkbd, focus_coords, actor_keys, detach=True)

            # Distribute results
            for idx, key in enumerate(actor_keys):
                action_queues[key].put((inf_focus[idx:idx+1], inf_mkbd[idx:idx+1], inf_val[idx]))


def pi_to_log_pi(
    pi_focus: Categorical,
    pi_kbd: Categorical,
    pi_yrel: Categorical,
    pi_xrel: Categorical,
    pi_mwhl: Categorical,
    act_focus: torch.Tensor,
    act_kbd: torch.Tensor,
    act_yrel: torch.Tensor,
    act_xrel: torch.Tensor,
    act_mwhl: torch.Tensor
) -> torch.Tensor:
    """
    Extract log probabilities corresponding to given tensors of sampled indices
    from multiple categorical distributions.
    """

    log_pi_focus = pi_focus.log_prob(act_focus)[:, None]
    log_pi_kbd = pi_kbd.log_prob(act_kbd)
    log_pi_yrel = pi_yrel.log_prob(act_yrel)[:, None]
    log_pi_xrel = pi_xrel.log_prob(act_xrel)[:, None]
    log_pi_mwhl = pi_mwhl.log_prob(act_mwhl)[:, None]

    return torch.cat((log_pi_focus, log_pi_kbd, log_pi_yrel, log_pi_xrel, log_pi_mwhl), dim=1)


def ppo_policy_loss(
    log_pi: torch.Tensor,
    log_pi_old: torch.Tensor,
    advantage: torch.Tensor,
    clip: float = 0.2
) -> torch.Tensor:
    """Clipped PPO policy loss, averaged over samples in a batch."""

    ratio = torch.exp(log_pi - log_pi_old)
    clipped_ratio = ratio.clamp(min=1.-clip, max=1.+clip)
    policy_reward = torch.min(ratio * advantage, clipped_ratio * advantage)

    return -policy_reward.mean()


def ppo_value_loss(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared error, averaged over samples in a batch."""

    return torch.pow(value - target, 2).mean()


def dist_setup(rank: int, world_size: int):
    """Initialise a distributed process group."""

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '49160'

    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def dist_cleanup():
    """Terminate a distributed process group."""

    dist.destroy_process_group()


class Optimiser:
    """
    A possibly distributed process wrapper that works similarly to the loop in
    the supervised learning script. Optimisers that are part of the same process
    group work on only one and the same model version, constructing batches
    from their own trajectory buffers.

    After enough distinct trajectories are available for epochwise BPTT,
    connected optimisers separately compute gradients according to PPO and
    synchronise (average) them between each other before updating the model.

    When the model is saved by the designated process on disk, the event is
    communicated through a version value, so that inferents residing on the same
    local machine can load the updated parameters.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        args: argparse.Namespace,
        keys: List[Hashable],
        model_version: mp.Value,
        model_branch: mp.Value,
        manager: mp.Manager,
        device: str,
        termination_event: mp.Event
    ):
        self.key_lock = manager.Lock()
        self.request_queue: List[Hashable] = manager.list()
        self.removal_queue: List[Hashable] = manager.list()
        self.trajectory_queues: Dict[Hashable, mp.Queue] = {key: manager.Queue() for key in keys}

        self.proc = mp.Process(
            target=self.worker,
            args=(
                rank,
                world_size,
                args,
                device,
                model_version,
                model_branch,
                self.key_lock,
                self.request_queue,
                self.removal_queue,
                self.trajectory_queues,
                termination_event),
            daemon=True)

        self.proc.start()

    @staticmethod
    def worker(
        rank: int,
        world_size: int,
        args: argparse.Namespace,
        device: str,
        model_version: mp.Value,
        model_branch: mp.Value,
        key_lock: mp.Lock,
        request_queue: List[Hashable],
        removal_queue: List[Hashable],
        trajectory_queues: Dict[Hashable, mp.Queue],
        termination_event: mp.Event
    ):
        is_distributed = world_size > 1
        is_main = device in ('cuda', f'cuda:{MAIN_RANK}', 'cpu')
        device = torch.device(device)

        logger = get_logger('main' if is_main else f'aux:{rank}', path=args.logging_path, level=args.logging_level)
        logger.propagate = False

        # Same seed for initialising model weights
        torch.manual_seed(args.seed)

        model = PCNet(manager=None, critic=True)

        # Setup gradient clipping on backward pass
        for parameter in model.parameters():
            parameter.register_hook(lambda grad: grad.clamp_(-args.clip_grad_val, args.clip_grad_val))

        # Load weights from checkpoint if given
        if args.checkpoint_path is not None:
            model = model.load(args.checkpoint_path, device=device)

        elif args.device != 'cpu':
            model = model.move(device)

        # Establish role within the process group
        if is_distributed:
            dist_setup(rank, world_size)
            model = DDP(model, device_ids=[], output_device=device)

        # Confirm weights are consistent by logging a checksum for some parameters
        sleep(RANK_DELAY * rank)

        with torch.no_grad():
            checksum = (model.module if is_distributed else model).motor_decoding.action_fc.weight.sum().item()

        logger.debug('Weight checksum: %f', checksum)

        sleep(RANK_DELAY * world_size)

        writer = SummaryWriter(os.path.join(args.logdir, args.model_name + (f':{rank}' if is_distributed else '')))

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr_init,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            args.lr_max,
            total_steps=args.steps,
            pct_start=args.pct_start,
            base_momentum=args.beta1,
            max_momentum=args.beta1_max,
            div_factor=(args.lr_max/args.lr_init),
            final_div_factor=(args.lr_init/args.lr_final),
            last_epoch=-1)

        # See: https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822/2
        for _ in range(args.resume_step):
            scheduler.step()

        if args.device.startswith('cuda'):
            torch.backends.cudnn.benchmark = True

        args.batch_size = min(args.batch_size, len(trajectory_queues))
        start_time = perf_counter()
        running_policy_loss = 0.
        running_value_loss = 0.
        running_entropy = 0.
        running_rewards = 0.
        epoch_losses = deque()

        logger.debug('Training...')

        for i in range(1, args.steps+1):
            if termination_event.is_set():
                break

            effective_step = args.resume_step + i

            # Print out progress
            if is_main:
                running_time = perf_counter() - start_time
                remaining_time = min(int(running_time * (args.steps - effective_step) / i), MAX_DISP_SECONDS)

                print(
                    f'\rStep {effective_step} of {args.steps}. ETA: ' + str(timedelta(seconds=remaining_time)) + ' '*8,
                    end='')

            # Sample actors
            while True:
                with key_lock:
                    # Remove deprecated keys
                    if removal_queue:
                        model.clear(keys=removal_queue)

                        for _ in range(len(removal_queue)):
                            del removal_queue[0]

                    # Get unique keys in queue, preserving order
                    unique_keys = list(dict.fromkeys(request_queue))

                    if len(unique_keys) > args.batch_size:
                        unique_keys = unique_keys[:args.batch_size]

                    # Confirm requests
                    if len(unique_keys) == args.batch_size:
                        for _ in unique_keys:
                            del request_queue[0]

                if len(unique_keys) < args.batch_size:
                    sleep(0.1)

                else:
                    break

            # Receive data
            images, spectra, mkbds, foci, keys, \
                focus_old, mkbd_old, act_foci, act_kbds, act_yrels, act_xrels, act_mwhls, \
                advantages, value_targets, rewards = list(
                    zip(*[trajectory_queues[key].get() for key in unique_keys]))

            images = torch.cat(images, dim=1).to(device=device)
            spectra = torch.cat(spectra, dim=1).to(device=device)
            mkbds = torch.cat(mkbds, dim=1).to(device=device)
            foci = torch.cat(foci, dim=1).to(device=device)
            actor_keys = keys
            focus_old = torch.cat(focus_old, dim=1).to(device=device)
            mkbd_old = torch.cat(mkbd_old, dim=1).to(device=device)
            act_foci = torch.cat(act_foci, dim=1).to(device=device)
            act_kbds = torch.cat(act_kbds, dim=1).to(device=device)
            act_yrels = torch.cat(act_yrels, dim=1).to(device=device)
            act_xrels = torch.cat(act_xrels, dim=1).to(device=device)
            act_mwhls = torch.cat(act_mwhls, dim=1).to(device=device)
            advantages = torch.as_tensor(np.vstack(advantages).T[..., None], device=device)
            value_targets = torch.as_tensor(np.vstack(value_targets).T[..., None], device=device)
            rewards = np.vstack(rewards).T

            data = (
                images, spectra, mkbds, foci, focus_old, mkbd_old, act_foci, act_kbds, act_yrels, act_xrels, act_mwhls,
                advantages, value_targets, rewards)

            # Repeated updates
            for _ in range(args.update_steps):

                # Reset accumulated gradients
                optimizer.zero_grad()

                # Loop over temporal dim of actor data
                for j in range(args.slice_length):
                    obs_image, obs_audio, obs_mkbd, obs_focus, logits_focus_old, logits_mkbd_old, \
                        act_focus, act_kbd, act_yrel, act_xrel, act_mwhl, advantage, value_target, reward = \
                        (arraylike[j] for arraylike in data)

                    # Reproduce model outputs
                    logits_focus, logits_mkbd, values = model(obs_image, obs_audio, obs_mkbd, obs_focus, actor_keys)

                    # Compute total loss per actor
                    pi = logits_to_pi(logits_focus, logits_mkbd)
                    log_pi = pi_to_log_pi(*pi, act_focus, act_kbd, act_yrel, act_xrel, act_mwhl)

                    pi_old = logits_to_pi(logits_focus_old, logits_mkbd_old)
                    log_pi_old = pi_to_log_pi(*pi_old, act_focus, act_kbd, act_yrel, act_xrel, act_mwhl)

                    policy_term = ppo_policy_loss(log_pi, log_pi_old, advantage)

                    value_term = ppo_value_loss(values, value_target) * args.value_weight

                    # Sum entropy across actions, average it across the batch
                    entropy_term = sum(pi_act.entropy().sum() for pi_act in pi).mean() * -args.entropy_weight

                    loss = policy_term + value_term + entropy_term
                    epoch_losses.append(loss)

                    # Update running losses
                    running_policy_loss += policy_term.item()
                    running_value_loss += value_term.item()
                    running_entropy += entropy_term.item()
                    running_rewards += np.mean(reward)

                # Average over time
                loss = torch.stack(tuple(epoch_losses)).mean()
                epoch_losses.clear()

                # Accumulate gradients and sync on backward pass
                loss.backward()

                # Additional gradient manipulation
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                # Update model and learning schedule
                optimizer.step()
                scheduler.step()

            # Detach final (first) hidden/cell states of LSTM cells for EBPTT
            (model.module if is_distributed else model).detach(keys=actor_keys)

            # Log running losses
            if not i % args.log_steps:
                den = (args.log_steps * args.update_steps * len(data))

                writer.add_scalar('policy loss', running_policy_loss / den, global_step=effective_step)
                writer.add_scalar('value loss', running_value_loss / den, global_step=effective_step)
                writer.add_scalar('entropy', running_entropy / den, global_step=effective_step)
                writer.add_scalar('reward', running_rewards / den, global_step=effective_step)

                running_policy_loss = 0.
                running_value_loss = 0.
                running_entropy = 0.
                running_rewards = 0.

            # Save model checkpoint
            if is_main and not i % args.branch_steps:
                model_branch.set(model_branch.value + 1)

            if is_main and not i % args.save_steps:
                (model.module if is_distributed else model).save(
                    os.path.join(args.model_dir, args.model_name + f'_v{model_branch.value:03d}.pth'))

                model_version.set(model_version.value + 1)

        # Save final model parameters
        if is_main:
            (model.module if is_distributed else model).save(
                os.path.join(args.model_dir, args.model_name + '_final.pth'))

            print(f'\nFinished {args.steps} steps.', end='')

        # Terminate process group
        if is_distributed:
            dist_cleanup()

        # Inform root process to join
        if not termination_event.wait(0.):
            termination_event.set()


def get_reward(
    actor: SDGRemoteActor,
    last_scores: Dict[str, Union[int, float]]
) -> Tuple[float, Dict[str, Union[int, float]]]:
    """Infer reward from tracked stats/state."""

    reward = 0.

    rounds_won = actor.stats.tracked_scores['ct_wins'] + actor.stats.tracked_scores['t_wins']
    rounds_lost = actor.stats.tracked_scores['ct_rounds'] + actor.stats.tracked_scores['t_rounds'] - rounds_won

    # +1 for winning a round (losing a round is not penalised)
    if rounds_won > last_scores['rounds_won']:
        reward += 1.
        last_scores['rounds_won'] = rounds_won

    # +/-5 for winning/losing the match (draw does grant reward)
    if rounds_won == actor.session.ROUNDS_TO_WIN and not last_scores['end_triggered']:
        reward += 5.
        last_scores['end_triggered'] = 1

    elif rounds_lost == actor.session.ROUNDS_TO_WIN and not last_scores['end_triggered']:
        reward -= 5.
        last_scores['end_triggered'] = 1

    # +0.002 per point of damage dealt (max +1 per round)
    if actor.stats.tracked_scores['damage'] > last_scores['damage']:
        reward += 0.002 * (actor.stats.tracked_scores['damage'] - last_scores['damage'])
        last_scores['damage'] = actor.stats.tracked_scores['damage']

    # +0.2 per kill (max +1 per round)
    if actor.stats.tracked_scores['kills'] > last_scores['kills']:
        reward += 0.2 * (actor.stats.tracked_scores['kills'] - last_scores['kills'])
        last_scores['kills'] = actor.stats.tracked_scores['kills']

    # -1 per own team kill (min -5 per round)
    if actor.stats.tracked_scores['own_team_kills'] < last_scores['own_team_kills']:
        reward -= 1. * (actor.stats.tracked_scores['own_team_kills'] - last_scores['own_team_kills'])
        last_scores['own_team_kills'] = actor.stats.tracked_scores['own_team_kills']

    # -0.1 for death
    if actor.stats.tracked_scores['deaths'] > last_scores['deaths']:
        reward -= 0.1
        last_scores['deaths'] = actor.stats.tracked_scores['deaths']

    # +0.5 for planting
    if actor.stats.tracked_scores['plants'] > last_scores['plants']:
        reward += 0.5
        last_scores['plants'] = actor.stats.tracked_scores['plants']

    # +1 for defusing
    if actor.stats.tracked_scores['defuses'] > last_scores['defuses']:
        reward += 1.
        last_scores['defuses'] = actor.stats.tracked_scores['defuses']

    return reward, last_scores


def gae(
    rewards: Deque[float],
    values: Deque[float],
    gamma: float = 0.999936,
    lambda_: float = 0.95
) -> Tuple[np.ndarray]:
    """
    Generalised advantage estimation.

    Value targets correspond to the sum of current and expected future rewards
    (next estimated value) and are used in the value loss function.

    NOTE: gamma = 0.5 ^ (1/(360*30)); half-life for approx. 3 rounds at 30 FPS

    Reference:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/ppo/gae.py
    """

    values = list(values)
    advantages = deque()
    last_advantage = 0.

    for reward, new_value, old_value in zip(reversed(rewards), reversed(values[1:]), reversed(values[:-1])):
        delta = reward + gamma * new_value - old_value

        last_advantage = delta + gamma * lambda_ * last_advantage
        advantages.appendleft(last_advantage)

    advantages = np.array(advantages)
    value_targets = advantages + np.array(values[:-1])

    # NOTE: Advantages are often standardised, i.e. the mean is subtracted as well
    # That way, each update has about half encouraged and half discouraged actions, which may or may not be beneficial
    advantages /= np.std(advantages)

    return advantages, value_targets


class Actor:
    """
    Remote actor wrapper that uses a custom running loop that allows
    interruption, reward extraction from the local game state, and interaction
    with associated queues.

    After the actor goes through a matchmaker and connects to a session,
    it conveys observations to the inferent, receives actions (and values) back,
    and relays trajectories labelled with GAE to the optimiser.

    NOTE: The matchmaker needs to be spawned (and shut down) separately as it is
    generally agnostic of its clients and independent of the RL controller.

    NOTE: Uses threads instead of processes to prevent specific operations,
    which already tend to parallelise their workload themselves,
    from interfering with each other and rendering themselves unfunctional.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        key: Hashable,
        inferent: Inferent,
        optimiser: Optimiser,
        termination_event: mp.Event
    ):
        self.key = key
        args.name = f'ai{key:02d}'
        args.track_stats = True
        args.expose = True

        self.inferent = inferent
        self.optimiser = optimiser

        self.observation_queue = inferent.observation_queues[key]
        self.action_queue = inferent.action_queues[key]
        self.trajectory_queue = optimiser.trajectory_queues[key]

        self.proc = threading.Thread(
            target=self.worker,
            args=(
                args,
                key,
                inferent.key_lock,
                inferent.request_queue,
                self.observation_queue,
                self.action_queue,
                optimiser.key_lock,
                optimiser.request_queue,
                self.trajectory_queue,
                termination_event),
            daemon=True)

        self.proc.start()

    def clear_queues(self):
        """Clear queues associated with the actor."""

        while self.observation_queue.qsize():
            _ = self.observation_queue.get()

        while self.action_queue.qsize():
            _ = self.action_queue.get()

        while self.trajectory_queue.qsize():
            _ = self.trajectory_queue.get()

    def signal_removal(self):
        """Signal other processes to clear states associated with the actor."""

        with self.inferent.key_lock:
            self.inferent.removal_queue.append(self.key)

        with self.optimiser.key_lock:
            self.optimiser.removal_queue.append(self.key)

    @staticmethod
    def worker(
        args: argparse.Namespace,
        key: int,
        inf_key_lock: mp.Lock,
        inf_req_queue: List[int],
        observation_queue: mp.Queue,
        action_queue: mp.Queue,
        opt_key_lock: mp.Lock,
        opt_req_queue: List[int],
        trajectory_queue: mp.Queue,
        termination_event: mp.Event
    ):
        actor = SDGRemoteActor(args, key, inf_key_lock, inf_req_queue, observation_queue, action_queue)

        last_scores = {
            'end_triggered': 0,
            'rounds_won': 0,
            'kills': 0,
            'own_team_kills': 0,
            'deaths': 0,
            'plants': 0,
            'defuses': 0,
            'damage': 0.}

        last_value = 0.
        # last_return = 0.
        values = actor.values
        rewards = deque()

        actor.logger.name = args.name
        actor.logger.propagate = False
        actor.logger.info('Running...')

        actor.session_running = True

        previous_clock: float = None
        current_clock: float = None

        try:
            while actor.session_running and not termination_event.is_set():
                # Update loop timekeeping
                current_clock = actor._clock()
                dt_loop = (current_clock - previous_clock) if previous_clock is not None else 0.
                previous_clock = current_clock

                # Advance local state
                actor.step(dt_loop, current_clock)

                # Get reward
                # NOTE: There is a discrepancy between when an action was made and when it was rewarded,
                # but this should logically correspond to visual observations
                # (e.g. actor gets rewarded upon seeing confirmation of a hit/kill)
                reward, last_scores = get_reward(actor, last_scores)
                rewards.append(reward)

                # NOTE: Labelling of sequences with GAE should be done on a longer window,
                # i. e. multiple TBPTT windows labelled together before being sent to the optimiser
                if len(actor.act) >= args.slice_length:
                    tmp_rew = list(rewards)[:args.slice_length]
                    tmp_val = list(values)[:args.slice_length+1]
                    advantages, value_targets = gae(tmp_rew, tmp_val)
                    # returns = last_return + np.cumsum(rewards)

                    # Concatenate tensors to send
                    images, spectra, mkbds, foci = zip(*list(actor.obs)[:args.slice_length])
                    images = torch.stack(images)
                    spectra = torch.stack(spectra)
                    mkbds = torch.stack(mkbds)
                    foci = torch.stack(foci)

                    focus_old, mkbd_old = zip(*list(actor.logits)[:args.slice_length])
                    focus_old = torch.stack(focus_old)[:, None]
                    mkbd_old = torch.stack(mkbd_old)[:, None]

                    act_focus, act_kbd, act_yrel, act_xrel, act_mwhl = zip(*list(actor.act)[:args.slice_length])
                    act_focus = torch.stack(act_focus)[:, None]
                    act_kbd = torch.stack(act_kbd)[:, None]
                    act_yrel = torch.stack(act_yrel)[:, None]
                    act_xrel = torch.stack(act_xrel)[:, None]
                    act_mwhl = torch.stack(act_mwhl)[:, None]

                    for _ in range(args.slice_length):
                        del actor.obs[0]
                        del actor.logits[0]
                        del actor.act[0]

                    trajectory_queue.put((
                        images, spectra, mkbds, foci, key, focus_old, mkbd_old,
                        act_focus, act_kbd, act_yrel, act_xrel, act_mwhl, advantages, value_targets, tmp_rew))

                    with opt_key_lock:
                        opt_req_queue.append(key)

                    last_value = tmp_val[-1]
                    # last_return = returns[-1]

                    # values.clear()
                    # rewards.clear()

                    for _ in range(args.slice_length):
                        del values[0]
                        del rewards[0]

                    values.appendleft(last_value)

                # Delay to target specified FPS
                actor._fps_limiter.update_and_delay(actor._clock() - current_clock, current_clock)

        except ConnectionError:
            actor.logger.debug('Lost connection to the server.')

        else:
            if termination_event.is_set():
                actor.logger.debug('Process ended by user.')

            # NOTE: End of game is signaled by the termination of this worker
            else:
                actor.logger.debug('Session ended.')

            # Explicitly send any final messages still in queue due to sending stride
            if current_clock is not None:
                actor._send_client_data(current_clock)

                # Include slight delay to allow them to reach the server before disconnecting
                actor._fps_limiter.delay(0.5)

        # Saving and cleanup
        actor._socket.close()
        actor.cleanup()

        actor.logger.info('Stopped.')


def run_rl_controller(args: argparse.Namespace):
    """
    Root CPU process, keeping track of participating workers and managing how
    objects are shared between them.

    When actors are spawned, they are distributed among optimisers to balance
    their load, while a single inferent handles all of them at once.
    When actors finish (or are disconnected), the controller adds their keys to
    synchronised lists, which the inferent and associated optimiser use to clear
    corresponding queues and states.

    The underlying RL setup is crude self-play of a single model (without even
    past opponents) running on a single machine. Thus, the role of the
    controller is greatly diminished, as it does not actively participate in
    the training process (e.g. by sampling opponents) nor keep track of the data
    to do so (e.g. match scores, games played per agent instance, MMR, etc.).
    """

    mp.set_start_method('spawn', force=True)

    root_logger = get_logger('root', path=args.logging_path, level=args.logging_level)

    if args.device == 'cuda' and torch.cuda.is_available():
        n_devices = min(torch.cuda.device_count(), args.max_nprocs)

    else:
        args.device = 'cpu'
        n_devices = 1

    # TODO: Specific context or spawn method might be needed to work with cuda
    # mp_ctx = mp.get_context('spawn')
    # termination_event = mp_ctx.Event()
    # mp_manager = mp_ctx.Manager()
    mp_manager = mp.Manager()
    model_version = mp_manager.Value('i', -1)
    model_branch = mp_manager.Value('i', 0)
    termination_event = mp_manager.Event()

    root_logger.info('Launching...')

    # Assign devices
    inf_device: str = f'cuda:{n_devices-1}' if n_devices > 1 else args.device
    opt_devices: List[str] = [f'cuda:{i_device}' for i_device in range(n_devices-1)] if n_devices > 1 else [args.device]

    # Assign key subsets
    actor_keys = list(range(args.n_actors))
    dev_to_keys = {
        opt_device: list(key_subset)
        for opt_device, key_subset in zip(opt_devices, np.array_split(actor_keys, len(opt_devices)))}

    dev_to_idx = {opt_device: opt_idx for opt_idx, opt_device in enumerate(opt_devices)}
    keys_to_idx: Dict[int, str] = {
        key: dev_to_idx[opt_device] for opt_device, keys in dev_to_keys.items() for key in keys}

    # Launch inferent, optimiser, and actor processes
    inferent = Inferent(args, actor_keys, model_version, model_branch, mp_manager, inf_device, termination_event)

    optimisers: List[Optimiser] = [
        Optimiser(
            opt_idx,
            len(opt_devices),
            args,
            dev_to_keys[opt_device],
            model_version,
            model_branch,
            mp_manager,
            opt_device,
            termination_event)
        for opt_idx, opt_device in enumerate(opt_devices)]

    actors: List[Actor] = [
        Actor(args, actor_key, inferent, optimisers[keys_to_idx[actor_key]], termination_event)
        for actor_key in actor_keys]

    # Check and relaunch actors on match end, clearing related queues and states
    try:
        while not termination_event.wait(INTERRUPT_CHECK_PERIOD):
            for actor_idx in range(len(actors)):
                actor = actors[actor_idx]

                if not actor.proc.is_alive():
                    actor.proc.join()

                    actor.clear_queues()
                    actor.signal_removal()

                    actors[actor_idx] = Actor(args, actor.key, actor.inferent, actor.optimiser, termination_event)

    except KeyboardInterrupt:
        termination_event.set()

        print()
        root_logger.debug('Training interrupted by user.')

    # Join and close all processes
    for actor in actors:
        actor.proc.join()
        actor.clear_queues()

    for proc in inferent.workers:
        proc.join()
        proc.close()

    for optimiser in optimisers:
        optimiser.proc.join()
        optimiser.proc.close()

    mp_manager.shutdown()

    root_logger.info('Done.')
