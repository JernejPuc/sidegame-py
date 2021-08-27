"""AI client interface of the live client for SDG"""

from abc import abstractmethod
from argparse import Namespace
from collections import deque
from typing import Any, Callable, Hashable, List, Tuple, Union
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.distributions.categorical import Categorical
from sidegame.networking.core import StridedFunction
from sidegame.game.shared import GameID, Map, Player
from sidegame.game.client.simulation import Simulation
from sidegame.game.client.base import SDGLiveClientBase
from sidegame.game.client.interface import SDGLiveClient
from sidegame.game.client.tracking import FocusTracker
from sidegame.audio import get_mel_basis, spectrify
from sdgai.utils import prepare_inputs
from sdgai.model import PCNet


def logits_to_pi(x_focus: torch.Tensor, x_mkbd: torch.Tensor) -> Tuple[Categorical]:
    """
    Convert tensors of focal and mouse/keyboard state logits to multiple
    categorical distributions.
    """

    x_focus = x_focus.reshape(len(x_focus), -1)
    x_kbd = torch.sigmoid(x_mkbd[:, :19, None])
    x_kbd = torch.cat((x_kbd, 1. - x_kbd), dim=-1)
    x_yrel = x_mkbd[:, 19:44]
    x_xrel = x_mkbd[:, 44:69]
    x_mwhl = x_mkbd[:, 69:72]

    pi_focus = Categorical(logits=x_focus)
    pi_kbd = Categorical(probs=x_kbd)
    pi_yrel = Categorical(logits=x_yrel)
    pi_xrel = Categorical(logits=x_xrel)
    pi_mwhl = Categorical(logits=x_mwhl)

    return pi_focus, pi_kbd, pi_yrel, pi_xrel, pi_mwhl


class SDGBaseActor(SDGLiveClientBase):
    """
    A client that infers actions from a trained (or in-training) PCNet model
    based on a stream of observations.

    NOTE: Might need more restrictions for sampling mode. Otherwise, the odds
    inevitably trigger disorienting view transitions too frequently and can
    hinder the agent past a playable point. The same should apply for RL actors.
    """

    # Reaction time in number of steps
    N_DELAY = 6

    # Categories
    MOUSE_BINS = [
        -108.0, -72.73, -48.87, -32.73, -21.82, -14.43, -9.44, -6.06, -3.78, -2.23, -1.19, -0.48,
        0., 0.48, 1.19, 2.23, 3.78, 6.06, 9.44, 14.43, 21.82, 32.73, 48.87, 72.73, 108.]

    MOUSE_WHEEL = [-1, 0, 1]

    # Audio constants
    SAMPLING_RATE = 44100
    HRIR_LEN = 256
    MIN_AMP = 1e-12
    N_FFT = 2048
    N_MEL = 64

    # Polled action before actions can be obtained from the model
    NULL_ACT = ([0, 0, 0, 0, 0, 0, 0, 160, 54, 1, 1, 1, GameID.VIEW_LOBBY, GameID.NULL, Map.PLAYER_ID_NULL, 0.], None)

    # Mouse/keyboard state indices
    # Movement
    # MKBD_IDX_W = 0  # Forward
    # MKBD_IDX_S = 1  # Backward
    # MKBD_IDX_D = 2  # Rightward
    # MKBD_IDX_A = 3  # Leftward

    # Item interaction
    MKBD_IDX_E = 4  # Use
    # MKBD_IDX_R = 5  # Reload
    # MKBD_IDX_G = 6  # Drop
    MKBD_IDX_B = 7  # Buy

    # Views & communication
    MKBD_IDX_X = 8          # Terms
    MKBD_IDX_C = 9          # Items
    MKBD_IDX_TAB = 10       # Map/stats
    MKBD_IDX_SPACE = 11     # Send/clear message

    # Mouse
    MKBD_IDX_LBTN = 12      # Fire
    MKBD_IDX_RBTN = 13      # Walk
    # MKBD_IDX_NUM = 14       # Slot key
    # MKBD_IDX_WHLY = 15      # Scroll
    # MKBD_IDX_YREL = 16      # Cursor mvmt.
    # MKBD_IDX_XREL = 17      # Cursor mvmt.
    # MKBD_IDX_CRSR_Y = 18    # Cursor pos.
    # MKBD_IDX_CRSR_X = 19    # Cursor pos.

    def __init__(self, args: Namespace):
        super().__init__(args)

        # Override audio-related settings and samples to properly account for change in time scale
        if args.time_scale != 1.:
            real_tick_rate = args.tick_rate / args.time_scale
            alt_sim = Simulation(real_tick_rate, args.volume, self.sim.own_player_id, self.rng)
            self.sim.audio_system = alt_sim.audio_system
            self.audio_stream = alt_sim.audio_system.external_buffer
            self.sim.inventory = alt_sim.inventory
            self.sim.sounds = alt_sim.sounds
            self.sim.movements = alt_sim.movements
            self.sim.keypresses = alt_sim.keypresses
            self.sim.footsteps = alt_sim.footsteps

        else:
            real_tick_rate = args.tick_rate

        # Audio variables
        self.mel_basis = get_mel_basis(sampling_rate=self.SAMPLING_RATE, n_fft=self.N_FFT, n_mel=self.N_MEL)
        self.ham_window: np.ndarray = np.hamming(int(self.SAMPLING_RATE/real_tick_rate + (self.HRIR_LEN-1)*2))[None]
        self.db_ref = self.ham_window.sum()**2 / 2.

        # Sampling conditions
        self.sampling_proba = args.sampling_proba
        self.sampling_thr = args.sampling_thr

        # Last action state
        self.cutout_centre_indices = (27, 80)
        self.mkbd_state = [0]*20
        self.space_time = 0.
        self.on_hold = True

        # Cutout centre tracking
        if args.focus_path is not None and args.record:
            self.focus = FocusTracker(path=args.focus_path, mode=FocusTracker.MODE_WRITE, start_active=True)
        else:
            self.focus = None

        # Inference base
        self.device = args.device
        self.strided_inference: Callable = StridedFunction(self.queue_inference, args.tick_rate / args.refresh_rate)

    @abstractmethod
    def clear_queue(self):
        """Clear inference/action data from shared structures."""
        raise NotImplementedError

    @abstractmethod
    def get_queue_length(self) -> int:
        """Estimate the number of actions in the inference pipeline."""
        raise NotImplementedError

    @abstractmethod
    def get_action_from_queue(self) -> Tuple[torch.Tensor]:
        """
        Pop the oldest tensors from the queue that is being accessed and updated
        by inferent workers.
        """
        raise NotImplementedError

    def expose_action(self, _inferred_t_action: Tuple[torch.Tensor], _extracted_t_action: Tuple[torch.Tensor]):
        """Allow inferred sub-actions to populate externally accessible structures."""

    @abstractmethod
    def infer_action(self, x_visual: torch.Tensor, x_audio: torch.Tensor, x_mkbd: torch.Tensor, x_focus: torch.Tensor):
        """Get and defer current observations to an available inference worker."""
        raise ValueError

    def poll_user_input(self, timestamp: float) -> Tuple[Any, Union[Any, None]]:
        """
        Poll or read peripheral events and interpret them as user input
        and optional local log data.
        """

        own_player: Player = self.own_entity
        sim = self.sim
        log = None

        # Wait for match to start and action queue to fill
        if not self.session.phase:
            self.clear_queue()
            return self.NULL_ACT

        elif self.get_queue_length() <= self.N_DELAY:
            self.cutout_centre_indices = (27, 80)
            return self.NULL_ACT

        # Get oldest action (or wait until one is available) and unpack it
        t_action = self.get_action_from_queue()
        x_focus, x_action = t_action[0], t_action[1]

        # Extract sub-actions from logits
        if self.sampling_proba and (self.sampling_proba == 1. or self.rng.random() < self.sampling_proba):
            cat_focus, cat_kbd, cat_yrel, cat_xrel, cat_whly = logits_to_pi(x_focus, x_action)

            act_focus = cat_focus.sample()[0]
            act_kbd = cat_kbd.sample()[0]
            act_yrel = cat_yrel.sample()[0]
            act_xrel = cat_xrel.sample()[0]
            act_whly = cat_whly.sample()[0]

            new_cutout_centre_indices = np.unravel_index(act_focus.item(), x_focus[0, 0].shape)
            kbd = act_kbd.numpy()

        else:
            x_focus = x_focus[0, 0]
            x_action = x_action[0]

            act_focus = x_focus.argmax()
            act_kbd = (torch.sigmoid(x_action[:19]) >= 0.5).to(torch.long)
            act_yrel = x_action[19:44].argmax()
            act_xrel = x_action[44:69].argmax()
            act_whly = x_action[69:72].argmax()

            new_cutout_centre_indices = np.unravel_index(act_focus, x_focus.shape)
            kbd = act_kbd.numpy().astype(int)

        mmot_yrel = self.MOUSE_BINS[act_yrel.item()]
        mmot_xrel = self.MOUSE_BINS[act_xrel.item()]
        wheel_y = self.MOUSE_WHEEL[act_whly.item()]

        # Update tracked focal point
        if self.focus is not None:
            self.focus.update(
                2*(new_cutout_centre_indices[0]-self.cutout_centre_indices[0]),
                2*(new_cutout_centre_indices[1]-self.cutout_centre_indices[1]))

        self.cutout_centre_indices = new_cutout_centre_indices

        # TODO: Handle confusing and debilitating repeated switching between views
        if self.rng.random() > 0.1:
            for act_idx, mkbd_idx in zip([13, 11, 12, 10], range(self.MKBD_IDX_B, self.MKBD_IDX_TAB)):
                if self.mkbd_state[mkbd_idx]:
                    act_kbd[act_idx] = (torch.sigmoid(x_action[act_idx]) >= 0.1).to(torch.long)

        if self.rng.random() > 0.1:
            for act_idx, mkbd_idx in zip([13, 11, 12, 10], range(self.MKBD_IDX_B, self.MKBD_IDX_TAB)):
                if not self.mkbd_state[mkbd_idx]:
                    act_kbd[act_idx] = (torch.sigmoid(x_action[act_idx]) >= 0.9).to(torch.long)

        # Expose action for RL
        self.expose_action(t_action, (act_focus, act_kbd, act_yrel, act_xrel, act_whly))

        # Keys
        lbtn, rbtn, space, ekey, gkey, rkey, dkey, akey, wkey, skey, tab, xkey, ckey, bkey = kbd[:14]
        skey *= -1
        akey *= -1

        kbd_nums = kbd[14:]
        draw_slot = kbd_nums.tolist().index(1) if any(kbd_nums) else 0

        # x/c/b/tab
        if xkey and sim.view != GameID.VIEW_TERMS:
            sim.view = GameID.VIEW_TERMS
            sim.cursor_y, sim.cursor_x = sim.WORLD_FRAME_CENTRE

        elif ckey and sim.view != GameID.VIEW_ITEMS:
            sim.view = GameID.VIEW_ITEMS
            sim.cursor_y, sim.cursor_x = sim.WORLD_FRAME_CENTRE

        elif bkey and sim.view != GameID.VIEW_STORE:
            can_view_store = sim.observed_player_id == sim.own_player_id and \
                self.session.check_player_buy_eligibility(sim.own_player_id)

            if can_view_store:
                sim.view = GameID.VIEW_STORE
                sim.cursor_y, sim.cursor_x = sim.WORLD_FRAME_CENTRE

        elif tab:
            sim.view = GameID.VIEW_MAPSTATS

        elif sim.view != GameID.VIEW_WORLD:

            # Add term to message
            if sim.view == GameID.VIEW_TERMS and self.mkbd_state[self.MKBD_IDX_X] and not xkey:
                sim.view = GameID.VIEW_WORLD
                log = sim.create_log(GameID.EVAL_MSG_TERM)

            # Add item to message
            elif sim.view == GameID.VIEW_ITEMS and self.mkbd_state[self.MKBD_IDX_C] and not ckey:
                sim.view = GameID.VIEW_WORLD
                log = sim.create_log(GameID.EVAL_MSG_ITEM)

            # Buy an item
            elif sim.view == GameID.VIEW_STORE and self.mkbd_state[self.MKBD_IDX_B] and not bkey:
                sim.view = GameID.VIEW_WORLD
                log = sim.create_log(GameID.EVAL_BUY)

            elif not (xkey or ckey or bkey or tab):
                sim.view = GameID.VIEW_WORLD

        # To clear the message draft, space hold must not be interrupted
        # When sampling, semi-argmax is used to restrict sampling below a threshold
        if self.sampling_proba and self.mkbd_state[self.MKBD_IDX_SPACE]:
            space_argmax = self.sampling_thr <= x_action[..., 2].item() <= (1. - self.sampling_thr)

            if space_argmax:
                space = 1

        if space:
            if self.mkbd_state[self.MKBD_IDX_SPACE]:
                if sim.message_draft and (timestamp - self.space_time) > 0.5:
                    sim.clear_message_draft()
            else:
                self.space_time = timestamp

        elif self.mkbd_state[self.MKBD_IDX_SPACE] and not space:
            log = sim.create_log(GameID.EVAL_MSG_SEND)
            self.space_time = 0.

        # Mouse buttons
        if self.mkbd_state[self.MKBD_IDX_LBTN] and not lbtn:
            if sim.view == GameID.VIEW_WORLD:
                sim.change_observed_player(rotate_upward=False)

            # Add mark to message
            elif sim.view == GameID.VIEW_MAPSTATS:
                log = sim.create_log(GameID.EVAL_MSG_MARK)

        if self.mkbd_state[self.MKBD_IDX_RBTN] and not rbtn and sim.view == GameID.VIEW_WORLD:
            sim.change_observed_player(rotate_upward=True)

        # Mouse wheel
        if wheel_y > 0:
            sim.wheel_y = max(sim.wheel_y-1, 0)

        elif wheel_y < 0:
            sim.wheel_y = min(sim.wheel_y+1, max(len(sim.chat)-5, 0))

        # Update cursor and angle movement
        sim.cursor_y = np.clip(sim.cursor_y + mmot_yrel, 2., 105.)

        if sim.view == GameID.VIEW_WORLD and not self.session.is_dead_or_spectator(sim.own_player_id):
            sim.cursor_x = 159.5
            d_angle = mmot_xrel
        else:
            sim.cursor_x = np.clip(sim.cursor_x + mmot_xrel, 66., 253.)
            d_angle = 0.

        # Get drawn item id from key num
        draw_id = SDGLiveClient.get_next_item_by_slot(self, draw_slot)

        # Get hovered (entity) ID
        hovered_id = GameID.NULL
        hovered_entity_id = Map.PLAYER_ID_NULL

        if sim.view == GameID.VIEW_WORLD:
            if self.session.is_dead_or_spectator(sim.own_player_id):
                hovered_entity_id = sim.observed_player_id
            else:
                _, hovered_entity_id, hovered_id = sim.get_cursor_obs_from_world_view()

        elif sim.view == GameID.VIEW_TERMS:
            hovered_id = sim.get_cursor_obs_from_code_view(sim.code_view_terms)

        elif sim.view == GameID.VIEW_ITEMS:
            hovered_id = sim.get_cursor_obs_from_code_view(sim.code_view_items)

        elif sim.view == GameID.VIEW_STORE:
            if own_player.team == GameID.GROUP_TEAM_T:
                hovered_id = sim.get_cursor_obs_from_code_view(sim.code_view_store_t)
            else:
                hovered_id = sim.get_cursor_obs_from_code_view(sim.code_view_store_ct)

        # To plant or defuse, E hold must not be interrupted
        # When sampling, semi-argmax is used to restrict sampling below a threshold
        can_plant_or_defuse = hovered_id == GameID.ITEM_C4 or \
            (own_player.held_object is not None and own_player.held_object.item.id == GameID.ITEM_C4)

        if self.sampling_proba and self.mkbd_state[self.MKBD_IDX_E] and can_plant_or_defuse:
            ekey_argmax = self.sampling_thr <= x_action[..., 3].item() <= (1. - self.sampling_thr)

            if ekey_argmax:
                ekey = 1

        state = [
            lbtn, rbtn, space, ekey, gkey, rkey, draw_id,
            round(sim.cursor_x), round(sim.cursor_y),
            wkey + skey + 1, dkey + akey + 1, max(-1, min(1, wheel_y)) + 1,
            sim.view, hovered_id, hovered_entity_id, d_angle]

        self.mkbd_state = [
            wkey, skey, dkey, akey, ekey, rkey, gkey, bkey, xkey, ckey, tab, space, lbtn, rbtn,
            draw_slot / 5, wheel_y, mmot_yrel / self.MOUSE_BINS[-1], mmot_xrel / self.MOUSE_BINS[-1],
            (sim.cursor_y - 53.5) / 51.5, (sim.cursor_x - 159.5) / 93.5]

        return state, log

    def generate_output(self, dt: float):
        if not self.session.phase:
            return

        # Get observations
        self.sim.eval_effects(dt * self.time_scale)

        # Log focus
        if self.focus is not None:
            self.focus.register(None, self.recorder.counter)

        self.strided_inference()

    def queue_inference(self):
        """
        Prepare data for model inference and run forward pass (immediately or
        remotely).
        """

        frame = self.sim.get_frame()

        self.sim.audio_system.step()
        sound = self.sim.audio_system.external_buffer.popleft()

        spectral_vectors = spectrify(
            sound, mel_basis=self.mel_basis, window=self.ham_window, sampling_rate=self.SAMPLING_RATE, n_fft=self.N_FFT,
            n_mel=self.N_MEL, eps=self.MIN_AMP, ref=self.db_ref)

        x_visual, x_audio, x_mkbd, x_focus = prepare_inputs(
            frame, spectral_vectors, self.mkbd_state, self.cutout_centre_indices, self.MIN_AMP, self.device)

        self.infer_action(x_visual, x_audio, x_mkbd, x_focus)

    def cleanup(self):
        if self.focus is not None:
            self.focus.finish(self.logger)


class SDGSimpleActor(SDGBaseActor):
    """An actor using serial (single-threaded) local inference."""

    def __init__(self, args: Namespace):
        super().__init__(args)

        self.actor_keys = [self.own_entity.name]
        self.actions = deque()

        if args.seed is not None:
            torch.manual_seed(args.seed)

        self.model = PCNet(critic=False)

        if args.checkpoint_path is not None:
            self.model = self.model.load(args.checkpoint_path, device=args.device)

        self.model.eval()

    def clear_queue(self):
        self.actions.clear()

    def get_queue_length(self) -> int:
        return len(self.actions)

    def get_action_from_queue(self) -> Tuple[torch.Tensor]:
        return self.actions.popleft()

    def infer_action(self, x_visual: torch.Tensor, x_audio: torch.Tensor, x_mkbd: torch.Tensor, x_focus: torch.Tensor):
        with torch.no_grad():
            self.actions.append(self.model(x_visual, x_audio, x_mkbd, x_focus, self.actor_keys, detach=True))


class SDGRemoteActor(SDGBaseActor):
    """An actor using remote multi-threaded inference."""

    def __init__(
        self,
        args: Namespace,
        key: Hashable,
        request_lock: mp.Lock,
        request_queue: List[Hashable],
        inference_queue: mp.Queue,
        action_queue: mp.Queue
    ):
        self.key = key
        self.actor_keys = key
        self.request_lock = request_lock
        self.request_queue = request_queue
        self.inference_queue = inference_queue
        self.action_queue = action_queue

        self.expose: bool = args.expose
        self.values = deque([0.])
        self.logits = deque()
        self.obs = deque()
        self.act = deque()

        super().__init__(args)

    def clear_queue(self):
        while self.inference_queue.qsize():
            _ = self.inference_queue.get()

        while self.action_queue.qsize():
            _ = self.action_queue.get()

    def get_queue_length(self) -> int:
        return self.action_queue.qsize() + self.inference_queue.qsize()

    def get_action_from_queue(self) -> Union[Tuple[torch.Tensor], None]:
        return self.action_queue.get()

    def expose_action(self, _inferred_t_action: Tuple[torch.Tensor], _extracted_t_action: Tuple[torch.Tensor]):
        if not self.expose:
            return

        # Inferent should also return a value, storing or exposing it for later steps
        self.values.append(_inferred_t_action[2].item())
        self.logits.append(_inferred_t_action[:2])
        self.act.append(_extracted_t_action)

    def infer_action(self, x_visual: torch.Tensor, x_audio: torch.Tensor, x_mkbd: torch.Tensor, x_focus: torch.Tensor):
        self.inference_queue.put((x_visual, x_audio, x_mkbd, x_focus, self.actor_keys))

        with self.request_lock:
            self.request_queue.append(self.key)

        if self.expose:
            self.obs.append((x_visual, x_audio, x_mkbd, x_focus))
