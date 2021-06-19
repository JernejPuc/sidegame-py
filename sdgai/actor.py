"""Trained AI client interface of the live client for SDG"""

from argparse import Namespace
from queue import Empty
from typing import Any, Callable, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.multiprocessing import Process, Queue, Event, set_start_method
from sidegame.networking.core import StridedFunction
from sidegame.game.shared import GameID, Map, Player
from sidegame.game.client.base import SDGLiveClientBase
from sidegame.game.client.interface import SDGLiveClient
from sidegame.audio import get_mel_basis, spectrify
from sdgai.utils import prepare_inputs, spatial_softmax
from sdgai.model import PCNet


set_start_method('spawn', force=True)


class SDGTrainedActor(SDGLiveClientBase):
    """
    A client for live communication with the SDG server.

    It uses multi-threaded inference to get regular actions (actor inputs)
    from a trained PCNet based on a stream of observations.

    NOTE: Might need more restrictions for sampling mode. Otherwise, the odds
    inevitably trigger disorienting view transitions too frequently and can
    hinder the agent past the playable point. The same should apply for RL
    actors.
    """

    N_DELAY = 6
    N_PROCS = N_DELAY

    MOUSE_BINS = [
        -108.0, -72.73, -48.87, -32.73, -21.82, -14.43, -9.44, -6.06, -3.78, -2.23, -1.19, -0.48,
        0., 0.48, 1.19, 2.23, 3.78, 6.06, 9.44, 14.43, 21.82, 32.73, 48.87, 72.73, 108.]

    MOUSE_WHEEL = [-1, 0, 1]

    NULL_STATE = [0, 0, 0, 0, 0, 0, 0, 160, 54, 1, 1, 1, GameID.VIEW_LOBBY, GameID.NULL, Map.PLAYER_ID_NULL, 0.]

    # Movement
    MKBD_IDX_W = 0  # Forward
    MKBD_IDX_S = 1  # Backward
    MKBD_IDX_D = 2  # Rightward
    MKBD_IDX_A = 3  # Leftward

    # Item interaction
    MKBD_IDX_E = 4  # Use
    MKBD_IDX_R = 5  # Reload
    MKBD_IDX_G = 6  # Drop
    MKBD_IDX_B = 7  # Buy

    # Views & communication
    MKBD_IDX_X = 8          # Terms
    MKBD_IDX_C = 9          # Items
    MKBD_IDX_TAB = 10       # Map/stats
    MKBD_IDX_SPACE = 11     # Send/clear message

    # Mouse
    MKBD_IDX_LBTN = 12      # Fire
    MKBD_IDX_RBTN = 13      # Walk
    MKBD_IDX_NUM = 14       # Slot key
    MKBD_IDX_WHLY = 15      # Scroll
    MKBD_IDX_YREL = 16      # Cursor mvmt.
    MKBD_IDX_XREL = 17      # Cursor mvmt.
    MKBD_IDX_CRSR_Y = 18    # Cursor pos.
    MKBD_IDX_CRSR_X = 19    # Cursor pos.

    def __init__(self, args: Namespace):
        self.inference_queue = Queue()
        self.action_queue = Queue()
        self.termination_event = Event()

        self.device = torch.device(args.device)
        self.model = PCNet()
        self.model.load(args.model_path, device=args.device)
        self.model.eval()

        self.workers = [
            Process(
                target=self.worker,
                args=(self.model, self.inference_queue, self.action_queue, self.termination_event), daemon=True)
            for _ in range(self.N_PROCS)]

        for worker in self.workers:
            worker.start()

        super().__init__(args)

        self.mel_basis = get_mel_basis(sampling_rate=44100, n_fft=2048, n_mel=64)
        self.window: np.ndarray = np.hamming(int(44100/args.refresh_rate + 255*2))[None]
        self.ref = self.window.sum()**2 / 2.
        self.eps = 1e-12
        self.actor_keys = [self.own_entity.name]

        self.sampling_proba = args.sampling_proba
        self.sampling_thr = args.sampling_thr
        self.cutout_centre_indices = (27, 80)
        self.mkbd_state = [0]*20
        self.space_time = 0.
        self.on_hold = True

        self.strided_inference: Callable = StridedFunction(self.queue_inference, args.tick_rate / args.refresh_rate)

    def poll_user_input(self, timestamp: float) -> Tuple[Any, Union[Any, None]]:
        """
        Poll or read peripheral events and interpret them as user input
        and optional local log data.

        Specifically, pop the oldest tensors from the queue that is being
        accessed and updated by inferent workers and convert them into the
        expected format.
        """

        own_player: Player = self.own_entity
        sim = self.sim
        log = None

        # Wait for match to start and action queue to fill
        if not self.session.phase:
            if not self.on_hold:
                self.on_hold = True

            self.clear_queues()

            return self.NULL_STATE, log

        elif self.on_hold and (self.action_queue.qsize() + self.inference_queue.qsize()) < self.N_DELAY:
            return self.NULL_STATE, log

        else:
            self.on_hold = False

        # Get oldest action (or wait until one is available) and unpack it
        try:
            t_action = self.action_queue.get(block=True, timeout=3.)

        except Empty:
            return self.NULL_STATE, log

        x_focus, x_action = t_action
        x_focus = spatial_softmax(x_focus)[0, 0]
        x_action = x_action[0].clone()

        if self.sampling_proba and self.sampling_proba == 1. or self.rng.random() < self.sampling_proba:
            self.cutout_centre_indices = np.unravel_index(Categorical(x_focus.view(-1)).sample(), x_focus.shape)

            kbd = torch.sigmoid(x_action[:19])
            kbd = Categorical(torch.vstack((kbd, 1. - kbd)).transpose(0, 1)).sample().numpy()

            mmot_yrel = self.MOUSE_BINS[Categorical(F.softmax(x_action[19:44], dim=0)).sample().item()]
            mmot_xrel = self.MOUSE_BINS[Categorical(F.softmax(x_action[44:69], dim=0)).sample().item()]

            wheel_y = self.MOUSE_WHEEL[Categorical(F.softmax(x_action[69:72], dim=0)).sample().item()]

        else:
            self.cutout_centre_indices = np.unravel_index(x_focus.argmax(), x_focus.shape)

            kbd = (torch.sigmoid(x_action[:19]) >= 0.5).numpy().astype(int)

            mmot_yrel = self.MOUSE_BINS[x_action[19:44].argmax().item()]
            mmot_xrel = self.MOUSE_BINS[x_action[44:69].argmax().item()]

            wheel_y = self.MOUSE_WHEEL[x_action[69:72].argmax().item()]

        # Clear shared data
        del t_action, x_focus

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
            elif sim.view == GameID.VIEW_ITEMS and self.mkbd_state[self.MKBD_IDX_X] and not xkey:
                sim.view = GameID.VIEW_WORLD
                log = sim.create_log(GameID.EVAL_MSG_ITEM)

            # Buy an item
            elif sim.view == GameID.VIEW_STORE and self.mkbd_state[self.MKBD_IDX_X] and not xkey:
                sim.view = GameID.VIEW_WORLD
                log = sim.create_log(GameID.EVAL_BUY)

            elif not (xkey or ckey or bkey or tab):
                sim.view = GameID.VIEW_WORLD

        # To clear the message draft, space hold must not be interrupted
        # When sampling, semi-argmax is used to restrict sampling below a threshold
        if self.sampling_proba and self.mkbd_state[self.MKBD_IDX_SPACE]:
            space_argmax = self.sampling_thr <= x_action[2] <= (1. - self.sampling_thr)

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
            ekey_argmax = self.sampling_thr <= x_action[3] <= (1. - self.sampling_thr)

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

        # Should be cloned, but still
        del x_action

        return state, log

    def generate_output(self, dt: float):
        if not self.session.phase:
            return

        # Get observations
        self.sim.eval_effects(dt)
        self.strided_inference()

    def queue_inference(self):
        """
        Get and defer current observations to an available inference worker
        through a queue.
        """

        frame = self.sim.get_frame()

        self.sim.audio_system.step()
        sound = self.sim.audio_system.external_buffer.popleft()

        spectral_vectors = spectrify(
            sound, mel_basis=self.mel_basis, window=self.window, sampling_rate=44100, n_fft=2048, n_mel=64,
            eps=self.eps, ref=self.ref)

        # Prepare data for model inference
        x_visual, x_audio, x_mkbd, x_focus = prepare_inputs(
            frame, spectral_vectors, self.mkbd_state, self.cutout_centre_indices, self.eps, self.device)

        self.inference_queue.put((x_visual, x_audio, x_mkbd, x_focus, self.actor_keys))

    def cleanup(self):
        self.termination_event.set()

        for worker in self.workers:
            worker.join()

        self.inference_queue.close()
        self.action_queue.close()

        self.inference_queue.join_thread()
        self.action_queue.join_thread()

    def clear_queues(self):
        """Clear shared data."""

        # Supposedly, shared tensors must be explicitly cleared?

        while self.action_queue.qsize():
            tmp = self.action_queue.get()
            del tmp

        while self.inference_queue.qsize():
            tmp = self.inference_queue.get()
            del tmp

    @staticmethod
    def worker(model: torch.nn.Module, inference_queue: Queue, action_queue: Queue, termination_event: Event):
        """Execute inference if input data is available."""

        while not termination_event.wait(0.):
            try:
                data = inference_queue.get(block=True, timeout=3.)

            except Empty:
                continue

            x_visual, x_audio, x_mkbd, focus_coords, actor_keys = data

            with torch.no_grad():
                inferred_actions = model(x_visual, x_audio, x_mkbd, focus_coords, actor_keys)

            # Clear shared data
            del data, x_visual, x_audio, x_mkbd, focus_coords, actor_keys

            action_queue.put(inferred_actions)
