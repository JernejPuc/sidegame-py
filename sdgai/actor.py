"""AI client interface of the live client for SDG"""

import asyncio
from argparse import Namespace
from collections import deque
from logging import ERROR

import numpy as np
import torch
from torch.distributions.categorical import Categorical

from sidegame.game import GameID, MapID
from sidegame.game.client.base import SDGLiveClientBase
from sidegame.game.client.tracking import FocusTracker
from sidegame.audio import get_mel_basis, spectrify

from sdgai.model import PCNet


# Audio constants
SAMPLING_RATE = 44100
HRIR_LEN = 256
MIN_AMP = 1e-12
LOG_MIN_AMP = -10. * np.log10(MIN_AMP)


def prepare_inputs(frames: np.ndarray, spectra: np.ndarray, mkbds: np.ndarray, foci: np.ndarray, device: str):
    """Convert stacks of observations into model inputs on target device."""

    # BGR order to RGB, channels last to first, and values to [0., 1.] range
    frames = np.moveaxis(frames[..., ::-1], -1, 1) / 255.
    spectra = spectra / LOG_MIN_AMP + 1.

    x_visual = torch.from_numpy(frames).to(device, dtype=torch.float32)
    x_audio = torch.from_numpy(spectra).to(device, dtype=torch.float32)
    x_manual = torch.from_numpy(mkbds).to(device, dtype=torch.float32)
    x_focal = torch.from_numpy(foci).to(device, dtype=torch.int64 if foci.dtype == np.int64 else torch.float32)

    return x_visual, x_audio, x_manual, x_focal


def logits_to_pi(x_focus: torch.Tensor, x_mkbd: torch.Tensor) -> tuple[Categorical, ...]:
    """
    Convert tensors of focal and mouse/keyboard state logits to multiple
    categorical distributions.
    """

    x_focus = x_focus.flatten(1)
    x_kbd = torch.sigmoid(x_mkbd[:, :19, None])
    x_kbd = torch.cat((1. - x_kbd, x_kbd), dim=-1)
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
    """A client that exchanges observations and actions with an external actor."""

    # Reaction time in number of steps
    N_DELAY = 6

    # Categories
    MOUSE_BINS = [
        -108.0, -72.73, -48.87, -32.73, -21.82, -14.43, -9.44, -6.06, -3.78, -2.23, -1.19, -0.48,
        0., 0.48, 1.19, 2.23, 3.78, 6.06, 9.44, 14.43, 21.82, 32.73, 48.87, 72.73, 108.]

    MOUSE_WHEEL = [-1, 0, 1]

    # Polled action before actions can be obtained from the model
    NULL_ACT = ([0, 0, 0, 0, 0, 0, 0, 160, 54, 1, 1, 1, GameID.VIEW_LOBBY, GameID.NULL, MapID.PLAYER_ID_NULL, 0.], None)

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

    cutout_centre_indices: tuple[int, int]
    mkbd_state: list[int]
    space_time: float
    action_skip_ctr: int

    def __init__(self, args: Namespace):
        super().__init__(args)

        # Sampling conditions
        self.action_skip = args.action_skip

        # Last action state
        self.observations: deque[tuple[np.ndarray, np.ndarray, list[int], tuple[int, int]]] = deque()
        self.actions: deque[tuple[int, int, np.ndarray, float, float, int]] = deque()
        self.reset_action_state()

        # Cutout centre tracking
        if args.focus_path is not None and args.record:
            self.focus = FocusTracker(path=args.focus_path, mode=FocusTracker.MODE_WRITE, start_active=True)
        else:
            self.focus = None

    def reset_action_state(self):
        self.observations.clear()
        self.actions.clear()
        self.cutout_centre_indices = (27, 80)
        self.mkbd_state = [0]*20
        self.space_time = 0.
        self.action_skip_ctr = 0

    def poll_user_input(self, timestamp: float) -> tuple[list, list | None]:
        """
        Poll or read peripheral events and interpret them as user input
        and optional local log data.
        """

        # Wait for match to start and action queue to fill
        if not self.session.phase or len(self.actions) <= self.N_DELAY:
            return self.NULL_ACT

        # Pop earliest action from queue
        frow, fcol, kbd, mmot_yrel, mmot_xrel, wheel_y = self.actions.popleft()

        # Update tracked focal point
        if self.focus is not None:
            self.focus.update(
                2*(frow-self.cutout_centre_indices[0]),
                2*(fcol-self.cutout_centre_indices[1]))

        self.cutout_centre_indices = frow, fcol

        # Keys
        lbtn, rbtn, space, ekey, gkey, rkey, dkey, akey, wkey, skey, tab, xkey, ckey, bkey = kbd[:14]
        skey *= -1
        akey *= -1

        kbd_nums = kbd[14:]
        draw_slot = kbd_nums.tolist().index(1) if any(kbd_nums) else 0

        # Reduce held button update freq. with eval. skip to prevent overly repeated switching
        if self.action_skip:
            if self.action_skip_ctr >= self.action_skip:
                self.action_skip_ctr = 0

            else:
                self.action_skip_ctr += 1

                # Override button actions
                # NOTE: ekey resolved below
                gkey = 0
                bkey = bkey if self.mkbd_state[self.MKBD_IDX_B] else 0
                xkey = xkey if self.mkbd_state[self.MKBD_IDX_X] else 0
                ckey = ckey if self.mkbd_state[self.MKBD_IDX_C] else 0
                tab = tab if self.mkbd_state[self.MKBD_IDX_TAB] else 0

        # x/c/b/tab
        sim = self.sim
        log = None

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
        sim.cursor_y = min(max(sim.cursor_y + mmot_yrel, 2.), 105.)

        if sim.view == GameID.VIEW_WORLD and not self.session.is_dead_or_spectator(sim.own_player_id):
            sim.cursor_x = 159.5
            d_angle = mmot_xrel
        else:
            sim.cursor_x = min(max(sim.cursor_x + mmot_xrel, 66.), 253.)
            d_angle = 0.

        # Get drawn item id from key num
        own_player = self.session.players[sim.own_player_id]

        draw_id = own_player.get_next_item_by_slot(draw_slot) if draw_slot else GameID.NULL

        # Get hovered (entity) ID
        hovered_id = GameID.NULL
        hovered_entity_id = MapID.PLAYER_ID_NULL

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
        if self.action_skip_ctr:
            can_plant_or_defuse = hovered_id == GameID.ITEM_C4 or \
                (own_player.held_object is not None and own_player.held_object.item.id == GameID.ITEM_C4)

            ekey = 1 if self.mkbd_state[self.MKBD_IDX_E] and can_plant_or_defuse else ekey

        # Assemble state
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
        if not self.session.phase or self.own_entity.team == GameID.GROUP_SPECTATORS:
            return

        # Get observations
        self.sim.eval_effects(dt * self.time_scale)

        # Log focus
        if self.focus is not None:
            self.focus.register(None, self.recorder.counter)

        # Prepare data for model inference
        frame = self.sim.get_frame()
        sound = self.sim.get_sound()

        self.observations.append((frame, sound, self.mkbd_state, self.cutout_centre_indices))

    def cleanup(self):
        if self.stats is not None:
            path_to_stats = self.stats.save()

            self.logger.info('Stats saved to: %s', path_to_stats)


class SDGActorClient:
    """
    An actor manager using asynchronous (two-threaded) local inference
    to infer actions of a trained PCNet model and update their client states.
    """

    def __init__(self, args: Namespace):
        self.sampling_prob = args.sampling_prob
        self.rng = np.random.default_rng(args.seed)
        self.event_loop = asyncio.get_event_loop()

        # Init. actors
        self.actor_keys = list(range(args.n_actors))
        self.actors: list[SDGBaseActor] = []

        for i in range(args.n_actors):
            args.name = f'ai{i:02d}'
            args.seed = None if args.seed is None else (args.seed + i + 1)

            self.actors.append(actor := SDGBaseActor(args))

            # Prevent duplication beyond the first actor
            if i == 0:
                actor.logger.name = 'AIClient'
                args.recording_path = None
                args.logging_level = ERROR

            else:
                actor.logger.name = f'AIClient{i:02d}'

        # Audio filters
        self.mel_basis = get_mel_basis(SAMPLING_RATE)
        self.ham_window: np.ndarray = np.hamming(int(SAMPLING_RATE/args.tick_rate + (HRIR_LEN-1)*2))
        self.db_ref = self.ham_window.sum()**2 / 2.

        # Load model
        if args.seed is not None:
            torch.manual_seed(args.seed)

        self.model = PCNet()
        self.device = torch.device(args.device)

        if args.checkpoint_path is not None:
            self.model = self.model.load(args.checkpoint_path, device=self.device)

        self.model.eval()

    async def infer(self):
        obs_queues = [actor.observations for actor in self.actors]

        # All queues must have at least one observation
        if not all(obs_queues):
            return

        # Gather observations
        frames, sounds, mkbds, foci = [], [], [], []

        for obs_queue in obs_queues:
            frame, sound, mkbd, focus = obs_queue.popleft()

            frames.append(frame)
            sounds.append(sound)
            mkbds.append(mkbd)
            foci.append(focus)

        # Batch observations
        frames = np.stack(frames)
        sounds = np.stack(sounds)
        mkbds = np.stack(mkbds)
        foci = np.stack(foci)

        # Preprocess
        spectra = spectrify(sounds, self.mel_basis, self.ham_window, SAMPLING_RATE, eps=MIN_AMP, ref=self.db_ref)

        x_visual, x_audio, x_manual, x_focal = prepare_inputs(frames, spectra, mkbds, foci, self.device)

        # Run forward pass
        with torch.inference_mode():
            x_focus, x_action = self.model(x_visual, x_audio, x_manual, x_focal, self.actor_keys, detach=True)

        # Extract sub-actions from logits
        if self.sampling_prob and (self.sampling_prob == 1. or self.rng.random() < self.sampling_prob):
            cat_focus, cat_kbd, cat_yrel, cat_xrel, cat_whly = logits_to_pi(x_focus, x_action)

            a_focus = cat_focus.sample()
            a_kbd = cat_kbd.sample()
            a_yrel = cat_yrel.sample()
            a_xrel = cat_xrel.sample()
            a_whly = cat_whly.sample()

        else:
            a_focus = x_focus.flatten(1).argmax(-1)
            a_kbd = (x_action[:, :19].sigmoid() >= 0.5).to('cpu', dtype=torch.int64)
            a_yrel = x_action[:, 19:44].argmax(-1)
            a_xrel = x_action[:, 44:69].argmax(-1)
            a_whly = x_action[:, 69:72].argmax(-1)

        a_focus = a_focus.cpu().numpy()
        n_cols = x_focus.shape[-1]
        a_frow = a_focus // n_cols
        a_fcol = a_focus % n_cols

        a_kbd = a_kbd.cpu().numpy()
        a_yrel = np.take(SDGBaseActor.MOUSE_BINS, a_yrel.cpu().numpy())
        a_xrel = np.take(SDGBaseActor.MOUSE_BINS, a_xrel.cpu().numpy())
        a_whly = np.take(SDGBaseActor.MOUSE_WHEEL, a_whly.cpu().numpy())

        # Update action queues
        for actor, *action in zip(self.actors, a_frow, a_fcol, a_kbd, a_yrel, a_xrel, a_whly):
            actor.actions.append(action)

    async def step(self, dt_loop: float, current_clock: float):
        for actor in self.actors:
            actor.step(dt_loop, current_clock)

    def run(self) -> int:
        spectator = self.actors[0]
        spectator.logger.info('Running...')

        for actor in self.actors:
            actor.session_running = True

        previous_clock: float = None
        current_clock: float = None
        session_time = 0.

        try:
            while spectator.session_running:

                # Update loop timekeeping
                current_clock = spectator._clock()
                dt_loop = (current_clock - previous_clock) if previous_clock is not None else 0.
                previous_clock = current_clock

                # Update recorder counter and timestamp
                spectator.recorder.update_meta(current_clock)

                # Advance local state
                self.event_loop.run_until_complete(asyncio.gather(self.step(dt_loop, current_clock), self.infer()))

                # Reset model state
                if spectator.session.time < session_time:
                    self.model.clear()

                    for actor in self.actors:
                        actor.reset_action_state()

                session_time = spectator.session.time

                # Cache and squeeze records
                spectator.recorder.cache_chunks()
                spectator.recorder.squeeze()

                # Delay to target specified FPS
                spectator._fps_limiter.update_and_delay(spectator._clock() - current_clock, current_clock)

        except KeyboardInterrupt:
            spectator.logger.debug('Process ended by user.')

        except (ConnectionError, TimeoutError):
            spectator._log_error('Lost connection to the server.')

        else:
            spectator.logger.debug('Session ended.')

            # Explicitly send any final messages still in queue due to sending stride
            if current_clock is not None:
                for actor in self.actors:
                    actor._send_client_data(current_clock)

                # Include slight delay to allow them to reach the server before disconnecting
                spectator._fps_limiter.delay(0.5)

        # Saving and cleanup
        if spectator.recorder.file_path is not None:
            spectator.recorder.restore_chunks()
            spectator.recorder.squeeze(all_=True)
            spectator.recorder.save()

            spectator.logger.info("Recording saved to: '%s'.", spectator.recorder.file_path)

        for actor in self.actors:
            actor._socket.close()
            actor.cleanup()

        spectator.logger.info('Stopped.')

        return 0
