"""Run basic RL in SDG."""

import argparse
from logging import DEBUG
from sdgai.reinforcement import run_rl_controller


def parse_args() -> argparse.Namespace:
    """Parse training args."""

    parser = argparse.ArgumentParser(description='Argument parser for supervised training from SDG demo files.')

    parser.add_argument(
        '-m', '--model_dir', type=str, default='models',
        help='Path to which model checkpoints will be written.')
    parser.add_argument(
        '-n', '--model_name', type=str, required=True,
        help='Name under which model checkpoints and events will be saved.')
    parser.add_argument(
        '-c', '--checkpoint_path', type=str, default=None,
        help='Path to parameters used to initialise the model before training.')
    parser.add_argument(
        '-r', '--resume_step', type=int, default=0,
        help='Step at which to resume the cycle of a learning rate scheduler.')
    parser.add_argument(
        '-l', '--logdir', type=str, default='runs',
        help='Path to which training events will be logged.')
    parser.add_argument(
        '--logging_path', type=str, default=None,
        help='If given, execution logs are written to a file at the specified location instead of stdout.')
    parser.add_argument(
        '--logging_level', type=int, default=DEBUG,
        help='Threshold above the severity of which the runtime messages are logged or displayed.')

    parser.add_argument(
        '--lr_init', type=float, default=5e-5,
        help='Initial learning rate in a scheduled cycle.')
    parser.add_argument(
        '--lr_max', type=float, default=1e-3,
        help='Peak learning rate in a scheduled cycle.')
    parser.add_argument(
        '--lr_final', type=float, default=1e-6,
        help='Final learning rate in a scheduled cycle.')
    parser.add_argument(
        '--pct_start', type=float, default=0.25,
        help='Ratio of the cycle at which the learning rate should peak.')
    parser.add_argument(
        '--beta1', '--beta1_base', type=float, default=0.8,
        help='(Base) 1st momentum-related parameter for the Adam optimiser.')
    parser.add_argument(
        '--beta1_max', type=float, default=0.9,
        help='(Max) 1st momentum-related parameter for the Adam optimiser.')
    parser.add_argument(
        '--beta2', type=float, default=0.975,
        help='2nd momentum-related parameter for the Adam optimiser.')
    parser.add_argument(
        '--weight_decay', type=float, default=5e-5,
        help='Regularisation parameter for the AdamW optimiser.')
    parser.add_argument(
        '--clip_grad_val', type=float, default=4.,
        help='Bandaid to mitigate exploding gradients by clipping them per module/parameter on backward pass. '
        'This distorts the gradients which can cause problems when propagated further.')
    parser.add_argument(
        '--clip_grad_norm', type=float, default=8.,
        help='Bandaid to mitigate exploding gradients by limiting their collective magnitude. '
        'As this is performed after backwarding, some gradients could have already exploded enough to cause issues.')

    parser.add_argument(
        '--pool_size', type=int, default=15,
        help='Number of different sequences that a training node can sample from.')
    parser.add_argument(
        '--batch_size', type=int, default=12,
        help='Number of different sequences processed simultaneously per each training node.')
    parser.add_argument(
        '--slice_length', '--k1', '--k2', type=int, default=30,
        help='Length of the longest differentiable sequence and number of steps between updates in epochwise BPTT.')
    parser.add_argument(
        '--eval_stride', '--k3', type=int, default=1,
        help='Number of steps between loss evaluation in epochwise TBPTT.')
    parser.add_argument(
        '--exp_length_on_reset', type=float, default=1.015,
        help='Start with sub-sequences of `slice_length`, then exponentiate their length (with rounding) '
        'to iteratively increase overall sequence length until they can be processed in full. '
        'Intended to vary and decorrelate initial batches and corresponding updates.')
    parser.add_argument(
        '--reduce_sum', action='store_true',
        help='Whether to reduce step-wise losses in epochwise TBPTT with summation or averaging.')

    parser.add_argument(
        '--steps', type=int, default=int(160e+3),
        help='Maximum number of steps within a training session.')
    parser.add_argument(
        '--save_steps', type=int, default=2,
        help='Step interval for saving current model parameters.')
    parser.add_argument(
        '--branch_steps', type=int, default=int(10e+3),
        help='Step interval for starting a new branch, i.e. path to save current model parameters.')
    parser.add_argument(
        '--log_steps', type=int, default=100,
        help='Step interval for computing and logging the running loss.')

    parser.add_argument(
        '--seed', type=int, default=42,
        help='Seed for initialising random number generators.')
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Option to launch training on (potentially) multiple GPUs or as a single CPU process.')
    parser.add_argument(
        '--max_nprocs', type=int, default=4,
        help='Limit the number of GPU devices that partake in training.')

    parser.add_argument('--update_steps', type=int, default=1, help='Number of updates per batch of trajectories.')
    parser.add_argument('--value_weight', type=float, default=0.5, help='Value loss term weight.')
    parser.add_argument('--entropy_weight', type=float, default=0.001, help='Entropy loss term weight.')

    parser.add_argument(
        '--time_scale', type=float, default=1., help='Simulation time factor affecting movement and decay formulae.')
    parser.add_argument(
        '--tick_rate', type=float, default=30.,
        help='Rate of updating the local game state in ticks (frames) per second.')
    parser.add_argument(
        '--refresh_rate', type=float, default=30.,
        help='Rate of updating the action queue in inference calls per second.')
    parser.add_argument(
        '--polling_rate', type=float, default=30.,
        help='Rate of accessing the action queue in actions per second.')
    parser.add_argument(
        '--sending_rate', type=float, default=30.,
        help='Rate of sending messages to the server in packets per second.')

    parser.add_argument('--address', type=str, default='localhost', help='Server address.')
    parser.add_argument('--port', type=int, default=49152, help='Server port.')

    parser.add_argument(
        '--record', '--rec', action='store_true',
        help='Record network data exchanged with the server and save it at the end of runtime.')
    parser.add_argument(
        '--recording_path', type=str, default=None,
        help='Path to an existing recording of network data exchanged with the server.')
    parser.add_argument(
        '--focus_path', type=str, default=None, help='Path to a recording of focal coordinates.')
    parser.add_argument(
        '--track_stats', action='store_true', help='Keep track of accumulated in-game values, '
        'which can be used to analyse the success and tendencies of the player.')
    parser.add_argument(
        '--show_fps', action='store_true',
        help='Print tracked frames-per-second to stdout.')
    parser.add_argument(
        '--discretise_mouse', action='store_true',
        help='Has no effect: Polled mouse movements are always based on discrete presets.')

    parser.add_argument(
        '--render_scale', type=float, default=1,
        help='Has no effect: The base render is never upscaled, i.e. its width and height are fixed.')
    parser.add_argument(
        '--mouse_sensitivity', type=float, default=1.,
        help='Has no effect: Mouse movement, i.e. pixel distance traversed between updates, is never augmented.')
    parser.add_argument(
        '--volume', type=float, default=1.,
        help='Initial factor by which in-game sound amplitudes are scaled.')
    parser.add_argument(
        '--interp_ratio', type=float, default=2.,
        help='Ratio between kept states for entity interpolation and the update rate of the server. '
        'Corresponds to the amount of artificial lag introduced to the client.')

    parser.add_argument(
        '--role_key', type=str, default='00000000',
        help='Role key used to introduce a client to the server. Used for authentication and to limit user privileges.')
    parser.add_argument(
        '--name', '--player_id', type=str, default='ai00',
        help='4-character name used to introduce a client to the server. Used to distinguish between clients.')
    parser.add_argument(
        '--mmr', type=float, default=1000.,
        help='Matchmaking rating (MMR) used to route a client through the matchmaking server. '
        'If 0, the address and port are assumed to belong to a session server, which will be contacted directly.')

    # Actor-specific args
    parser.add_argument('--n_actors', type=int, default=2, help='Number of agent copies to spawn.')
    parser.add_argument(
        '--sampling_proba', '--sp', type=float, default=0.01,
        help='Probability of sampling to get actions from probabilities instead of argmax. '
        '0 corresponds to argmax and 1 to sampling on every step.')
    parser.add_argument(
        '--sampling_thr', '--st', type=float, default=0.1,
        help='Determines the upper and lower probability thresholds between which actions can be sampled.')
    parser.add_argument('--expose', action='store_true', help='Store intermediate tensors.')
    parser.add_argument('--strict', action='store_true', help='Load model params strictly.')
    parser.add_argument('--simple', action='store_true', help='Use local serial inference.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_rl_controller(args)
