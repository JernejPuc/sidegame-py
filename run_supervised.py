#!/usr/bin/env python

"""Train a SDG AI agent with behavioural cloning."""

import os
import argparse
from logging import DEBUG
from collections import deque
from time import sleep, perf_counter
from datetime import timedelta
import numpy as np
import h5py
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from sdgai.model import PCNet
from sdgai.utils import supervised_loss, Dataset
from sidegame.networking.core import get_logger


N_DELAY = 6
CENTRE_FOCUS_Y = (108/2) / 2
CENTRE_FOCUS_X = (64 + 192/2) / 2
INTERRUPT_CHECK_PERIOD = 1.
MAIN_RANK = 0
RANK_DELAY = 0.05
MAX_DISP_SECONDS = 99*24*3600


def parse_args() -> argparse.Namespace:
    """Parse training args."""

    parser = argparse.ArgumentParser(description='Argument parser for supervised training from SDG demo files.')

    parser.add_argument(
        '-d', '--data_dir', type=str, required=True,
        help='Path to a directory of files with observations and corresponding actions.')
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
        '--lr_init', type=float, default=3e-5,
        help='Initial learning rate in a scheduled cycle.')
    parser.add_argument(
        '--lr_max', type=float, default=6e-4,
        help='Peak learning rate in a scheduled cycle.')
    parser.add_argument(
        '--lr_final', type=float, default=1e-6,
        help='Final learning rate in a scheduled cycle.')
    parser.add_argument(
        '--pct_start', type=float, default=0.15,
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
        '--steps', type=int, default=int(300e+3),
        help='Maximum number of steps within a training session.')
    parser.add_argument(
        '--mkbd_release_start', type=int, default=15000,
        help='Step until which mouse/keyboard input is suppressed.')
    parser.add_argument(
        '--mkbd_release_steps', type=int, default=1000,
        help='Number of steps over which mouse/keyboard input is increased to full value.')
    parser.add_argument(
        '--save_steps', type=int, default=500,
        help='Step interval for saving current model parameters.')
    parser.add_argument(
        '--branch_steps', type=int, default=int(15e+3),
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

    return parser.parse_args()


def setup(rank: int, world_size: int):
    """Initialise a distributed process group."""

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '49160'

    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def cleanup():
    """Terminate a distributed process group."""

    dist.destroy_process_group()


def run_imitation(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    termination_event: mp.Event
):
    """
    Synchronise between processing nodes to train an AI model with behavioural
    cloning based on demonstrations extracted from SDG recordings.

    Uses epochwise backpropagation through time (EBPTT) for recurrent elements.
    Compared to true TBPTT, epochwise variants will propagate gradients
    less consistently (a memory state can take part in `k3` to `k1` passes),
    but should be more efficient due to fewer backward and update steps.

    Processing nodes can each be assigned a subset of available demonstration
    sequences. If the subsets overlap, there can be some redundancy in the
    computed gradients. If they are exclusive, there can be less variety
    in constructed batches (because a sequence is restricted to some node
    and must eventually progress chronologically, its frames may only be
    batched with a handful of different others).
    """

    is_distributed = world_size > 1
    device = f'cuda:{rank}' if args.device == 'cuda' and is_distributed else args.device

    is_main = device in ('cuda', f'cuda:{MAIN_RANK}', 'cpu')
    device = torch.device(device)

    logger = get_logger('main' if is_main else f'aux:{rank}', path=args.logging_path, level=args.logging_level)
    logger.propagate = False

    # Determine subset of available sequences
    data_files = [filename for filename in sorted(os.listdir(args.data_dir)) if filename.endswith('.h5')]

    if is_distributed:
        pool_idx = round(rank * (len(data_files) - args.pool_size) / (world_size - 1))
        file_slice = slice(pool_idx, pool_idx + args.pool_size)

    else:
        file_slice = slice(len(data_files))

    data_files = data_files[file_slice]
    assert data_files, 'No data for file slice in given directory.'

    # Confirm file subset per node
    sleep(RANK_DELAY * rank)
    logger.debug('Last file and file number: %s (%s)', max(data_files), len(data_files))
    sleep(RANK_DELAY * world_size)

    data_files = [h5py.File(os.path.join(args.data_dir, filename), 'r') for filename in data_files]

    # Same seed for initialising model weights
    torch.manual_seed(args.seed)

    model = PCNet()

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
        setup(rank, world_size)
        model = DDP(model, device_ids=[], output_device=device)

    # Confirm weights are consistent by logging a checksum for some parameters
    sleep(RANK_DELAY * rank)

    with torch.no_grad():
        checksum = (model.module if is_distributed else model).motor_decoding.action_fc.weight.sum().item()

    logger.debug('Weight checksum: %f', checksum)

    sleep(RANK_DELAY * world_size)

    # NOTE: Different seeds for sampling data
    # NOTE: `args.truncated_length` includes the remainder in the computational graph, determining place of detachment
    dataset = Dataset(
        data_files,
        slice_length=args.slice_length,
        max_steps_with_repeat=args.steps,
        max_batch_size=args.batch_size,
        resume_step=args.resume_step,
        exp_length_on_reset=(args.exp_length_on_reset if args.exp_length_on_reset > 1. else None),
        seed=(args.seed + rank),
        device=device)

    init_focus = torch.tensor([CENTRE_FOCUS_Y, CENTRE_FOCUS_X], dtype=torch.long, device=device)
    delayed_foci = {seq.key: deque(init_focus for _ in range(N_DELAY)) for seq in dataset.sequences}

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
        torch.backends.cudnn.enabled = True

    start_time = perf_counter()
    model_branch = 0
    epoch_losses = deque()
    last_train_loss = np.Inf
    running_train_loss = 0.
    running_scalars = {
        'kbd': {},
        'mmot_y': {},
        'mmot_x': {},
        'mwhl_y': {},
        'focus': {},
        'num_alpha': {},
        'loss_per_key': {}}

    k1 = args.slice_length
    k3 = args.eval_stride
    assert k3 <= k1, f'`k3` must be less than or equal to `k1`, got `k3`: {k3} and `k1`: {k1}.'

    n_unrolled_steps = args.log_steps * k1

    logger.debug('Training...')

    for i, data in enumerate(dataset, start=1):
        if termination_event.is_set():
            break

        effective_step = args.resume_step + i

        # Print out progress
        if is_main:
            running_time = perf_counter() - start_time
            remaining_time = min(int(running_time * (args.steps - effective_step) / i), MAX_DISP_SECONDS)

            print(
                f'\rStep {effective_step} of {args.steps}. Last score: {last_train_loss:.4f}. ETA: ' +
                str(timedelta(seconds=remaining_time)) + '        ',
                end='')

        # Handle reset/repeated sequences by setting initial states
        if dataset.reset_keys:
            (model.module if is_distributed else model).clear(keys=dataset.reset_keys)

            for key in dataset.reset_keys:
                delayed_foci[key].clear()
                delayed_foci[key].extend(init_focus for _ in range(N_DELAY))

            dataset.reset_keys.clear()

        # Reset accumulated gradients
        optimizer.zero_grad()

        # Unpack data
        (images, spectra, mkbds, foci, keys), actions = data

        # Suppress mouse/keyboard input at the start to emphasise learning of audio-visual layers
        if effective_step < args.mkbd_release_start:
            mkbds = mkbds * 0.

        elif effective_step < (args.mkbd_release_start + args.mkbd_release_steps):
            mkbds = mkbds * ((effective_step - args.mkbd_release_start) / args.mkbd_release_steps)

        # Loop over the temporal dimension
        # NOTE: `k1 == dataset.slice_length`
        for j in range(k1):

            # Demo output
            demo_focus = foci[j]
            demo_action = actions[j]

            # NOTE: Current focus and action respond to observations from `N_DELAY` steps ago, including past focus
            key_list = keys[j].tolist()

            for key in keys[j]:
                delayed_foci[key].append(demo_focus[key_list.index(key)])

            demo_focus = demo_focus // 2

            foci_j = torch.stack([delayed_foci[key].popleft() for key in keys[j]])

            # Model output
            x_focus, x_action = model(images[j], spectra[j], mkbds[j], foci_j, keys[j])

            # Compute loss and keep data in memory
            if not (j+1) % k3:
                key_strings = [str(key) for key in keys[j]]
                train_loss, scalars = supervised_loss(x_focus, x_action, demo_focus, demo_action, keys=key_strings)
                epoch_losses.append(train_loss)

                for scalar_key, scalars_per_sequence in scalars.items():
                    running_scalars_per_sequence = running_scalars[scalar_key]

                    for sequence_key, scalar_val in scalars_per_sequence.items():
                        if sequence_key not in running_scalars_per_sequence:
                            running_scalars_per_sequence[sequence_key] = scalar_val

                        else:
                            running_scalars_per_sequence[sequence_key] += scalar_val

        # Get average or total loss
        # NOTE: Sources apparently point to summation, but this makes the values high and dependant on batch size:
        # https://stats.stackexchange.com/questions/219914/rnns-when-to-apply-bptt-and-or-update-weights/220111#220111
        loss = torch.stack(tuple(epoch_losses)).sum() if args.reduce_sum else torch.stack(tuple(epoch_losses)).mean()
        epoch_losses.clear()

        # Accumulate gradients and sync on backward pass
        loss.backward()

        # Additional gradient manipulation
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

        # Update model and learning schedule
        optimizer.step()
        scheduler.step()

        # Detach final (first) hidden/cell states of LSTM cells for EBPTT
        (model.module if is_distributed else model).detach(keys=keys[-1])

        # Log running train loss
        running_train_loss += loss.item()

        if not i % args.log_steps:
            last_train_loss = running_train_loss / args.log_steps
            running_train_loss = 0.

            writer.add_scalar('imitation loss', last_train_loss, global_step=effective_step)

            for scalar_key, scalars_per_sequence in running_scalars.items():
                for sequence_key in scalars_per_sequence:
                    scalars_per_sequence[sequence_key] /= n_unrolled_steps

                writer.add_scalars(scalar_key, scalars_per_sequence, global_step=effective_step)

                for sequence_key in scalars_per_sequence:
                    scalars_per_sequence[sequence_key] = 0.

        # Save model checkpoint
        if is_main and not i % args.branch_steps:
            model_branch += 1

        if is_main and not i % args.save_steps:
            (model.module if is_distributed else model).save(
                os.path.join(args.model_dir, args.model_name + f'_v{model_branch:03d}.pth'))

    # Save final model parameters
    if is_main:
        (model.module if is_distributed else model).save(
            os.path.join(args.model_dir, args.model_name + '_final.pth'))

        print(f'\nFinished {args.steps} steps. Last score: {last_train_loss:.4f}', end='')

    # Close data files
    for data_file in data_files:
        data_file.close()

    # Terminate process group
    if is_distributed:
        cleanup()

    # Inform root process to join
    if not termination_event.wait(0.):
        termination_event.set()


if __name__ == '__main__':
    args = parse_args()
    root_logger = get_logger('root', path=args.logging_path, level=args.logging_level)

    if args.device == 'cuda' and torch.cuda.is_available():
        nprocs = min(torch.cuda.device_count(), args.max_nprocs)

    else:
        args.device = 'cpu'
        nprocs = 1

    # NOTE: Explicit 'spawn' context to prevent unexpected segmentation fault
    # See: https://discuss.pytorch.org/t/problems-with-torch-multiprocess-spawn-and-simplequeue/69674/2
    tmp_ctx = mp.get_context('spawn')
    termination_event = tmp_ctx.Event()

    root_logger.info('Launching...')

    ctx = mp.spawn(run_imitation, args=(nprocs, args, termination_event), nprocs=nprocs, join=False, daemon=True)

    try:
        while not termination_event.is_set():
            sleep(INTERRUPT_CHECK_PERIOD)

    except KeyboardInterrupt:
        termination_event.set()

        print()
        root_logger.debug('Training interrupted by user.')

    while not ctx.join():
        pass

    root_logger.info('Done.')
