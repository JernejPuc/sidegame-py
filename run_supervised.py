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
        '--lr_init', type=float, default=1e-3,
        help='Initial learning rate in a scheduled cycle.')
    parser.add_argument(
        '--lr_max', type=float, default=2e-2,
        help='Peak learning rate in a scheduled cycle.')
    parser.add_argument(
        '--lr_final', type=float, default=5e-6,
        help='Final learning rate in a scheduled cycle.')
    parser.add_argument(
        '--pct_start', type=float, default=0.34,
        help='Ratio of the cycle at which the learning rate should peak.')
    parser.add_argument(
        '--beta1', type=float, default=0.9,
        help='1st momentum-related parameter for the Adam optimiser.')
    parser.add_argument(
        '--beta2', type=float, default=0.995,
        help='2nd momentum-related parameter for the Adam optimiser.')
    parser.add_argument(
        '--weight_decay', type=float, default=1e-5,
        help='Regularisation parameter for the AdamW optimiser.')

    parser.add_argument(
        '--pool_size', type=int, default=24,
        help='Number of different sequences that a training node can sample from.')
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Number of different sequences processed simultaneously per each training node.')
    parser.add_argument(
        '--truncated_length', type=int, default=30,
        help='Length of the longest differentiable sequence in epochwise BPTT.')
    parser.add_argument(
        '--decimation', type=int, default=6,
        help='Determines several sub-sequence lengths in epochwise BPTT as its multiples, up to the truncated length. '
        'If equal to truncated length, training regime becomes TBPTT with `k1 == k2`. '
        'If set to 1, a backward pass will be made on every step, retaining graph until the last step in the sequence.')
    parser.add_argument(
        '--steps', type=int, default=int(1e+5),
        help='Maximum number of steps within a training session.')
    parser.add_argument(
        '--save_steps', type=int, default=250,
        help='Step interval for saving current model parameters.')
    parser.add_argument(
        '--branch_steps', type=int, default=int(1e+4),
        help='Step interval for starting a new branch, i.e. path to save current model parameters.')
    parser.add_argument(
        '--log_steps', type=int, default=50,
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

    Uses epochwise BPTT instead of TBPTT, decimating backwards ops to
    approximate it for different `k1` and `k2`. Compared to TBPTT, gradients
    will be less consistent, but it is simpler to code and more efficient to
    execute (because only last states need to be tracked for detachment).

    Processing nodes can each be assigned a subset of available demonstration
    sequences. If the subsets overlap, there can be some redundancy in the
    computed gradients. If they are exclusive, there can be less variety
    in constructed batches (because a sequence is restricted to some node
    and must progress chronologically, its frames will only be batched with
    a handful of different others).
    """

    is_distributed = world_size > 1
    device = f'cuda:{rank}' if args.device == 'cuda' and is_distributed else args.device

    is_main = device in ('cuda', f'cuda:{MAIN_RANK}', 'cpu')
    device = torch.device(device)

    logger = get_logger('main' if is_main else f'aux:{rank}', path=args.logging_path, level=args.logging_level)
    logger.propagate = False

    # Determine subset of available sequences
    data_files = [filename for filename in os.listdir(args.data_dir) if filename.endswith('.h5')]

    if is_distributed:
        pool_idx = round(rank * (len(data_files) - args.pool_size) / (world_size - 1))
        file_slice = slice(pool_idx, pool_idx + args.pool_size)

    else:
        file_slice = slice(len(data_files))

    data_files = [h5py.File(os.path.join(args.data_dir, filename), 'r') for filename in data_files[file_slice]]

    assert data_files, 'No data for file slice in given directory.'

    # Same seed for initialising model weights
    torch.manual_seed(args.seed)

    model = PCNet()

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
    sleep(0.05 * rank)

    with torch.no_grad():
        checksum = (model.module if is_distributed else model).motor_decoding.action_fc.weight.sum().item()

    logger.debug('Weight checksum: %f', checksum)

    sleep(0.05 * world_size)

    # Different seeds for sampling data
    dataset = Dataset(
        data_files,
        truncated_length=args.truncated_length,
        max_steps_with_repeat=args.steps,
        max_batch_size=args.batch_size,
        resume_step=args.resume_step,
        seed=(args.seed + rank),
        device=device)

    init_focus = torch.tensor([CENTRE_FOCUS_Y, CENTRE_FOCUS_X], dtype=torch.long, device=device)
    delayed_foci = {seq.key: deque(init_focus for _ in range(N_DELAY)) for seq in dataset.sequences}

    k2 = dataset.truncated_length
    k1 = k2 // args.decimation
    assert (k2 % k1) == 0, 'Final forward and backward steps must be synchronised.'

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
        div_factor=(args.lr_max/args.lr_init),
        final_div_factor=(args.lr_init/args.lr_final),
        last_epoch=(args.resume_step-1))

    start_time = perf_counter()
    model_branch = 0
    running_train_loss = 0.
    last_train_loss = np.Inf

    logger.debug('Training...')

    for i, data in enumerate(dataset, start=1):
        if termination_event.wait(0.):
            break

        # Print out progress
        if is_main:
            effective_step = args.resume_step + i
            running_time = perf_counter() - start_time
            remaining_time = min(int(running_time * (args.steps - effective_step) / i), 99*24*3600)

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

        # Loop over the temporal dimension
        (images, spectra, mkbds, foci, keys), actions = data

        for j in range(k2):
            # Demo output
            demo_focus = foci[j]
            demo_action = actions[j]

            # NOTE: Current focus and action respond to observations from `N_DELAY` steps ago, including past focus
            key_list = keys[j].tolist()

            for key in keys[j]:
                delayed_foci[key].append(demo_focus[key_list.index(key)])

            foci_j = torch.stack([delayed_foci[key].popleft() for key in keys[j]])

            # Model output
            x_focus, x_action = model(images[j], spectra[j], mkbds[j], foci_j, keys[j])

            # Compute loss and accumulate gradients
            if not (j+1) % k1:
                loss = supervised_loss(x_focus, x_action, demo_focus, demo_action)

                # Retain graph and hold off sync until the final backward operation
                if (j+1) == k2:
                    loss.backward()

                elif is_distributed:
                    with model.no_sync():
                        loss.backward(retain_graph=True)

                else:
                    loss.backward(retain_graph=True)

        # Update model and learning schedule
        optimizer.step()
        scheduler.step()

        # Detach final (first) hidden/cell states of LSTM cells for TBPTT
        (model.module if is_distributed else model).detach(keys=keys[-1])

        # Log running train loss
        running_train_loss += loss.item()

        if not i % args.log_steps:
            last_train_loss = running_train_loss / args.log_steps
            running_train_loss = 0.

            writer.add_scalar('imitation loss', last_train_loss, global_step=(args.resume_step + i))

        # Save model checkpoint
        if is_main and not i % args.branch_steps:
            model_branch += 1

        if is_main and not i % args.save_steps:
            (model.module if is_distributed else model).save(
                os.path.join(args.model_dir, args.model_name + f'_v{model_branch:3d}.pth'))

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
        while not termination_event.wait(0.):
            sleep(INTERRUPT_CHECK_PERIOD)

    except KeyboardInterrupt:
        termination_event.set()

        print()
        root_logger.debug('Training interrupted by user.')

    while not ctx.join():
        pass

    root_logger.info('Done.')
