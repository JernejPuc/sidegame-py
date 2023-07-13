"""Compute model accuracy on extracted demo data."""

import os
import argparse
from collections import deque
import json
import h5py
import numpy as np
import torch
from sdgai.model import PCNet
from sdgai.utils import Dataset


N_DELAY = 6
MOUSE_WHEEL = [-1, 0, 1]
MOUSE_BINS = np.array([
    -108.0, -72.73, -48.87, -32.73, -21.82, -14.43, -9.44, -6.06, -3.78, -2.23, -1.19, -0.48,
    0., 0.48, 1.19, 2.23, 3.78, 6.06, 9.44, 14.43, 21.82, 32.73, 48.87, 72.73, 108.])


def parse_args() -> argparse.Namespace:
    """Parse input/model/output args."""

    parser = argparse.ArgumentParser(description='Argument parser for the model accuracy script.')

    parser.add_argument(
        '-d', '--dataset', type=str, required=True, help='Extracted data to run through.')
    parser.add_argument(
        '-m', '--model', type=str, required=True, help='Trained model to test.')
    parser.add_argument(
        '-o', '--output', type=str, required=True, help='Path, where results will be written to.')

    parser.add_argument(
        '-s', '--seed', type=int, default=42, help='Seed for initialising random number generators.')
    parser.add_argument(
        '-n', '--max_steps', type=int, default=int(1e+6), help='Option to end the process prematurely.')
    parser.add_argument(
        '-v', '--device', type=str, default=None, help='Model inference device.')

    return parser.parse_args()


def get_std(sum2: float, sum1: float, avg: float, num: int) -> float:
    """Compute standard deviation from its unrolled form."""

    return max(0., (sum2 - 2*sum1*avg + num*avg**2) / num)**0.5


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)

    if args.device is not None and args.device.startswith('cuda'):
        _ = torch.cuda.device_count()

    data_file = h5py.File(args.dataset, 'r')
    dataset = Dataset((data_file,), slice_length=1, seed=args.seed, device=args.device)

    scores = {
        'focal_dist_sum': 0.,
        'focal_dist_sum2': 0.,
        'focal_dist_avg': 0.,
        'focal_dist_std': 0.,

        'kbd_acc_avg': [0.]*19,
        'kbd_acc_sum': np.array([0.]*19),
        'kbd_pos_avg': [0.]*19,
        'kbd_pos_sum': np.array([0.]*19),
        'kbd_pos_num': np.array([0.]*19),

        'mmoty_acc_avg': 0.,
        'mmoty_pos_avg': 0.,
        'mmoty_acc_sum': 0.,
        'mmoty_pos_sum': 0.,
        'mmoty_pos_num': 0.,

        'mmoty_acc5_avg': 0.,
        'mmoty_pos5_avg': 0.,
        'mmoty_acc5_sum': 0.,
        'mmoty_pos5_sum': 0.,

        'mmotx_acc_avg': 0.,
        'mmotx_pos_avg': 0.,
        'mmotx_acc_sum': 0.,
        'mmotx_pos_sum': 0.,
        'mmotx_pos_num': 0.,

        'mmotx_acc5_avg': 0.,
        'mmotx_pos5_avg': 0.,
        'mmotx_acc5_sum': 0.,
        'mmotx_pos5_sum': 0.,

        'num': 0,
        'src_data': args.dataset,
        'src_model': args.model,
        'src_seed': args.seed}

    model = PCNet()
    model.load(args.model)

    if args.device is not None:
        model = model.move(args.device)

    init_focus = torch.ones((len(dataset.sequences), 2), dtype=torch.long, device=dataset.device)
    init_focus[:, 0] = 27
    init_focus[:, 1] = 80
    delayed_foci = deque(init_focus for _ in range(N_DELAY))

    frame_number = 1
    FRAME_TOTAL = len(data_file['focus'])

    with torch.no_grad():
        for data in dataset:
            perc = frame_number / FRAME_TOTAL * 100.
            print(f'\rProcessing tick {frame_number} of {FRAME_TOTAL} ({perc:.2f}%)          ', end='')

            if dataset.reset_keys or frame_number >= args.max_steps:
                break

            (images, spectra, mkbds, foci, keys), actions = data

            demo_focus = foci[0]
            demo_action = actions[0]

            delayed_foci.append(demo_focus)
            demo_focus = demo_focus // 2

            # Model output
            x_focus, x_action = model(images[0], spectra[0], mkbds[0], delayed_foci.popleft(), keys[0], detach=True)

            # Get distance to target focus
            x_focus = x_focus[0, 0]
            act_focus = x_focus.argmax().cpu()
            act_focus = np.unravel_index(act_focus, x_focus.shape)
            focal_dist = np.linalg.norm(np.array(act_focus) - demo_focus.cpu()[0].numpy())

            scores['focal_dist_sum'] += focal_dist
            scores['focal_dist_sum2'] += focal_dist**2

            # Get accuracy for keyboard status (any and pressed/positive samples)
            act_kbd = (torch.sigmoid(x_action[0, :19]) >= 0.5)
            pos_kbd = (act_kbd * demo_action[0, :19]).cpu().numpy()
            act_kbd = (act_kbd == demo_action[0, :19]).cpu().to(torch.float).numpy()

            scores['kbd_acc_sum'] += act_kbd
            scores['kbd_pos_sum'] += pos_kbd
            scores['kbd_pos_num'] += demo_action[0, :19].cpu().numpy()

            # Get (top-1 and top-5) accuracy for vertical mouse motion
            mmoty = x_action[0, 19:44].cpu().numpy()
            dmoty = demo_action[0, 19:44].argmax().cpu().item()

            pos_flag = dmoty != 12

            if pos_flag:
                scores['mmoty_pos_num'] += 1

            if dmoty == np.argmax(mmoty):
                scores['mmoty_acc_sum'] += 1

                if pos_flag:
                    scores['mmoty_pos_sum'] += 1

            if dmoty in np.argsort(mmoty)[::-1][:5]:
                scores['mmoty_acc5_sum'] += 1

                if pos_flag:
                    scores['mmoty_pos5_sum'] += 1

            # Get accuracy for horizontal mouse motion
            mmotx = x_action[0, 44:69].cpu().numpy()
            dmotx = demo_action[0, 44:69].argmax().cpu().item()

            pos_flag = dmotx != 12

            if pos_flag:
                scores['mmotx_pos_num'] += 1

            if dmotx == np.argmax(mmotx):
                scores['mmotx_acc_sum'] += 1

                if pos_flag:
                    scores['mmotx_pos_sum'] += 1

            if dmotx in np.argsort(mmotx)[::-1][:5]:
                scores['mmotx_acc5_sum'] += 1

                if pos_flag:
                    scores['mmotx_pos5_sum'] += 1

            # NOTE: Mouse wheel can be ignored

            scores['num'] += 1
            frame_number += 1

    # Compute accuracy from accumulated data
    scores['focal_dist_avg'] = scores['focal_dist_sum'] / scores['num']
    scores['focal_dist_std'] = get_std(
        scores['focal_dist_sum2'], scores['focal_dist_sum'], scores['focal_dist_avg'], scores['num'])
    scores['kbd_acc_avg'] = (scores['kbd_acc_sum'] / scores['num']).tolist()
    scores['kbd_acc_sum'] = scores['kbd_acc_sum'].tolist()
    scores['kbd_pos_avg'] = (scores['kbd_pos_sum'] / np.maximum(1., scores['kbd_pos_num'])).tolist()
    scores['kbd_pos_sum'] = scores['kbd_pos_sum'].tolist()
    scores['kbd_pos_num'] = scores['kbd_pos_num'].tolist()
    scores['mmoty_acc_avg'] = scores['mmoty_acc_sum'] / scores['num']
    scores['mmoty_acc5_avg'] = scores['mmoty_acc5_sum'] / scores['num']
    scores['mmoty_pos_avg'] = scores['mmoty_pos_sum'] / scores['mmoty_pos_num']
    scores['mmoty_pos5_avg'] = scores['mmoty_pos5_sum'] / scores['mmoty_pos_num']
    scores['mmotx_acc_avg'] = scores['mmotx_acc_sum'] / scores['num']
    scores['mmotx_acc5_avg'] = scores['mmotx_acc5_sum'] / scores['num']
    scores['mmotx_pos_avg'] = scores['mmotx_pos_sum'] / scores['mmotx_pos_num']
    scores['mmotx_pos5_avg'] = scores['mmotx_pos5_sum'] / scores['mmotx_pos_num']

    # Add to existing score file
    if os.path.exists(args.output):
        with open(args.output, 'r') as total_file:
            totals = json.load(total_file)

    else:
        totals = {}

    totals[data_file.attrs['src']] = scores

    with open(args.output, 'w') as total_file:
        json.dump(totals, total_file)
