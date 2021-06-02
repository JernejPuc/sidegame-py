"""WIP script to train a SDG AI agent with behavioural cloning"""

import os
import sys
from collections import deque
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sdgai.model import PCNet
from sdgai.utils import supervised_loss, Dataset


def train_model(
    model: torch.nn.Module,
    dataset: Dataset,
    decimation: int = 4,
    log_steps: int = 50,
    name: str = 'pc-sup'
):
    """
    Uses epochwise BPTT instead of TBPTT, decimating backwards ops to
    approximate it for different k1 and k2. Compared to TBPTT, gradients
    will be less consistent, but it is simpler to code and more efficient to
    execute (because only one set of states needs to be tracked for detachment).
    """

    k2 = dataset.truncated_length
    k1 = k2 // decimation
    assert (k2 % k1) == 0, 'Final forward and backward steps must be synchronised.'

    writer = SummaryWriter(os.path.join('runs', name))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.995), weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 0.01, total_steps=dataset.max_steps_with_repeat, pct_start=0.34,
        div_factor=20., final_div_factor=500.)

    init_focus = torch.ones((len(dataset.sequences), 2), dtype=torch.long, device=dataset.device)
    init_focus[:, 0] = 27
    init_focus[:, 1] = 80
    delayed_foci = deque(init_focus for _ in range(6))

    running_train_loss = 0.
    last_train_loss = np.Inf

    for i, data in enumerate(dataset, start=1):
        # Print out progress
        print(f'\rStep {i} of {dataset.max_steps_with_repeat}. Last score: {last_train_loss:.4f}    ', end='')

        # Handle reset/repeated sequences by setting initial states
        if dataset.reset_keys:
            model.clear(keys=dataset.reset_keys)
            cleared_focus_indices = list(dataset.reset_indices)

            for focus in delayed_foci:
                focus[cleared_focus_indices] = init_focus[cleared_focus_indices]

            dataset.reset_keys.clear()
            dataset.reset_indices.clear()

        # Reset accumulated gradients
        optimizer.zero_grad()

        # Loop over the temporal dimension
        (images, spectra, mkbds, foci, keys), actions = data

        for j in range(k2):
            # Demo output
            demo_focus = foci[j]
            demo_action = actions[j]

            delayed_foci.append(demo_focus)

            # Model output
            x_focus, x_action = model(images[j], spectra[j], mkbds[j], delayed_foci.popleft(), keys[j])

            # Compute loss and accumulate gradients
            if not (j+1) % k1:
                loss = supervised_loss(x_focus, x_action, demo_focus, demo_action)

                # Retain graph until the final backward operation
                if (j+1) == k2:
                    loss.backward()

                else:
                    loss.backward(retain_graph=True)

        # Update model and learning schedule
        optimizer.step()
        scheduler.step()

        # Detach final (first) hidden/cell states of LSTM cells for TBPTT
        model.detach(keys=keys[-1])

        # Log running train loss
        running_train_loss += loss.item()

        if not i % log_steps:
            last_train_loss = running_train_loss / log_steps
            running_train_loss = 0.

            writer.add_scalar('imitation loss', last_train_loss, i)

    print(f'\rFinished step {i} of {dataset.max_steps_with_repeat}. Last score: {last_train_loss:.4f}    ', end='')


if __name__ == '__main__':
    assert len(sys.argv) > 4, 'Not enough input arguments.'

    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_name = sys.argv[3]
    steps = int(sys.argv[4])

    data = [np.load(os.path.join(data_dir, filename)) for filename in os.listdir(data_dir) if filename.endswith('npz')]

    assert data, 'No data found in given directory.'

    seed = 42
    torch.manual_seed(seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = Dataset(
        data,
        truncated_length=16,
        max_steps_with_repeat=steps,
        max_batch_size=16,
        seed=seed,
        device=device)

    model = PCNet()

    if torch.cuda.is_available():
        model = model.move(device)

    try:
        train_model(model, dataset, name=model_name)

    except KeyboardInterrupt:
        print('\nTraining interrupted. Saving intermediate model parameters...')

    if torch.cuda.is_available():
        model = model.move(torch.device('cpu'))

    model.save(os.path.join(output_dir, model_name + '.pth'))

    print('Done.\n')
