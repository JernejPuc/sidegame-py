# SiDeGame - Simplified Defusal Game

*SiDeGame* is an asynchronous environment for training and evaluating AI agents
in a cooperation- and communication-based competitive setting, designed to be
consistent across human and AI interfaces.

<p float="center">
  <img src="user_data/screenshot_000.png" width="49%" />
  <img src="user_data/clip_000.gif" width="49%" alt="Highlight replay" /> 
</p>

<p float="center">
  <img src="user_data/screenshot_002.png" width="49%" />
  <img src="user_data/screenshot_003.png" width="49%" /> 
</p>


## Disclaimer

*SiDeGame* is inspired by
[Counter-Strike: Global Offensive](https://store.steampowered.com/app/730/CounterStrike_Global_Offensive/).

Due to technical limitations of distributed AI training processes,
particularly the restriction of environment simulations to CPU,
*SiDeGame* was created as a kind of a 2D imitation,
but aiming to preserve many of the original characteristics,
resulting in a pseudo-FPS experience with top-down view.

The game rules and balancing values were either obtained through various
sources on the internet, approximated through experimentation, or otherwise changed.
Note that many aspects of the game do not translate well to a 2D setting.
Most obviously, the loss of verticality limits the map pool
and how the world can be perceived.

Some assets from CSGO were retained. For example, an old radar image of `de_cache`
was modified and repurposed as a base for the in-game world, while a subset of
original game sounds was either copied or slightly modified. Additionally,
the positional audio implementation relies on data from
[The FABIAN head-related transfer function data base](http://dx.doi.org/10.14279/depositonce-5718.5).
All other assets, such as icons, sprites, the HUD, etc. were made by me.

Unless explicitly referenced in specific docstrings or notes in development notebooks,
the code is of my own making. Systems, such as positional audio or multiplayer networking,
were based on comments or documents written by Valve or members of online communities,
but did not build on any specific code.


## Installation

Start by downloading or cloning this repository.

If you already have `python` on your system, the packages listed in
`requirements.txt` should have their dependencies met or handled during setup,
but `pysdl2` and `pyaudio` might need some prior attention.


### Dependencies of dependencies

On Linux, you may need to execute the following:

```bash
sudo apt install libsdl2-dev libsdl2-2.0-0
sudo apt install portaudio19-dev
```

Should `pyaudio` fail to install on Windows, you could try installing it from a
[wheel](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio).

See the project descriptions of [pysdl2](https://pypi.org/project/PySDL2/)
and [pyaudio](https://people.csail.mit.edu/hubert/pyaudio/) for more details.


### Editable install

Afterwards, `sidegame` can be installed (in editable/development mode)
from the `sidegame-py` directory with:

```bash
pip install -e .
```

More streamlined methods of installation may be made available in the future,
e.g. in the form of wheels and/or a PyPI distribution.


## Running

### Human user interface

Assuming that `sidegame-py` is the current working directory,
run this command to establish a game session:

```bash
python run_server.py
```

To connect to it and play, clients should be run from separate terminal tabs/windows
or remote machines (if the network configuration allows it) with the following:

```bash
python run_client.py
```

The matchmaking server and replay client can be launched similarly. Note that
the latter does not require a server to bind to, but rather a pre-recorded
demo (network message history).


#### Parameter configuration

There are multiple ways to override default launch parameters.
The order of priority is as follows:
1. In-terminal argument specification
2. Selected or default sub-configuration in `user_data/config.json`
3. Defaults of the parser in `name-of-script.py`

Editing or adding a sub-configuration to `config.json` should be the most
convenient way to simplify repeated launches in customised settings.
In any case, you can execute `python name-of-script.py --help`
for details on optional arguments.


#### Controls and in-game commands

The controls adhere to the following scheme:

![controls](controls.png)

As a basic measure of organisation, clients are assigned a role, which
determines their privileges, i.e. available in-game commands in a session.
Additionally, there are some local commands, which do not interact with the
server and can be used freely.

0. Local commands:
    - `mouse`: Toggle whether the system mouse cursor is hidden and restricted to the window.
    - `stats`: Print out the summary of statistics if stat tracking is turned on.
    - `exit`: End (own) client process.

1. Spectator commands:
    - `set role ABCDEFGH`: Set role corresponding to an 8-character (hex) key.
    - `set name ABCD`: Change name to a 4-character string (of select characters).

2. Player commands:
    - `set team B`: Move yourself to team (group) `B`, i.e. `T`, `CT`, or `S`.
    - `ping`: Print out the round-trip latencies of all active players in the match.

3. Admin commands:
    - `set team A B`: Move client with ID `A` to team (group) `B`.
    - `set team N`: Move the first `N` clients into player teams.
    - `start`: Start the match.
    - `stop`: Stop the match.
    - `quit`: End the session, and both server and each connected client process along with it.
    - `dev mode`: Enable buying of items regardless of money, match phase, or distance to the spawn point,
    and prevent incoming damage from affecting health points.
    - `max money`: Set money to its cap value.
    - `rundown`: Run down the timer for the buy or main (plant) phase.
    - `add bot`: Add an unresponsive entity to the spectators group.
    - `kick A`: Remove client with ID `A` from the remainder of the session.


### AI actor interface

Actor implementations can inherit from the `sidegame.game.client.base::SDGLiveClientBase`
class and facilitate interaction with a pre-trained model, remote inference or
optimisation servers, etc.

As an example, `sdgai.actor::SDGActorClient` can run multiple instances of `models/pcnet-sl.pth`,
a model trained with imitation learning via `sdgai/run_supervised.py`.
Note that its performance is limited due to overfitting on a small dataset.

A reinforcement learning example replaces the clients and server with `sdgai.rl::SDGSyncEnv`,
a lock-step version of the environment without real-time networking constraints.
The implementation in `sdgai.rl` is based on [`discit`](https://github.com/jernejpuc/discit),
but no trained model is available to be used at this time.

The examples rely on additional `requirements-ai.txt` and `requirements-rl.txt`,
respectively.


### Notes on performance

`sidegame` should be able to be rendered at hundreds to a thousand of FPS
on a single thread (on average - drops can still occur), which should
make it light enough for AI training setups. In distributed setups,
note that network and device synchronisation also play a role.

For human interfaces, it is expected to be upscaled to higher resolutions,
using the argument `render_scale`. Here are a few scale-to-resolution correspondences:
- `0` -- **Fullscreen**
- `1` -- `256x144` **(original)**
- `4` -- `1024x576`
- `5` -- `1280x720`
- `6.25` -- `1600x900`
- `7.5` -- `1920x1080`
- `10` -- `2560x1440`


### Notes on networking

Allowing clients and servers to communicate within your network may require
you to configure the Firewall settings on participating devices, set up
port forwarding, etc.


## More screenshots

<p float="center">
  <img src="user_data/screenshot_004.png" width="49%" />
  <img src="user_data/screenshot_005.png" width="49%" /> 
</p>

<p float="center">
  <img src="user_data/screenshot_006.png" width="49%" />
  <img src="user_data/screenshot_007.png" width="49%" /> 
</p>

<p float="center">
  <img src="user_data/screenshot_008.png" width="49%" />
  <img src="user_data/screenshot_009.png" width="49%" /> 
</p>

<p float="center">
  <img src="user_data/screenshot_010.png" width="49%" />
  <img src="user_data/screenshot_011.png" width="49%" /> 
</p>


## Citation

`sidegame-py` was created in the scope of my [master's thesis](https://plus.si.cobiss.net/opac7/bib/75138819) (in Slovene):

```
@article{Puc_2021,
 title={Online benchmark environment for multi-agent reinforcement learning: Master's thesis},
 author={Puc, Jernej},
 year={2021},
 url={https://repozitorij.uni-lj.si/IzpisGradiva.php?lang=eng&id=129594}}
```

A much shorter [conference paper](https://plus.si.cobiss.net/opac7/bib/86401795) is available in English:

```
@inproceedings{Puc_Sadikov_2021,
 title={SiDeGame: An Online Benchmark Environment for Multi-Agent Reinforcement Learning},
 author={Puc, Jernej and Sadikov, Aleksander},
 booktitle={Slovenian Conference on Artificial Intelligence, Proceedings of the 24th international multiconference: Information Society - IS 2021, volume A},
 publisher={Institute Jožef Stefan},
 editor={Luštrek, Mitja and Gams, Matjaž and Piltaver, Rok},
 ISBN={2630-371X},
 ISSN={2630-371X},
 year={2021},
 month={oct},
 pages={35–38},
 url={http://library.ijs.si/Stacks/Proceedings/InformationSociety/2021/IS2021_Volume_A.pdf}}
```


## Going forward

After a longer period of inactivity, my priorities and obligations have changed
to the point where no further experiments or papers are planned for SiDeGame.

The prospect of revisiting it remains and any interest in it is welcome.
Hopefully, the latest round of updates will leave it at an accessible point.

The following points remain posted here as suggestions:


#### Gameplay
- Add more maps by modifying the original map for different team sizes:
  1v0 (aim practice), 1v1, 2v2, 3v3, 4v4, 5v5 (existing/original).
- Experiment with partial border or background colouring of the side panel
  based on type, sender or time to make messages stand out and aid panel navigation.
- Experiment with message wheel views to improve selection accuracy and speed.

#### Supervised pretraining
- Revise the current imitation learning implementation and model.
- Add option to extract segmentation maps from every frame.
- Pretrain the visual encoder part of the model on frame-segmentation pairs.

#### Reinforcement learning
- Revise the encoder, policy, and critic model architectures.
- Add trained policy models for each stage of a curriculum of transferring agents
  from smaller to larger maps and team sizes.


## Backlog

Other points to be addressed, with lower priority:


#### Optimisation
- Tweak the audio channel system to better handle many overlapping sounds.

#### Gameplay
- Add a command to fill all empty player positions with AI agents or dummies.
- Enable players to take over in place of a dummy entity.

#### Networking
- Add an event to signal the server to handle intended disconnections
  immediately instead of eventually catching a timeout or related errors.
- Keep the state of unexpectedly disconnected clients and allow them to resume from it
  if they manage to reconnect.

#### User interface
- Allow the client loop to run regardless of server connection,
  rendering the console/lobby view.
- Add options for the client to update and search through a list of known server IPs.

#### Replays
- Use state checkpoints to speed up jumps between time points.
- Include effects in state checkpoints for accurate replay when jumping.

#### Distribution
- Proper PyPI distribution and documentation.
