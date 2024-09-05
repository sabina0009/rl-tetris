# Learning to play Tetris with the Option-critic Architeture
This repository contains all the code developed during the course of the dissertation.

## Basic usage 
To run version 0 of PPO in 20x10 Tetris for 5 runs of 100,000 episodes, run the following command:

```
python main.py
```

## Other options
To run different versions, use the following arguments:

### Environments: 
Tetris, CartPole (not available for AOC), FourRooms (not available for AOC) 

### Agents: 
PPO, DQN, OC, PPOC, AOC

Board size: 20x10, 8x4 (both only for Tetris)

Steps: any int

Runs: any int

Versions: 0, 1, 2 (only applicable for PPO and DQN)

Options: any int for OC or PPOC, 2 or 4 for AOC

Testing: True or False

Test Eps: any int

Version 0 of PPO is PPO with n_epochs=10, batch_size=32 and N=2048

Version 1 of PPO is PPO with n_epochs=10, batch_size=32 and N=2048

Version 2 of PPO is PPO with shared actor-critic networks (in other versions they are spearate)

Version 0 of DQN has only a policy network

Version 1 of DQN has a policy and target network which updates every 200 time steps

Version 2 of DQN has a policy and target network which does soft updates every time step

Steps is how many time steps to train for. Runs is how many rounds of training to do (done simultaneous through multiprocessing)

testing=True runs a trained agent for testeps number of episodes, rendering the environment. If testing please ensure to only change testing to True and use all the same arguments as used for training.

For example, if I wanted to run DQN for 200,000 timesteps, 3 runs, and version 1, in the CartPole environment, I would run:

```
python main.py --environment CartPole --agent DQN --steps 200000 --runs 3 --version 1
```

If I wanted to test 5 episodes of an AOC agent I have trained for 150,000 timesteps in 8x4 Tetris for 2 options, I would run:

```
python main.py --environment Tetris --agent AOC --boardsize 8x4 --steps 150,000 --options 2 --testing True --testeps 5
```

## Requirements

```
pytorch>=1.12.1
gym>=0.15.3
gym_simpletetris (adapted version) which can be downloaded from https://github.com/sabina0009/gym-simpletetris
```
